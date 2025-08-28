import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
from collections import deque
import numpy as np

from env import Env
from agent import Agent
from utils import *  # 若有原 make_gif，可继续保留；这里会提供 make_gif_safe
from node_manager import NodeManager
from ground_truth_node_manager import GroundTruthNodeManager
from parameter import *
import math
from rendezvous_picker import pick_rendezvous_point


if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def make_gif_safe(frame_paths, out_path, duration_ms=120):
    """
    将 frame_paths 合成 GIF，强制统一尺寸并转为 RGB，避免花屏/受损。
    """
    # 过滤不存在的帧，并排序
    frame_paths = [p for p in frame_paths if os.path.exists(p)]
    frame_paths.sort()
    if len(frame_paths) == 0:
        print("[gif] no frames to write:", out_path)
        return

    frames = []
    base_size = None

    for p in frame_paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[gif] skip broken frame {p}: {e}")
            continue

        if base_size is None:
            base_size = im.size
        elif im.size != base_size:
            im = im.resize(base_size, Image.BILINEAR)
        frames.append(im)

    if len(frames) == 0:
        print("[gif] no valid frames after filtering:", out_path)
        return

    # 注意：optimize=False，可避免调色板过度优化导致某些“花屏”
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False
    )

class RegroupScheduler:
    def __init__(self):
        self.active = False
        self.P = None
        self.r_meet = 0.0
        self.T_meet = -1
        self.meta = None
        self.host_id = 0
        self.t0 = 0                   # 新增：建约时的 episode 步号

    def propose(self, P, T_meet, r_meet, meta=None, host_id=0, start_step=0):
        self.active  = True
        self.P       = np.array(P, dtype=float)
        self.T_meet  = int(T_meet)    # 绝对步号（相对于 episode 的 t）
        self.r_meet  = float(r_meet)
        self.meta    = meta if isinstance(meta, dict) else {}
        self.host_id = int(host_id)
        self.t0      = int(start_step)

    def clear(self):
        self.active = False
        self.P = None
        self.T_meet = -1
        self.r_meet = 0.0
        self.meta = None
        self.t0 = 0


class Worker:
    def __init__(self, meta_agent_id, policy_net, predictor, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.node_manager = NodeManager(plot=self.save_image)

        self.robots = [
            Agent(i, policy_net, predictor, self.node_manager, device=self.device, plot=self.save_image)
            for i in range(N_AGENTS)
        ]
        self.gtnm = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info,
                                           device=self.device, plot=self.save_image)

        self.episode_buffer = [[] for _ in range(27)]
        self.perf_metrics = dict()

        # —— intent/位置可见性缓存（每个体有一份视角） —— #
        self.last_known_locations = [self.env.robot_locations.copy() for _ in range(N_AGENTS)]
        self.last_known_intents = [{aid: [] for aid in range(N_AGENTS)} for _ in range(N_AGENTS)]

        # —— 每个 worker/episode 独占帧目录 —— #
        self.run_dir = os.path.join(
            gifs_path,
            f"run_g{self.global_step}_w{self.meta_agent_id}_{os.getpid()}_{int(time.time()*1000)}"
        )
        if self.save_image:
            os.makedirs(self.run_dir, exist_ok=True)
        # 清空本轮帧列表
        self.env.frame_files = []
        
        self.debug_rdv = None

        # === RDV 可视化状态 ===
        self.debug_rdv = None          # 当前用于绘图的 RDV (center_xy, r_meet, meta) 或 None
        self._rdv_last_step = -10**9   # 上一次成功刷新 RDV 的步号
        self._rdv_cache = None         # 最近一次成功的 RDV 结果
        self._rdv_cache_ttl = 0        # 失败时沿用缓存的剩余帧数

        # === RDV 刷新策略（仅可视化用，不影响动作）===
        self.rdv_cfg = dict(
            rdv_period=4,          # 至少每多少步刷新一次
            trigger_frac=0.72,     # 当最大两两距离 > 0.72*COMMS_RANGE 时立刻刷新
            keep_after_fail=6,     # 失败时沿用缓存 RDV 的帧数
            # 传给 rendezvous_picker 的参数（H_max_meter 若为 None 则按地图对角线自适应）
            params=dict(
                H_max_meter=None,          # 例如 0.85*map_diag
                r_meet_frac=0.45,
                H_post_meter=24.0,
                sync_tol_meter=18.0,
                target_frac=0.70,
                cand_opts=dict(tau_free=0.58, near_frontier_only=False, max_band_m=9999.0)
            )
        )

        self._rdv_cache = None
        self._rdv_cache_ttl = 0

        self._rdv_paths_per_agent = [set() for _ in range(N_AGENTS)]
        
        self.scheduler = RegroupScheduler()

    # === critic 输入维度补齐：把 7 维补到 NODE_INPUT_DIM(=7+N_AGENTS) ===
    def _match_intent_channels(self, obs_pack):
        """
        把 critic 的 node_inputs 从 7 维补到 NODE_INPUT_DIM（7 + N_AGENTS），
        其余张量原样返回。
        """
        n, m, e, ci, ce, ep = obs_pack  # node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask
        need = NODE_INPUT_DIM
        got = n.shape[-1]
        if got < need:
            pad = torch.zeros((n.shape[0], n.shape[1], need - got), dtype=n.dtype, device=n.device)
            n = torch.cat([n, pad], dim=-1)
        elif got > need:
            n = n[..., :need]  # 容错截断
        return [n, m, e, ci, ce, ep]

    def run_episode(self):

        done = False

        # ========== 初始化：图与预测 ==========
        for i, r in enumerate(self.robots):
            map_i = self.env.get_agent_map(i)
            r.update_graph(map_i, self.env.robot_locations[i])

        for r in self.robots:
            r.update_predict_map()

        # ========== 初始化：每个体“视角”下的队友位置/意图缓存 ==========
        self.last_known_locations = [self.env.robot_locations.copy() for _ in range(N_AGENTS)]
        self.last_known_intents   = [{aid: [] for aid in range(N_AGENTS)} for _ in range(N_AGENTS)]
        for i, r in enumerate(self.robots):
            r.update_planning_state(self.env.robot_locations)
            # 自己知道自己的意图
            self.last_known_intents[r.id][r.id] = deepcopy(r.intent_seq)

        # ---- 局部门控：是否需要会合（断联 + 短期不会自然相遇）----
        def should_rdv(T_natural_meter: float = 0.40 * COMMS_RANGE) -> bool:
            groups = self.env._compute_comm_groups()
            if len(groups) <= 1:
                return False  # 已连通

            # 估计“是否很快自然相遇”：看跨组最近两人的相向性 + 距离是否已不远
            locs = [np.array(r.location, dtype=float) for r in self.robots]

            # 方向：用各自的 intent 末端指向
            headings = []
            for r in self.robots:
                path = r.intent_seq if isinstance(r.intent_seq, list) else []
                if len(path) >= 1:
                    v = np.array(path[-1], dtype=float) - np.array(r.location, dtype=float)
                    n = np.linalg.norm(v)
                    h = v / n if n > 1e-6 else None
                else:
                    h = None
                headings.append(h)

            # 标注组别
            gid = {}
            for gi, g in enumerate(groups):
                for idx in g:
                    gid[idx] = gi

            min_cross = 1e9
            face_ok = False
            for i in range(N_AGENTS):
                for j in range(i + 1, N_AGENTS):
                    if gid[i] == gid[j]:
                        continue
                    d = float(np.linalg.norm(locs[i] - locs[j]))
                    if d < min_cross:
                        min_cross = d
                        # 是否相向
                        vij = locs[j] - locs[i]
                        nij = np.linalg.norm(vij)
                        if nij > 1e-6 and (headings[i] is not None) and (headings[j] is not None):
                            ldir = vij / nij
                            cos_i = float(np.dot(headings[i],  ldir))
                            cos_j = float(np.dot(headings[j], -ldir))
                            face_ok = (cos_i > 0.4 and cos_j > 0.4)  # 阈值可调
                        else:
                            face_ok = False

            # 距离不远且相向 -> 让其自然相遇
            if (min_cross <= T_natural_meter) and face_ok:
                return False
            return True

        # 若要首帧，可在此调用一次 plot_env（仅在需要会合时刷新 RDV）
        if self.save_image:
            if should_rdv():
                # 更保守的参数：目标均距靠近些；其余用 __init__ 里的 rdv_cfg 节流策略
                self.rdv_cfg['params']['target_frac'] = 0.55     # ← 更保守，不追特别远的点
                self.rdv_cfg['params']['r_meet_frac'] = 0.45
                # 触发刷新（带节流/缓存）
                self._refresh_debug_rdv_if_needed(step=0)
            else:
                self.debug_rdv = None
                self._rdv_cache_ttl = 0

            # 基于当前 RDV 计算每个体的 RDV 最短路节点集合（仅断链时非空）
            if hasattr(self, "_update_rdv_paths_per_agent"):
                self._update_rdv_paths_per_agent()
                # 把集合灌到各 agent（观测时会作为 rdv_path 通道）
                for i, ag in enumerate(self.robots):
                    ag.rdv_path_nodes_set = getattr(self, "_rdv_paths_per_agent", [set()]*N_AGENTS)[i]
            self.plot_env(step=0)

        # ========== 主循环 ==========
        for t in range(MAX_EPISODE_STEP):
            picks, dists, chosen_idx, chosen_act = [], [], [], []

            # —— 选动作前：把上一步准备好的 RDV 路径集合灌到 agent（训练/评估都生效）——
            if hasattr(self, "_rdv_paths_per_agent"):
                for i, ag in enumerate(self.robots):
                    ag.rdv_path_nodes_set = self._rdv_paths_per_agent[i]

            # ★★★ 新增：给每个体注入 time_left_norm（0~1，越接近 0 越紧迫） ★★★
            if self.scheduler.active:
                T_deadline = self.scheduler.T_meet + int(MEET_LATE_TOL)
                total_win  = max(1, T_deadline - self.scheduler.t0)
                time_left  = max(0, T_deadline - t)
                tl_norm    = float(np.clip(time_left / total_win, 0.0, 1.0))
            else:
                tl_norm = 0.0
            for ag in self.robots:
                ag.time_left_norm = tl_norm
            # ★★★ 新增结束 ★★★

            # ---- 3) 选动作（每个体用自己“视角”的位置 + intents） ----
            for i, r in enumerate(self.robots):
                obs = r.get_observation(
                    robot_locations=self.last_known_locations[i],
                    global_intents=self.last_known_intents[i]
                )
                # critic obs（GT视角），补齐 intent 通道至 NODE_INPUT_DIM
                c_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, robot_locations=self.env.robot_locations
                )
                c_obs = self._match_intent_channels(c_obs)

                r.save_observation(obs, c_obs)

                nxt, node_idx, act = r.select_next_waypoint(obs)
                r.save_action(act)

                node = r.node_manager.nodes_dict.find((r.location[0], r.location[1]))
                neigh = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
                assert nxt[0] + nxt[1] * 1j in (neigh[:, 0] + neigh[:, 1] * 1j), \
                    f"next {nxt} not neighbor of {r.location}"
                assert not (np.allclose(nxt[0], r.location[0]) and np.allclose(nxt[1], r.location[1])), \
                    "picked same node"

                picks.append(nxt)
                dists.append(np.linalg.norm(nxt - r.location))
                chosen_idx.append(node_idx)
                chosen_act.append(act)

            # ---- 4) 冲突消解 ----
            picks = self.resolve_conflicts(picks, dists)

            # —— 记录“本步前”的 max 与 total —— #
            prev_max   = self.env.get_agent_travel().max()
            prev_total = self.env.get_total_travel()

            # ---- 5) 执行动作 ----
            for r, loc in zip(self.robots, picks):
                curr_node = r.node_manager.nodes_dict.find((r.location[0], r.location[1])).data
                neigh = np.array(list(curr_node.neighbor_set)).reshape(-1, 2)
                if not np.any((neigh[:, 0] + 1j * neigh[:, 1]) == (loc[0] + 1j * loc[1])):
                    loc = r.location.copy()  # 不合法则等待
                self.env.step(loc, r.id)

            # —— 计算 Δmax 与 Δtotal，并更新 env.max_travel_dist —— #
            curr_agent_travel = self.env.get_agent_travel()
            new_max = curr_agent_travel.max()
            delta_max   = new_max - prev_max
            delta_total = self.env.get_total_travel() - prev_total
            self.env.max_travel_dist = new_max

            # ---- 5.5) 通信分组内：同步“可见”的位置与意图（仅组内互通）----
            groups = self.env._compute_comm_groups()
            for g in groups:
                for i in g:
                    for j in g:
                        self.last_known_locations[i][j] = self.env.robot_locations[j].copy()
                        self.last_known_intents[i][j]   = deepcopy(self.robots[j].intent_seq)

            # ---- 6) 合并后：更新图、预测 & 让每个体基于“自己视角”重算意图 ----
            for i, r in enumerate(self.robots):
                r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
                r.update_predict_map()
                r.update_planning_state(
                    robot_locations=self.last_known_locations[i],
                    intents_view=self.last_known_intents[i]
                )
                self.last_known_intents[i][r.id] = deepcopy(r.intent_seq)

            # —— 6.5) 会合门控 + 刷新 RDV（仅在需要会合时触发，可视化/引导）—— #
            if should_rdv():
                # 保守策略：将 RDV 目标均距收敛些
                self.rdv_cfg['params']['target_frac'] = 0.55
                self._refresh_debug_rdv_if_needed(step=t + 1)
            else:
                self.debug_rdv = None
                self._rdv_cache_ttl = 0

            # —— 计算通信分组 —— 
            groups = self.env._compute_comm_groups()
            fully_connected = (len(groups) == 1)

            # —— 若无合约且当前连通：在未知区+预测可行上显式建约 —— 
            if (not self.scheduler.active) and fully_connected:
                try:
                    # 注意：picker 返回等待步数 T_wait（而不是绝对步号）
                    P, r_meet, T_wait, meta = pick_rendezvous_point(
                        self,
                        H_max_meter=None,                    # 内部会自适应地图对角线
                        r_meet_frac=MEET_RADIUS_FRAC,
                        H_post_meter=24.0,
                        sync_tol_meter=MEET_SYNC_TOL_M,
                        target_frac=0.60,
                        cand_opts=dict(tau_free=0.58, near_frontier_only=False, max_band_m=9999.0)
                    )
                    if P is not None:
                        # 用当前 t 转为绝对截止步，并记录 start_step=t
                        self.scheduler.propose(P, t + int(T_wait), r_meet, meta=meta, host_id=0, start_step=t)
                        # 供可视化：把 debug_rdv 对齐到当前合约
                        self.debug_rdv = (self.scheduler.P.copy(), self.scheduler.r_meet, self.scheduler.meta or {})
                except Exception:
                    pass

            # ======== 6.6 刷新 RDV 最短路节点集合（断链时才非空） ========
            if hasattr(self, "_update_rdv_paths_per_agent"):
                self._update_rdv_paths_per_agent()

            # ---- 7) 奖励：个体 utility + 团队 frontier - 惩罚（Δmax/Δtotal） ----
            indiv_rewards = []
            for i, (r, picked) in enumerate(zip(self.robots, picks)):
                try:
                    idx = np.argwhere(
                        r.node_coords[:, 0] + r.node_coords[:, 1] * 1j == picked[0] + picked[1] * 1j
                    )[0][0]
                except Exception:
                    nn = self.node_manager.nodes_dict.nearest_neighbors(picked.tolist(), 1)[0].data.coords
                    idx = np.argwhere(
                        r.node_coords[:, 0] + r.node_coords[:, 1] * 1j == nn[0] + nn[1] * 1j
                    )[0][0]
                utility_term = r.utility[idx] / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
                indiv_rewards.append(float(utility_term))

            team_reward = self.env.calculate_reward() - 0.5
            max_penalty   = (delta_max   / UPDATING_MAP_SIZE) * MAX_TRAVEL_COEF
            total_penalty = (delta_total / UPDATING_MAP_SIZE) * TOTAL_TRAVEL_COEF
            team_reward -= (max_penalty + total_penalty)

            # —— 显式会合成功/过期（一次性） —— 
            if self.scheduler.active:
                # 可视化也跟随
                self.debug_rdv = (self.scheduler.P.copy(), self.scheduler.r_meet, self.scheduler.meta or {})

                # 成功：全网连通 且 所有机器人都进入会合半径
                fully_connected = (len(self.env._compute_comm_groups()) == 1)
                if fully_connected:
                    ok_all = True
                    for r in self.robots:
                        if float(np.linalg.norm(np.array(r.location, dtype=float) - self.scheduler.P)) > self.scheduler.r_meet:
                            ok_all = False; break
                    if ok_all:
                        team_reward += float(R_MEET_SUCCESS)
                        self.scheduler.clear()

                # 过期：超过 T_meet + 宽容
                if self.scheduler.active:
                    if (t >= self.scheduler.T_meet + int(MEET_LATE_TOL)):
                        team_reward -= float(R_MEET_LATE)
                        self.scheduler.clear()

            # ---- 8) 终止 ----
            utilities_empty = all([(r.utility <= 0).all() for r in self.robots])
            done = utilities_empty or (self.env.explored_rate > 0.995)
            if done:
                team_reward += 10.0

            # ---- 9) 写入缓冲 & next_obs（各自视角） ----
            for i, r in enumerate(self.robots):
                r.save_reward(indiv_rewards[i] + team_reward)
                r.save_done(done)

                # next obs 前再次把 rdv_path_nodes_set 灌好（与上面一致）
                if hasattr(self, "_rdv_paths_per_agent"):
                    r.rdv_path_nodes_set = self._rdv_paths_per_agent[i]

                next_obs = r.get_observation(
                    robot_locations=self.last_known_locations[i],
                    global_intents=self.last_known_intents[i]
                )
                c_next_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, robot_locations=self.env.robot_locations
                )
                c_next_obs = self._match_intent_channels(c_next_obs)
                r.save_next_observations(next_obs, c_next_obs)

            if self.save_image:
                self.plot_env(step=t + 1)

            if done:
                break

        # ========== 10) 指标 ==========
        self.perf_metrics['travel_dist']   = self.env.get_total_travel()
        self.perf_metrics['max_travel']    = self.env.get_max_travel()
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate']  = done

        # 用“安全 GIF 合成”，只读自己 run_dir 的帧
        if self.save_image:
            gif_out = os.path.join(self.run_dir, f"episode_{self.global_step}_w{self.meta_agent_id}.gif")
            make_gif_safe(self.env.frame_files, gif_out, duration_ms=120)

        # 归并 episode buffer
        for r in self.robots:
            for k in range(len(self.episode_buffer)):
                self.episode_buffer[k] += r.episode_buffer[k]



    def resolve_conflicts(self, picks, dists):
        picks = np.array(picks).reshape(-1, 2)
        order = np.argsort(np.array(dists))
        chosen_complex = set()
        resolved = [None] * len(self.robots)

        for rid in order:
            robot = self.robots[rid]
            target = picks[rid]

            curr_node = robot.node_manager.nodes_dict.find((robot.location[0], robot.location[1])).data
            neighbor_coords = [np.array(c, dtype=float) for c in list(curr_node.neighbor_set)]
            neighbor_coords_sorted = sorted(neighbor_coords, key=lambda c: np.linalg.norm(c - target))

            picked = None
            for cand in neighbor_coords_sorted:
                key = complex(cand[0], cand[1])
                if key not in chosen_complex:
                    picked = cand
                    break
            if picked is None:
                picked = robot.location.copy()

            resolved[rid] = picked
            chosen_complex.add(complex(picked[0], picked[1]))

        return np.array(resolved).reshape(-1, 2)

    def _pairwise_max_distance(self):
        """当前 N_AGENTS 机器人间的最大两两距离（米）"""
        locs = self.env.robot_locations
        m = 0.0
        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                d = float(np.linalg.norm(locs[i] - locs[j]))
                if d > m:
                    m = d
        return m

    def _refresh_debug_rdv_if_needed(self, step):
        """
        仅用于可视化：按“节流 + 触发”策略刷新 self.debug_rdv。
        失败时短期沿用上一帧的 RDV，避免 GIF 上忽隐忽现。
        统一把 rendezvous_picker 的返回标准化为 (center_xy, r_meet, meta) 三元组。
        """
        # 不存图就不要算（省事）
        if not self.save_image:
            return

        # —— 刷新触发：到期 or 即将断链 ——
        trig_period = (step - self._rdv_last_step) >= int(self.rdv_cfg['rdv_period'])
        trig_break  = self._pairwise_max_distance() > float(self.rdv_cfg['trigger_frac']) * float(COMMS_RANGE)
        if not (trig_period or trig_break):
            # 没触发：如有缓存 TTL，继续沿用；否则保持当前（可能是 None）
            if self._rdv_cache_ttl > 0 and self._rdv_cache is not None:
                self.debug_rdv = self._rdv_cache
                self._rdv_cache_ttl -= 1
            return

        # —— 计算 H_max 的自适应上限（地图对角线的 85%） ——
        params = dict(self.rdv_cfg['params'])  # 浅拷贝
        if params.get('H_max_meter') is None:
            H, W = self.env.global_belief.shape
            map_diag = math.hypot(H * CELL_SIZE, W * CELL_SIZE)
            params['H_max_meter'] = 0.85 * map_diag

        # —— 调 rendezvous_picker —— 
        try:
            rdv = pick_rendezvous_point(self, **params)
            # rendezvous_picker 可能返回 (P, r_meet, meta) 或 (P, r_meet, T_wait, meta)
            # 我们统一成 (P, r_meet, meta)
            center_xy, r_meet, meta = None, None, {}
            if isinstance(rdv, tuple):
                if len(rdv) >= 4:
                    center_xy, r_meet, _Twait_unused, meta = rdv[0], rdv[1], rdv[2], rdv[3]
                elif len(rdv) == 3:
                    center_xy, r_meet, meta = rdv
                elif len(rdv) == 2:
                    center_xy, r_meet = rdv
                    meta = {}
            if center_xy is not None:
                rdv_vis = (np.array(center_xy, dtype=float), float(r_meet), meta if isinstance(meta, dict) else {})
            else:
                rdv_vis = None

            # 成功：更新当前、缓存、步号、TTL
            self.debug_rdv = rdv_vis
            self._rdv_cache = rdv_vis
            self._rdv_cache_ttl = int(self.rdv_cfg['keep_after_fail'])
            self._rdv_last_step = step
        except Exception:
            # 失败：尽量用缓存兜底
            if self._rdv_cache is not None and self._rdv_cache_ttl > 0:
                self.debug_rdv = self._rdv_cache
                self._rdv_cache_ttl -= 1
            else:
                self.debug_rdv = None



    def _bfs_path_rc(self, trav_mask, start_rc, goal_rc):
        H, W = trav_mask.shape
        sr, sc = start_rc; gr, gc = goal_rc
        if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
            return []
        if (not trav_mask[sr, sc]) or (not trav_mask[gr, gc]):
            return []
        prev = -np.ones((H, W, 2), dtype=np.int32)
        q = deque()
        q.append((sr, sc))
        prev[sr, sc] = (sr, sc)
        while q:
            r, c = q.popleft()
            if r == gr and c == gc:
                break
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and trav_mask[nr, nc] and prev[nr, nc, 0] == -1:
                    prev[nr, nc] = (r, c)
                    q.append((nr, nc))
        if prev[gr, gc, 0] == -1:  # unreachable
            return []
        # 回溯
        path = []
        r, c = gr, gc
        while not (r == sr and c == sc):
            path.append((r, c))
            r, c = int(prev[r, c, 0]), int(prev[r, c, 1])
        path.append((sr, sc))
        path.reverse()
        return path

    def _cells_to_nodecoord_set(self, rc_path, map_info):
        if not rc_path:
            return set()
        xs = map_info.map_origin_x + np.array([c for _, c in rc_path], dtype=float) * map_info.cell_size
        ys = map_info.map_origin_y + np.array([r for r, _ in rc_path], dtype=float) * map_info.cell_size
        pts = np.stack([xs, ys], axis=1)
        # 直接四舍五入到 0.1，后续与 node_coords 的 (round(x,1), round(y,1)) 比较
        pts = np.around(pts, 1)
        # 也可做降采样：每隔若干格取一个
        return set(map(lambda p: (float(p[0]), float(p[1])), pts.reshape(-1,2)))
    
    def _update_rdv_paths_per_agent(self):
        # 无合约 -> 清空
        if not getattr(self.scheduler, 'active', False) or self.scheduler.P is None:
            self._rdv_paths_per_agent = [set() for _ in range(N_AGENTS)]
            for a in self.robots:
                a.rdv_path_nodes_set = set()
            return

        a0 = self.robots[0]
        if a0.pred_mean_map_info is None:
            return

        pred = a0.pred_mean_map_info.map.astype(np.float32)
        belief = a0.map_info.map
        p_free = pred / float(FREE)

        # 与 rendezvous_picker 相同的可行网格逻辑
        tau = 0.60
        traversable = ((p_free >= tau) | (belief == FREE)) & (belief != OCCUPIED)

        # 通信图是否连通
        fully_connected = (len(self.env._compute_comm_groups()) == 1)
        if fully_connected:
            # 连通时不引导（避免压抑探索）
            self._rdv_paths_per_agent = [set() for _ in range(N_AGENTS)]
            for a in self.robots:
                a.rdv_path_nodes_set = set()
            return

        # 目标 cell（来自合约的 P）
        center_xy = self.scheduler.P
        gr, gc = self._world_to_cell_rc(center_xy, a0.map_info)

        new_sets = []
        for r in self.robots:
            sr, sc = self._world_to_cell_rc(r.location, a0.map_info)

            # 起点/终点若在不可行上，找最近可行点
            if not (0 <= sr < traversable.shape[0] and 0 <= sc < traversable.shape[1]) or not traversable[sr, sc]:
                sr, sc = self._nearest_free_rc(traversable, sr, sc)
            rr, cc = gr, gc
            if not (0 <= rr < traversable.shape[0] and 0 <= cc < traversable.shape[1]) or not traversable[rr, cc]:
                rr, cc = self._nearest_free_rc(traversable, rr, cc)

            # BFS 最短路
            rc_path = self._bfs_path_rc(traversable, (sr, sc), (rr, cc))
            node_set = self._cells_to_nodecoord_set(rc_path, a0.map_info)
            new_sets.append(node_set)

        self._rdv_paths_per_agent = new_sets
        for i, ag in enumerate(self.robots):
            ag.rdv_path_nodes_set = self._rdv_paths_per_agent[i]


    # —— 小工具（与 rendezvous_picker 内部一致）——
    def _world_to_cell_rc(self, world_xy, map_info: MapInfo):
        cell = get_cell_position_from_coords(np.array(world_xy, dtype=float), map_info)
        return int(cell[1]), int(cell[0])

    def _nearest_free_rc(self, trav, r, c):
        H, W = trav.shape
        if 0 <= r < H and 0 <= c < W and trav[r, c]:
            return r, c
        for rad in range(1, 16):
            r0, r1 = max(0, r - rad), min(H - 1, r + rad)
            c0, c1 = max(0, c - rad), min(W - 1, c + rad)
            found = np.argwhere(trav[r0:r1 + 1, c0:c1 + 1])
            if found.size > 0:
                rr, cc = found[0]
                return r0 + int(rr), c0 + int(cc)
        return r, c  # 实在不行就原位

    def _should_rendezvous(self, T_natural_meter: float = 0.45 * COMMS_RANGE) -> bool:
        """
        True -> 需要主动会合；False -> 顺其自然（不强迫 RDV）
        逻辑：
        1) 当前通信图是否已连通？已连通则 False
        2) 若断联：估计是否“很快自然重连”
        - 用当前 last_known_intents 做个粗略 ETA：各组边界最近两人，若相向且最近
            距离 <= T_natural_meter，就判定会很快自然相遇 -> False
        """
        groups = self.env._compute_comm_groups()
        if len(groups) <= 1:
            return False  # 已连通，不需要

        # —— 估计自然相遇 —— #
        # 获取每个 agent 的“下一段意图方向”向量
        headings = []
        for i, r in enumerate(self.robots):
            path = r.intent_seq if isinstance(r.intent_seq, list) else []
            if len(path) >= 1:
                v = np.array(path[-1], dtype=float) - np.array(r.location, dtype=float)
                n = np.linalg.norm(v)
                h = v / n if n > 1e-6 else None
            else:
                h = None
            headings.append(h)

        # 只检查“跨组”的最近两人
        locs = [np.array(r.location, dtype=float) for r in self.robots]
        min_cross = 1e9
        will_face = False
        # 组编号映射
        group_id = {}
        for gi, g in enumerate(groups):
            for idx in g:
                group_id[idx] = gi

        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                if group_id[i] == group_id[j]:
                    continue
                d = float(np.linalg.norm(locs[i] - locs[j]))
                if d < min_cross:
                    min_cross = d
                    # 是否相向（两个方向向量都存在，且与连线方向夹角小）
                    vij = (locs[j] - locs[i])
                    nij = np.linalg.norm(vij)
                    if nij > 1e-6 and (headings[i] is not None) and (headings[j] is not None):
                        ldir = vij / nij
                        cos_i = float(np.dot(headings[i],  ldir))
                        cos_j = float(np.dot(headings[j], -ldir))
                        will_face = (cos_i > 0.4 and cos_j > 0.4)  # 阈值可调
                    else:
                        will_face = False

        # 距离本身足够近，且“看起来在相向”，就判定会自然相遇
        if (min_cross <= T_natural_meter) and will_face:
            return False

        return True  # 断联且短期不会自然相遇 -> 需要会合

    def plot_env(self, step):
        """
        v2: 左大右小的双栏布局（全局 + 每个体局部），更清晰且适合做 GIF。
        重点改动：
        - 更稳定的版式（GridSpec），抗抖动；
        - 统一深色底面，轨迹/连线/标注用高对比色；
        - RDV 显著（红点+白描边+虚线圈），并可叠加每体的 RDV 最短路引导；
        - HUD 面板：step、explored、travel、分组、mission 倒计时等；
        - 右侧每个体小图：局部观测 + 自己/队友位置 + intents + 通信圈（只画一次圆）；
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib import patheffects as pe
        from matplotlib.gridspec import GridSpec

        plt.switch_backend('agg')

        # --------- 配色与风格 ----------
        mpl.rcParams.update({
            "figure.facecolor": "#0d0f12",
            "axes.facecolor": "#0d0f12",
            "savefig.facecolor": "#0d0f12",
            "axes.edgecolor":  "#181b20",
            "axes.linewidth":   1.2,
            "font.size":        11,
            "xtick.color":      "#a8b3c7",
            "ytick.color":      "#a8b3c7",
            "text.color":       "#e6edf3",
        })
        # 颜色（与 tab10 相近，但更亮）
        agent_colors = [
            "#4cc9f0", "#f72585", "#43aa8b", "#ffca3a", "#b5179e", "#4895ef",
            "#ff924c", "#90be6d", "#577590", "#f94144"
        ]
        def ac(i): return agent_colors[i % len(agent_colors)]

        # --------- 布局：左大右小 ----------
        rows = max(N_AGENTS, 3)
        fig = plt.figure(figsize=(12, 7), dpi=150)
        gs  = GridSpec(nrows=rows, ncols=2, width_ratios=[3.2, 1.2], wspace=0.06, hspace=0.05, figure=fig)
        ax_global = fig.add_subplot(gs[:, 0])  # 左侧全局：占满所有行
        ax_locals = [fig.add_subplot(gs[i, 1]) for i in range(rows)]  # 右侧每行一个

        # =============== 左侧：全局 ===============
        gb = self.env.global_belief
        ax_global.imshow(gb, cmap='gray', vmin=0, vmax=255, zorder=0)
        ax_global.set_title("Global • map & comm & RDV", pad=8)

        # UNKNOWN 轻微绿罩（便于看 RDV 是否落在未知带）
        try:
            unknown_mask = (gb == UNKNOWN).astype(float)
            ax_global.imshow(unknown_mask, cmap='Greens', alpha=0.16, zorder=1)
        except Exception:
            pass

        ax_global.set_axis_off()

        # --- 轨迹与当前位置 ---
        for r in self.robots:
            col = ac(r.id)
            try:
                # 全局轨迹
                if hasattr(r, 'trajectory_x') and len(r.trajectory_x) > 1:
                    xs = (np.array(r.trajectory_x) - r.map_info.map_origin_x) / r.cell_size
                    ys = (np.array(r.trajectory_y) - r.map_info.map_origin_y) / r.cell_size
                    ax_global.plot(xs, ys, color=col, linewidth=1.8, alpha=0.95, zorder=2)
                # 当前位置
                cell = get_cell_position_from_coords(r.location, r.map_info)
                ax_global.plot(cell[0], cell[1], marker='o', markersize=7.5,
                            markerfacecolor=col, markeredgecolor='white',
                            markeredgewidth=1.2, zorder=6)
            except Exception:
                pass

        # --- 通信用：半径圈 + 组内连线 & 组号 ---
        try:
            # 通信分组
            groups = [sorted(list(g)) for g in self.env._compute_comm_groups()]
            # 组号与连线
            for gi, g in enumerate(groups):
                # 组内连线
                for i in range(len(g)):
                    for j in range(i + 1, len(g)):
                        try:
                            r1, r2 = self.robots[g[i]], self.robots[g[j]]
                            p1 = get_cell_position_from_coords(r1.location, r1.map_info)
                            p2 = get_cell_position_from_coords(r2.location, r2.map_info)
                            ax_global.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                        color="#66ffb3", linewidth=1.5, alpha=0.7, zorder=3)
                        except Exception:
                            pass
                # 组号标注
                for idx in g:
                    try:
                        r = self.robots[idx]
                        rc = get_cell_position_from_coords(r.location, r.map_info)
                        txt = ax_global.text(rc[0]+3, rc[1]+3, f"G{gi}",
                                            fontsize=9, color='white', zorder=7)
                        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="#0d0f12")])
                    except Exception:
                        pass

            # 通信半径圈（描边淡一些）
            for r in self.robots:
                try:
                    rc = get_cell_position_from_coords(r.location, r.map_info)
                    r_pix = float(COMMS_RANGE) / float(r.cell_size)
                    circ = patches.Circle((rc[0], rc[1]), r_pix, fill=False,
                                        linestyle=':', linewidth=1.1,
                                        edgecolor='#8ef0d0', alpha=0.35, zorder=2.5)
                    ax_global.add_patch(circ)
                except Exception:
                    pass
        except Exception:
            pass

        # --- RDV：中心 + 半径圈 + 文本 ---
        try:
            if getattr(self, 'debug_rdv', None) is not None and self.debug_rdv is not None:
                if isinstance(self.debug_rdv, tuple) and len(self.debug_rdv) >= 2 and self.debug_rdv[0] is not None:
                    center_xy, r_meet, meta = self.debug_rdv
                    cc = get_cell_position_from_coords(center_xy, self.robots[0].map_info)
                    H, W = gb.shape
                    if 0 <= cc[0] < W and 0 <= cc[1] < H:
                        # 中心点
                        ax_global.plot(cc[0], cc[1], marker='o', markersize=10,
                                    markerfacecolor='#ff4d4f', markeredgecolor='white',
                                    markeredgewidth=2.0, zorder=12)
                        # 半径圈
                        r_pix = np.clip(float(r_meet) / float(self.robots[0].cell_size), 6.0, 400.0)
                        ring = patches.Circle((cc[0], cc[1]), r_pix, fill=False, linestyle='--',
                                            linewidth=2.0, edgecolor='#ff4d4f', alpha=0.95, zorder=11)
                        ax_global.add_patch(ring)
                        # 文本（得分/同步）
                        if isinstance(meta, dict):
                            s_total = float(meta.get('total', 0.0))
                            t_sp = meta.get('t_spread', None)
                            jain = meta.get('jain', None)
                            lines = [f"RDV  S={s_total:.2f}"]
                            if t_sp is not None: lines.append(f"ΔT={float(t_sp):.1f}m")
                            if jain is not None: lines.append(f"J={float(jain):.2f}")
                            lbl = "\n".join(lines)
                        else:
                            lbl = "RDV"
                        t = ax_global.text(cc[0] + 4, cc[1] + 4, lbl, fontsize=10, zorder=13)
                        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="#0d0f12")])
                    else:
                        t = ax_global.text(8, 18, "RDV off-canvas", color='#ff4d4f', fontsize=10, zorder=13)
                        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="#0d0f12")])
        except Exception:
            pass

        # --- 每个体的 RDV 引导路径点（只在断联时展示） ---
        try:
            for r in self.robots:
                if hasattr(r, 'rdv_path_nodes_set') and r.rdv_path_nodes_set:
                    pts = np.array(sorted(list(r.rdv_path_nodes_set))).reshape(-1, 2)
                    cells = get_cell_position_from_coords(pts, r.map_info).reshape(-1, 2)
                    ax_global.scatter(cells[:, 0], cells[:, 1],
                                    s=10, marker='s', c=ac(r.id), alpha=0.9, zorder=8, linewidths=0)
        except Exception:
            pass

        # --- HUD：关键信息叠加 ---
        try:
            explored = getattr(self.env, 'explored_rate', 0.0)
            tot = self.env.get_total_travel()
            mx  = self.env.get_max_travel()
            groups = self.env._compute_comm_groups()
            gnum = len(groups)
            hud_lines = [
                f"Step {step}/{MAX_EPISODE_STEP}",
                f"Explored: {explored*100:.1f}%",
                f"Travel  total={tot:.1f}  max={mx:.1f}",
                f"Comm groups: {gnum}",
            ]
            if self.scheduler.active:
                T_deadline = self.scheduler.T_meet + int(MEET_LATE_TOL)
                time_left  = max(0, T_deadline - step)
                hud_lines.append(f"Mission: ACTIVE  T_left={time_left}  r={self.scheduler.r_meet:.1f}")
            else:
                hud_lines.append("Mission: -")
            hud = "\n".join(hud_lines)
            txt = ax_global.text(10, 18, hud, fontsize=10, color="#e6edf3", va='top', ha='left', zorder=20,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor=(0,0,0,0.35), edgecolor="#3a3f45"))
            txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="#0d0f12")])
        except Exception:
            pass

        # =============== 右侧：每个体的局部 ===============
        for i in range(N_AGENTS):
            r = self.robots[i]
            ax = ax_locals[i]
            ax.set_axis_off()
            try:
                obs_map = r.updating_map_info.map
                ax.imshow(obs_map, cmap='gray', vmin=0, vmax=255, zorder=0)
                # 自己
                rc2 = get_cell_position_from_coords(r.location, r.updating_map_info)
                ax.plot(rc2[0], rc2[1], 'o', markersize=7, zorder=6,
                        markerfacecolor=ac(r.id), markeredgecolor='white', markeredgewidth=1.2, label='self')
                # 通信圈（只画 viewer 的）
                try:
                    r_pix_local = float(COMMS_RANGE) / float(r.cell_size)
                    Hloc, Wloc = obs_map.shape
                    if 0 <= rc2[0] < Wloc and 0 <= rc2[1] < Hloc:
                        circ_l = patches.Circle((rc2[0], rc2[1]), r_pix_local, fill=False,
                                                linestyle=':', linewidth=1.0,
                                                edgecolor='#8ef0d0', alpha=0.45, zorder=4)
                        ax.add_patch(circ_l)
                except Exception:
                    pass

                # 已知队友位置（可通信加粗）
                connected_ids = set()
                try:
                    for g in self.env._compute_comm_groups():
                        if i in g:
                            connected_ids = set(g) - {i}
                            break
                except Exception:
                    connected_ids = set()
                if hasattr(self, 'last_known_locations'):
                    known_positions = self.last_known_locations[i]
                    for aid in range(N_AGENTS):
                        pos = known_positions[aid]
                        try:
                            pc = get_cell_position_from_coords(pos, r.updating_map_info)
                            is_conn = (aid in connected_ids)
                            ax.plot(pc[0], pc[1], 'o',
                                    markersize=(6.5 if is_conn else 5.0),
                                    markerfacecolor=ac(aid), markeredgecolor='white',
                                    markeredgewidth=(1.5 if is_conn else 0.8),
                                    zorder=5)
                        except Exception:
                            pass

                # intents（带起点拼接）
                if hasattr(self, 'last_known_intents'):
                    intents_view = self.last_known_intents[i]
                    for aid, path in intents_view.items():
                        if not path: 
                            continue
                        path_cells = []
                        try:
                            if aid == r.id:
                                start_cell = get_cell_position_from_coords(r.location, r.updating_map_info)
                                path_cells.append(start_cell)
                            else:
                                pos_aid = self.last_known_locations[i][aid]
                                start_cell = get_cell_position_from_coords(pos_aid, r.updating_map_info)
                                path_cells.append(start_cell)
                            for p in path:
                                cell = get_cell_position_from_coords(np.array(p, dtype=float), r.updating_map_info)
                                path_cells.append(cell)
                        except Exception:
                            path_cells = []
                        if len(path_cells) > 0:
                            ix = [c[0] for c in path_cells]; iy = [c[1] for c in path_cells]
                            ax.plot(ix, iy, linestyle=('--' if aid==r.id else ':'), marker='x',
                                    linewidth=1.2, markersize=4.5, color=ac(aid), alpha=0.95, zorder=6)

                # RDV 引导路径（若有）
                try:
                    if hasattr(r, 'rdv_path_nodes_set') and r.rdv_path_nodes_set:
                        pts = np.array(sorted(list(r.rdv_path_nodes_set))).reshape(-1, 2)
                        cells = []
                        for p in pts:
                            try:
                                cells.append(get_cell_position_from_coords(p, r.updating_map_info))
                            except Exception:
                                pass
                        if len(cells) > 0:
                            cells = np.array(cells).reshape(-1, 2)
                            ax.scatter(cells[:, 0], cells[:, 1], s=12, marker='s',
                                    c=ac(r.id), alpha=0.95, zorder=7, linewidths=0)
                except Exception:
                    pass

                ax.set_title(f"Agent {r.id}", fontsize=10, pad=3)
            except Exception as e:
                ax.text(0.5, 0.5, f'Local plot err:\n{e}', ha='center', va='center', fontsize=8)

        # 多出来的空行（当 rows>N_AGENTS）隐藏
        for j in range(N_AGENTS, rows):
            ax_locals[j].set_visible(False)

        # 固定像素输出，稳定 GIF 帧
        out_path = os.path.join(self.run_dir, f"t{step:04d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        self.env.frame_files.append(out_path)
