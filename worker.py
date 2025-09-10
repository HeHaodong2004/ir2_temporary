# worker.py
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
from collections import deque
import heapq
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from env import Env
from agent import Agent
from utils import *
from node_manager import NodeManager
from ground_truth_node_manager import GroundTruthNodeManager
from parameter import *
import math

try:
    from attention_viz import dump_attn_debug_pngs, AttnRecorder
except Exception:
    dump_attn_debug_pngs, AttnRecorder = None, None

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

def make_gif_safe(frame_paths, out_path, duration_ms=120):
    frame_paths = [p for p in frame_paths if os.path.exists(p)]
    frame_paths.sort()
    if not frame_paths: return
    frames = []
    base_size = None
    for p in frame_paths:
        try:
            im = Image.open(p).convert("RGB")
            if base_size is None: base_size = im.size
            elif im.size != base_size: im = im.resize(base_size, Image.BILINEAR)
            frames.append(im)
        except Exception:
            continue
    if not frames: return
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


# -------------------- 显式合同结构 --------------------
class Contract:
    """
    Rendezvous 合同（全队）
    P: 中心点（世界坐标）
    r: 区域半径（米）
    t_min, t_max: 到达时间窗
    participants: 参与的 agent id 集合
    created_t: 合同创建步
    meta: 包含 CC-ODS 的 T_tar 与每个机器人 t_dep、q_i 等
    status: 'active' | 'done' | 'failed'
    """
    def __init__(self, P, r, t_min, t_max, participants, created_t, meta=None):
        self.P = np.array(P, dtype=float)
        self.r = float(r)
        self.t_min = int(t_min)
        self.t_max = int(t_max)
        self.participants = set(participants)
        self.created_t = int(created_t)
        self.meta = meta if isinstance(meta, dict) else {}
        self.status = 'active'

    def within_region(self, pos_xy):
        return np.linalg.norm(np.asarray(pos_xy, dtype=float) - self.P) <= self.r


# -------------------- D* Lite 实现（4-邻接） --------------------
class PriorityQueue:
    def __init__(self): self.data = []
    def push(self, k, s): heapq.heappush(self.data, (k, s))
    def pop(self): return heapq.heappop(self.data)
    def top_key(self):
        return (float('inf'), float('inf')) if not self.data else self.data[0][0]
    def empty(self): return len(self.data) == 0
    def remove(self, s):
        self.data = [(k, x) for (k, x) in self.data if x != s]
        heapq.heapify(self.data)

class DStarLite:
    """
    最小实现，参考 Koenig & Likhachev (2002)。
    - 4-邻接
    - 成本来自 cost_map[r,c]（>=1）
    - OCCUPIED/不可行在外面过滤：传入的 cost_map 对不可行位置给 np.inf
    """
    def __init__(self, cost_map, start_rc, goal_rc, heuristic=lambda a,b: (abs(a[0]-b[0])+abs(a[1]-b[1]))):
        self.H, self.W = cost_map.shape
        self.cmap = cost_map
        self.s_start = tuple(start_rc)
        self.s_goal = tuple(goal_rc)
        self.rhs = {}
        self.g = {}
        self.U = PriorityQueue()
        self.km = 0.0
        self.h = heuristic
        self.s_last = self.s_start
        for r in range(self.H):
            for c in range(self.W):
                self.g[(r,c)] = float('inf')
                self.rhs[(r,c)] = float('inf')
        self.rhs[self.s_goal] = 0.0
        self.U.push(self._calc_key(self.s_goal), self.s_goal)

    def _calc_key(self, s):
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + self.h(self.s_start, s) + self.km, g_rhs)

    def _neighbors(self, s):
        r, c = s
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.H and 0 <= nc < self.W:
                if np.isfinite(self.cmap[nr, nc]):
                    yield (nr, nc)

    def _cost(self, a, b):
        return self.cmap[b[0], b[1]]

    def update_vertex(self, s):
        if s != self.s_goal:
            self.rhs[s] = min([self.g[n] + self._cost(s, n) for n in self._neighbors(s)] + [float('inf')])
        in_U = False
        for k, x in self.U.data:
            if x == s:
                in_U = True
                break
        if self.g[s] != self.rhs[s]:
            if in_U: self.U.remove(s)
            self.U.push(self._calc_key(s), s)
        else:
            if in_U: self.U.remove(s)

    def compute_shortest_path(self, max_expand=100000):
        cnt = 0
        while (self.U.top_key() < self._calc_key(self.s_start)) or (self.rhs[self.s_start] != self.g[self.s_start]):
            if cnt >= max_expand: break
            cnt += 1
            k_old, u = self.U.pop()
            k_new = self._calc_key(u)
            if k_old < k_new:
                self.U.push(k_new, u)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for n in self._neighbors(u):
                    self.update_vertex(n)
            else:
                self.g[u] = float('inf')
                for n in list(self._neighbors(u)) + [u]:
                    self.update_vertex(n)

    def update_start(self, new_start):
        hdiff = self.h(self.s_last, new_start)
        self.km += hdiff
        self.s_start = tuple(new_start)
        self.s_last = tuple(new_start)

    def update_cost_map(self, new_cost_map, changed_cells=None):
        self.cmap = new_cost_map
        if changed_cells is None:
            iters = [(r,c) for r in range(self.H) for c in range(self.W) if np.isfinite(self.cmap[r,c])]
        else:
            iters = set()
            for (r,c) in changed_cells:
                iters.add((r,c))
                for n in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
                    if 0 <= n[0] < self.H and 0 <= n[1] < self.W and np.isfinite(self.cmap[n[0],n[1]]):
                        iters.add(n)
        for s in iters:
            self.update_vertex(s)

    def next_step_on_policy(self):
        s = self.s_start
        best_n, best_v = s, float('inf')
        for n in self._neighbors(s):
            v = self.g[n] + self._cost(s, n)
            if v < best_v:
                best_v, best_n = v, n
        return best_n if best_n != s else s

    def extract_path(self, max_len=100000):
        path = [self.s_start]
        cur = self.s_start
        seen = set([cur])
        for _ in range(max_len):
            nxt = self.next_step_on_policy()
            if nxt == cur: break
            path.append(nxt)
            if nxt in seen: break
            seen.add(nxt)
            cur = nxt
            if cur == self.s_goal: break
        return path


class Worker:
    def __init__(self, meta_agent_id, policy_net, predictor, global_step, device='cpu', save_image=False, attn_recorder: AttnRecorder=None):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.node_manager = NodeManager(plot=self.save_image)
        self.robots = [Agent(i, policy_net, predictor, self.node_manager, device=device, plot=save_image) for i in range(N_AGENTS)]
        self.gtnm = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info, device=device, plot=save_image)

        self.episode_buffer = [[] for _ in range(27)]
        self.perf_metrics = dict()

        self.last_known_locations = [self.env.robot_locations.copy() for _ in range(N_AGENTS)]
        self.last_known_intents = [{aid: [] for aid in range(N_AGENTS)} for _ in range(N_AGENTS)]

        self.run_dir = os.path.join(gifs_path, f"run_g{self.global_step}_w{self.meta_agent_id}_{os.getpid()}_{int(time.time()*1000)}")
        if self.save_image: os.makedirs(self.run_dir, exist_ok=True)
        self.env.frame_files = []

        # Rendezvous
        self.contract: Contract = None
        self.candidate_buffer = []  # [{'P','score','etas','risk','ig_total','t_min','t_max'}...]
        self.cand_last_update_t = -1

        # D* Lite 规划器（返程后才用）
        self._planners = [None] * N_AGENTS
        self._planner_goal = [None] * N_AGENTS

        self.was_fully_connected = False
        self.attn_recorder = attn_recorder

        gt_map = self.env.ground_truth_info.map
        self._gt_free_total = int(np.count_nonzero(gt_map == FREE))

    # ---------- critic 通道对齐 ----------
    def _match_intent_channels(self, obs_pack):
        n, m, e, ci, ce, ep = obs_pack
        need, got = NODE_INPUT_DIM, n.shape[-1]
        if got < need:
            pad = torch.zeros((n.shape[0], n.shape[1], need - got), dtype=n.dtype, device=n.device)
            n = torch.cat([n, pad], dim=-1)
        return [n, m, e, ci, ce, ep]

    # ========================= 主循环 =========================
    def run_episode(self):
        done = False

        # 初始化：图、预测、意图
        for i, r in enumerate(self.robots):
            r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
        for r in self.robots:
            r.update_predict_map()
        for i, r in enumerate(self.robots):
            r.update_planning_state(self.env.robot_locations)
            self.last_known_intents[r.id][r.id] = deepcopy(r.intent_seq)

        groups0 = self._compute_groups_from_positions(self.env.robot_locations)
        self.was_fully_connected = (len(groups0) == 1 and len(groups0[0]) == N_AGENTS)

        if self.save_image:
            self.plot_env(step=0)

        # ================= 主循环 =================
        for t in range(MAX_EPISODE_STEP):
            # 全局地图/掩膜
            global_map_info = MapInfo(self.env.global_belief,
                                      self.env.belief_origin_x,
                                      self.env.belief_origin_y,
                                      self.env.cell_size)
            belief_map = global_map_info.map
            if self.robots[0].pred_mean_map_info is not None:
                p_free = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
            else:
                p_free = (belief_map == FREE).astype(np.float32)
            trav_mask = (((p_free) >= RDV_TAU_FREE) | (belief_map == FREE)) & (belief_map != OCCUPIED)
            unknown_mask = (belief_map == UNKNOWN)

            # 计算当前连通分量
            groups = self._compute_groups_from_positions(self.env.robot_locations)
            is_fully_connected = (len(groups) == 1 and len(groups[0]) == N_AGENTS)

            # 只在“全连通 & 无合同 & 降频”时更新候选池
            if RDV_ONLY_WHEN_FULLY_CONNECTED and is_fully_connected and self.contract is None and (t % RDV_CAND_UPDATE_EVERY == 0):
                try:
                    self._update_candidate_buffer(global_map_info, trav_mask, unknown_mask, p_free, t)
                except Exception as e:
                    print(f"[RDV] candidate update failed at t={t}: {e}")

            # ---------- 原策略先给“自由探索”的下一步 ----------
            picks_raw, dists = [], []
            for i, r in enumerate(self.robots):
                obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_observation(obs, self._match_intent_channels(c_obs))

                if RDV_USE_ATTENTION_VIZ and (self.attn_recorder is not None) and self.save_image:
                    self.attn_recorder.begin_step(t)

                nxt, _, act = r.select_next_waypoint(obs)

                if RDV_USE_ATTENTION_VIZ and (self.attn_recorder is not None) and self.save_image:
                    attn_records = self.attn_recorder.end_forward()
                    try:
                        _, _, _, current_index_t, current_edge, _ = obs
                        current_index = int(current_index_t.item())
                        neighbor_indices = current_edge[0, :, 0].detach().cpu().numpy()
                    except Exception:
                        current_index = 0
                        neighbor_indices = (r.neighbor_indices
                                            if r.neighbor_indices is not None
                                            else np.array([], dtype=int))
                    try:
                        dump_attn_debug_pngs(
                            run_dir=self.run_dir,
                            step=t,
                            agent_id=r.id,
                            records=attn_records,
                            node_coords=r.node_coords,
                            current_index=current_index,
                            neighbor_indices=neighbor_indices,
                            map_info=r.map_info
                        )
                    except Exception as e:
                        print(f"[AttnViz] Dump failed at t={t}, agent={r.id}: {e}")

                r.save_action(act)
                picks_raw.append(nxt)
                dists.append(np.linalg.norm(nxt - r.location))

            # ---------- 分离预测 & 当步建约（全连通 + 即将分离 + 无合同） ----------
            cur_groups = self._compute_groups_from_positions(self.env.robot_locations)
            next_positions = [p.copy() for p in picks_raw]
            next_groups = self._compute_groups_from_positions(next_positions)

            if self.contract is None and is_fully_connected and len(next_groups) > len(cur_groups):
                best = self._select_best_candidate_from_buffer()
                if best is not None:
                    P_c = best['P']
                    r_meet = RDV_REGION_FRAC * COMMS_RANGE
                    sched = self._chance_constrained_schedule(P_c, r_meet, t, global_map_info, belief_map, p_free)
                    if sched is not None:
                        T_tar, dep_times, q_i, eps_used = sched
                        self.contract = Contract(
                            P=P_c,
                            r=r_meet,
                            t_min=best['t_min'],
                            t_max=best['t_max'],
                            participants=set(range(N_AGENTS)),
                            created_t=t,
                            meta={'score': best['score'], 'ig': best['ig_total'], 'risk': best['risk'],
                                  'T_tar': T_tar, 't_dep': dep_times, 'q_i': q_i, 'eps': eps_used}
                        )
                        self._planners = [None] * N_AGENTS
                        self._planner_goal = [None] * N_AGENTS
                        if RDV_VERBOSE:
                            print(f"[RDV] Contract@t={t} center={P_c}, window=({best['t_min']},{best['t_max']}), "
                                  f"T_tar={T_tar}, t_dep={dep_times}")

            # ---------- 执行：出发前完全不干预；出发后强制 D* Lite ----------
            picks = []
            for i, r in enumerate(self.robots):
                if self.contract is None or self.contract.status != 'active':
                    picks.append(picks_raw[i])
                    continue

                t_dep_i = int(self.contract.meta['t_dep'][i])
                T_tar = int(self.contract.meta['T_tar'])

                # 已在合同区域内：圈内巡航
                if self.contract.within_region(r.location):
                    picks.append(self._in_zone_patrol_step(i, r, global_map_info))
                    continue

                # 出发前：完全不干预（注意 <=，确保出发当步仍自由探索）
                if t <= t_dep_i:
                    picks.append(picks_raw[i])
                    continue

                # 出发后：D* Lite 导航
                goal_rc = self._nearest_reachable_in_region(self.contract.P, self.contract.r, trav_mask, global_map_info)
                if goal_rc is None:
                    picks.append(picks_raw[i])  # 暂不可达，等待回退逻辑
                    continue

                cost_map = self._build_cost_map(belief_map, p_free)
                start_rc = self._world_to_cell_rc(r.location, global_map_info)

                if self._planners[i] is None or self._planner_goal[i] != tuple(goal_rc):
                    self._planners[i] = DStarLite(cost_map, start_rc, goal_rc)
                    self._planner_goal[i] = tuple(goal_rc)
                    # 首次进入返程：立即规划一次
                    self._planners[i].compute_shortest_path(max_expand=RDV_DSTAR_MAX_EXPAND)
                else:
                    self._planners[i].update_start(start_rc)
                    if (t - t_dep_i) % RDV_COSTMAP_UPDATE_EVERY == 0:
                        self._planners[i].update_cost_map(cost_map, changed_cells=None)
                    if (t - t_dep_i) % RDV_PLAN_REPLAN_EVERY == 0:
                        self._planners[i].compute_shortest_path(max_expand=RDV_DSTAR_MAX_EXPAND)

                nxt_rc = self._planners[i].next_step_on_policy()
                nxt_xy = np.array([
                    global_map_info.map_origin_x + nxt_rc[1] * global_map_info.cell_size,
                    global_map_info.map_origin_y + nxt_rc[0] * global_map_info.cell_size
                ], dtype=float)
                picks.append(nxt_xy)

            # ---------- 冲突消解 & 推进 ----------
            picks = self.resolve_conflicts(picks, dists)
            prev_max = self.env.get_agent_travel().max()
            prev_total = self.env.get_total_travel()
            for r, loc in zip(self.robots, picks):
                self.env.step(loc, r.id)
            self.env.max_travel_dist = self.env.get_agent_travel().max()
            delta_max = self.env.max_travel_dist - prev_max
            delta_total = self.env.get_total_travel() - prev_total

            # 通信同步
            groups_after_move = self._compute_groups_from_positions(self.env.robot_locations)
            for g in groups_after_move:
                for i in g:
                    for j in g:
                        self.last_known_locations[i][j] = self.env.robot_locations[j].copy()
                        self.last_known_intents[i][j] = deepcopy(self.robots[j].intent_seq)

            # 图/预测/意图更新
            for i, r in enumerate(self.robots):
                r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
                r.update_predict_map()
                r.update_planning_state(self.last_known_locations[i], self.last_known_intents[i])
                self.last_known_intents[i][r.id] = deepcopy(r.intent_seq)

            # 合同进度与回退（降频检查）
            if self.contract is not None and self.contract.status == 'active' and (t % RDV_PROGRESS_CHECK_EVERY == 0):
                try:
                    self._contract_progress_and_fallback(t, global_map_info, trav_mask, unknown_mask, p_free)
                except Exception as e:
                    print(f"[RDV] progress/fallback error at t={t}: {e}")

            # 奖励与终止（保持你的逻辑）
            team_reward_env, per_agent_obs_rewards = self.env.calculate_reward()
            team_reward = (
                team_reward_env
                - ((delta_max / UPDATING_MAP_SIZE) * MAX_TRAVEL_COEF)
                - ((delta_total / UPDATING_MAP_SIZE) * TOTAL_TRAVEL_COEF)
            )

            utilities_empty = all((r.utility <= 3).all() for r in self.robots)
            if self._gt_free_total > 0:
                per_agent_free_counts = [int(np.count_nonzero(r.map_info.map == FREE)) for r in self.robots]
                per_agent_cov = [c / self._gt_free_total for c in per_agent_free_counts]
                coverage_ok = all(c >= 0.995 for c in per_agent_cov)
            else:
                coverage_ok = False

            done = utilities_empty or coverage_ok
            if done:
                team_reward += 10.0

            for i, r in enumerate(self.robots):
                indiv_total = team_reward + per_agent_obs_rewards[i]
                r.save_reward(indiv_total)
                r.save_done(done)
                next_obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_next_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_next_observations(next_obs, self._match_intent_channels(c_next_obs))

            if self.save_image:
                self.plot_env(step=t + 1)

            if done:
                break

        # 总结
        self.perf_metrics.update({
            'travel_dist': self.env.get_total_travel(),
            'max_travel': self.env.get_max_travel(),
            'explored_rate': self.env.explored_rate,
            'success_rate': done
        })

        if self.save_image:
            make_gif_safe(self.env.frame_files,
                          os.path.join(self.run_dir, f"ep_{self.global_step}.gif"))

        for r in self.robots:
            for k in range(len(self.episode_buffer)):
                self.episode_buffer[k] += r.episode_buffer[k]

    # ---------------- 冲突消解（保留你的实现） ----------------
    def resolve_conflicts(self, picks, dists):
        picks = np.array(picks).reshape(-1, 2)
        order = np.argsort(np.array(dists))
        chosen_complex, resolved = set(), [None] * len(self.robots)
        for rid in order:
            robot = self.robots[rid]
            try:
                neighbor_coords = sorted(
                    list(robot.node_manager.nodes_dict.find(robot.location.tolist()).data.neighbor_set),
                    key=lambda c: np.linalg.norm(np.array(c) - picks[rid])
                )
            except Exception:
                neighbor_coords = [robot.location.copy()]
            picked = next((cand for cand in neighbor_coords if complex(cand[0], cand[1]) not in chosen_complex), None)
            resolved[rid] = np.array(picked) if picked is not None else robot.location.copy()
            chosen_complex.add(complex(resolved[rid][0], resolved[rid][1]))
        return np.array(resolved).reshape(-1, 2)

    # ======================= 候选池（选点） =======================
    def _update_candidate_buffer(self, map_info: MapInfo, trav_mask, unknown_mask, p_free, t_now: int):
        H, W = trav_mask.shape
        cell_size = float(map_info.cell_size)

        r_idx, c_idx = np.where(unknown_mask & trav_mask)
        if r_idx.size == 0:
            self.candidate_buffer = []
            self.cand_last_update_t = t_now
            return

        # 随机下采样
        n_cand = min(r_idx.size, RDV_CAND_K)
        sel = np.random.choice(r_idx.size, n_cand, replace=False)
        rc = np.stack([r_idx[sel], c_idx[sel]], axis=1)

        # 稀疏化
        if RDV_CAND_STRIDE > 1:
            keep, seen = [], set()
            for r, c in rc:
                key = (int(r // RDV_CAND_STRIDE), int(c // RDV_CAND_STRIDE))
                if key in seen: continue
                seen.add(key); keep.append((r, c))
            rc = np.array(keep, dtype=int)

        # BFS 距离图
        dist_maps = []
        for agent in self.robots:
            start_rc = self._world_to_cell_rc(agent.location, map_info)
            if not trav_mask[start_rc[0], start_rc[1]]:
                start_rc = self._find_nearest_valid_cell(trav_mask, np.array(start_rc))
            dist_maps.append(self._bfs_dist_map(trav_mask, tuple(start_rc)))

        candidates = []
        R_info_pix = int(RDV_INFO_RADIUS_M / cell_size)
        for (r_c, c_c) in rc:
            r0 = max(0, r_c - R_info_pix); r1 = min(H, r_c + R_info_pix + 1)
            c0 = max(0, c_c - R_info_pix); c1 = min(W, c_c + R_info_pix + 1)
            local_ig = int(unknown_mask[r0:r1, c0:c1].sum())

            etas, risks, path_ig = [], [], []
            feasible = True
            for j, agent in enumerate(self.robots):
                d_steps = dist_maps[j][r_c, c_c]
                if not np.isfinite(d_steps):
                    feasible = False; break
                eta_j = d_steps / max(NODE_RESOLUTION, 1e-6)
                etas.append(float(eta_j))
                r_s, c_s = self._world_to_cell_rc(agent.location, map_info)
                line = self._bresenham_line_rc(r_s, c_s, r_c, c_c)
                line_risk, line_ig = 0.0, 0
                for (rr, cc) in line:
                    if 0 <= rr < H and 0 <= cc < W:
                        line_risk += float(1.0 - p_free[rr, cc])
                        if unknown_mask[rr, cc]: line_ig += 1
                risks.append(line_risk / max(len(line), 1))
                path_ig.append(line_ig)
            if not feasible: continue

            ig_total = float(RDV_ALPHA * (sum(path_ig) + local_ig))
            disp = float(max(etas) - min(etas))
            risk_total = float(sum(risks))
            score = ig_total - RDV_BETA * disp - RDV_GAMMA * risk_total

            eta_max = max(etas)
            t_mid = t_now + int(round(eta_max))
            t_min = t_mid - int(round(RDV_WINDOW_ALPHA_EARLY * eta_max + RDV_WINDOW_BETA_EARLY))
            t_max = t_mid + int(round(RDV_WINDOW_ALPHA_LATE  * eta_max + RDV_WINDOW_BETA_LATE))

            P_world = np.array([map_info.map_origin_x + c_c * cell_size,
                                map_info.map_origin_y + r_c * cell_size], dtype=float)
            candidates.append({
                'P': P_world, 'score': score, 'etas': etas, 'risk': risk_total, 'ig_total': ig_total,
                't_min': t_min, 't_max': t_max
            })

        candidates.sort(key=lambda d: d['score'], reverse=True)
        self.candidate_buffer = candidates[:RDV_TOP_M]
        self.cand_last_update_t = t_now

    def _select_best_candidate_from_buffer(self):
        return self.candidate_buffer[0] if self.candidate_buffer else None

    # ======================= CC-ODS：机会约束调度 =======================
    def _chance_constrained_schedule(self, P, r, t_now, map_info, belief_map, p_free):
        """
        返回 (T_tar, dep_times[aid], q_i[aid], eps_used) 或 None (不可行)
        方法：
          - 采样 RDV_TT_N_SAMPLES 次“可通行图”，对每个 agent 求到“区域内最近可达点”的 BFS 距离 -> 时间
          - 取 (1-RDV_EPSILON) 分位数为 q_i
          - T_tar = clamp(max_i(t_now + q_i), [t_min, t_max]) —— t_min/t_max 用候选最优的窗口
          - dep_i = T_tar - q_i，并至少保留 RDV_MIN_LEAD_STEPS 的自由探索
        """
        H, W = belief_map.shape
        cell_size = float(map_info.cell_size)

        def nearest_goal_rc(trav_mask_s):
            r_pix = int(max(1, round(r / cell_size)))
            c_rc = self._world_to_cell_rc(P, map_info)
            best, bestd = None, float('inf')
            for rr in range(max(0, c_rc[0]-r_pix), min(H, c_rc[0]+r_pix+1)):
                for cc in range(max(0, c_rc[1]-r_pix), min(W, c_rc[1]+r_pix+1)):
                    if not trav_mask_s[rr, cc]: continue
                    d = (rr - c_rc[0])**2 + (cc - c_rc[1])**2
                    if d < bestd:
                        bestd, best = d, (rr, cc)
            return best

        # 采样可通行图
        samples = []
        for _ in range(RDV_TT_N_SAMPLES):
            rand = np.random.rand(H, W).astype(np.float32)
            trav_s = (((rand < p_free) | (belief_map == FREE)) & (belief_map != OCCUPIED))
            samples.append(trav_s)

        # 样本旅行时间
        T_samples = {aid: [] for aid in range(N_AGENTS)}
        for s in range(RDV_TT_N_SAMPLES):
            trav_s = samples[s]
            goal_rc = nearest_goal_rc(trav_s)
            if goal_rc is None:
                for aid in range(N_AGENTS):
                    T_samples[aid].append(float('inf'))
                continue
            dist_maps = []
            for aid in range(N_AGENTS):
                start_rc = self._world_to_cell_rc(self.robots[aid].location, map_info)
                if not trav_s[start_rc[0], start_rc[1]]:
                    start_rc = self._find_nearest_valid_cell(trav_s, np.array(start_rc))
                dist_maps.append(self._bfs_dist_map(trav_s, tuple(start_rc)))
            for aid in range(N_AGENTS):
                d = dist_maps[aid][goal_rc[0], goal_rc[1]]
                T_samples[aid].append(float(d / max(NODE_RESOLUTION, 1e-6)) if np.isfinite(d) else float('inf'))

        # q_i
        q_i = []
        for aid in range(N_AGENTS):
            arr = np.array(T_samples[aid], dtype=np.float32)
            feasible_mask = np.isfinite(arr)
            if feasible_mask.sum() < max(1, int((1.0 - RDV_EPSILON) * RDV_TT_N_SAMPLES)):
                return None
            vals = np.sort(arr[feasible_mask])
            k = max(0, int(math.ceil((1.0 - RDV_EPSILON) * len(vals))) - 1)
            q_i.append(float(vals[k]))

        # 候选窗口
        if not self.candidate_buffer:
            return None
        t_min_cand = int(self.candidate_buffer[0]['t_min'])
        t_max_cand = int(self.candidate_buffer[0]['t_max'])

        # 共同目标到达时刻
        T_tar = min(t_max_cand, max(t_min_cand, max([t_now + qi for qi in q_i])))

        # 出发时刻（至少保留最小提前量）
        dep_times = [int(math.floor(T_tar - qi)) for qi in q_i]
        dep_times = [max(t_now + RDV_MIN_LEAD_STEPS, d) for d in dep_times]
        T_tar = max(dep_times[i] + q_i[i] for i in range(N_AGENTS))
        if T_tar > t_max_cand:
            return None
        return int(T_tar), [int(d) for d in dep_times], q_i, RDV_EPSILON

    # ======================= D* Lite 相关辅助 =======================
    def _build_cost_map(self, belief_map, p_free):
        """
        cost = 1 + λ_risk*(1-p_free) + ε_known
        - OCCUPIED -> inf
        - 未知格子更“便宜”（不加 ε_known）
        """
        H, W = belief_map.shape
        cost = np.ones((H, W), dtype=np.float32)
        risk = RDV_RISK_LAMBDA * (1.0 - p_free)
        cost += risk
        known = (belief_map != UNKNOWN) & (belief_map != OCCUPIED)
        cost += RDV_INFO_EPS * known.astype(np.float32)
        cost[belief_map == OCCUPIED] = np.inf
        return cost

    def _nearest_reachable_in_region(self, P, r, trav_mask, map_info):
        H, W = trav_mask.shape
        r_pix = int(max(1, round(r / map_info.cell_size)))
        c_rc = self._world_to_cell_rc(P, map_info)
        best, bestd = None, float('inf')
        for rr in range(max(0, c_rc[0]-r_pix), min(H, c_rc[0]+r_pix+1)):
            for cc in range(max(0, c_rc[1]-r_pix), min(W, c_rc[1]+r_pix+1)):
                if not trav_mask[rr, cc]: continue
                d = (rr - c_rc[0])**2 + (cc - c_rc[1])**2
                if d < bestd:
                    bestd, best = d, (rr, cc)
        return best

    def _in_zone_patrol_step(self, aid, agent, map_info):
        """
        圈内巡航：挑选一个不离开圈的邻居，优先前沿/信息。
        """
        try:
            node = self.node_manager.nodes_dict.find(agent.location.tolist()).data
            neighbor_coords = list(node.neighbor_set)
        except Exception:
            neighbor_coords = [agent.location.copy()]

        def inside(xy): return self.contract.within_region(xy)

        best, best_score = agent.location.copy(), -1e18
        for nb in neighbor_coords:
            if not inside(nb): continue
            s_frontier = 0.0
            try:
                nd = self.node_manager.nodes_dict.find(np.around(nb, 1).tolist()).data
                s_frontier = float(max(nd.utility, 0.0))
            except Exception:
                pass
            rr, cc = self._world_to_cell_rc(nb, map_info)
            pfree = (self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
                     if self.robots[0].pred_mean_map_info is not None else 1.0)
            risk = 1.0
            if isinstance(pfree, np.ndarray):
                if 0 <= rr < pfree.shape[0] and 0 <= cc < pfree.shape[1]:
                    risk = 1.0 - float(pfree[rr, cc])
            score = s_frontier - RDV_RISK_LAMBDA * risk
            if score > best_score:
                best_score, best = score, np.array(nb, dtype=float)
        return best

    # ======================= 合同进度与回退 =======================
    def _contract_progress_and_fallback(self, t_now, map_info, trav_mask, unknown_mask, p_free):
        if self.contract is None or self.contract.status != 'active':
            return

        # 完成判定
        all_in = True
        for aid in self.contract.participants:
            if not self.contract.within_region(self.robots[aid].location):
                all_in = False
                break
        if all_in:
            self.contract.status = 'done'
            if RDV_VERBOSE:
                print(f"[RDV] DONE@t={t_now}")
            return

        # 超过 t_max + 宽限 或 中心不可达/有人很远：回退
        if (t_now > self.contract.t_max + RDV_LATE_TOL) or True:
            goal_rc_center = self._world_to_cell_rc(self.contract.P, map_info)
            center_reachable = trav_mask[goal_rc_center[0], goal_rc_center[1]]
            eta_far = False
            for aid in self.contract.participants:
                eta_steps, _ = self._eta_steps_and_path(trav_mask, map_info, self.robots[aid].location, self.contract.P)
                if eta_steps > RDV_LATE_EXPAND_STEPS:
                    eta_far = True; break

            if (not center_reachable) or eta_far:
                factor = RDV_FAIL_EXPAND_FACTOR
                r_pix = int((self.contract.r * factor) / map_info.cell_size)
                c_xy = self.contract.P
                c_rc = self._world_to_cell_rc(c_xy, map_info)
                H, W = trav_mask.shape
                r0 = max(0, c_rc[0] - r_pix); r1 = min(H, c_rc[0] + r_pix + 1)
                c0 = max(0, c_rc[1] - r_pix); c1 = min(W, c_rc[1] + r_pix + 1)
                best_rc, best_score = None, -1e18
                for rr in range(r0, r1):
                    for cc in range(c0, c1):
                        if not trav_mask[rr, cc]: continue
                        loc_ig = int(unknown_mask[max(0, rr-2):min(H, rr+3), max(0, cc-2):min(W, cc+3)].sum())
                        d_pen = 0.01 * math.hypot(rr - c_rc[0], cc - c_rc[1])
                        sc = loc_ig - d_pen
                        if sc > best_score:
                            best_score, best_rc = sc, (rr, cc)
                if best_rc is not None:
                    self.contract.P = np.array([map_info.map_origin_x + best_rc[1] * map_info.cell_size,
                                                map_info.map_origin_y + best_rc[0] * map_info.cell_size])
                    self.contract.r *= factor
                    self.contract.meta['fallback'] = 'expand_region'
                    if RDV_VERBOSE:
                        print(f"[RDV] Fallback: expand region -> P={self.contract.P}, r={self.contract.r:.1f}")
                    return

                safe_rc = self._nearest_known_free_to(self.contract.P, map_info)
                if safe_rc is not None:
                    self.contract.P = np.array([map_info.map_origin_x + safe_rc[1] * map_info.cell_size,
                                                map_info.map_origin_y + safe_rc[0] * map_info.cell_size])
                    self.contract.r = max(self.contract.r, RDV_SAFE_HUB_RADIUS)
                    self.contract.meta['fallback'] = 'safe_hub'
                    if RDV_VERBOSE:
                        print(f"[RDV] Fallback: safe hub -> P={self.contract.P}, r={self.contract.r:.1f}")
                else:
                    self.contract.status = 'failed'
                    if RDV_VERBOSE:
                        print("[RDV] Fallback: FAILED (no feasible hub)")

    # ======================= 小工具 & 图/几何 =======================
    def _compute_groups_from_positions(self, positions):
        n = len(positions)
        if n == 0: return []
        used = [False] * n
        groups = []
        for i in range(n):
            if used[i]: continue
            comp, q = [], [i]
            used[i] = True
            while q:
                u = q.pop()
                comp.append(u)
                for v in range(n):
                    if used[v]: continue
                    if np.linalg.norm(np.asarray(positions[u]) - np.asarray(positions[v])) <= COMMS_RANGE + 1e-6:
                        used[v] = True; q.append(v)
            groups.append(tuple(sorted(comp)))
        return groups

    def _world_to_cell_rc(self, world_xy, map_info):
        cell = get_cell_position_from_coords(np.array(world_xy, dtype=float), map_info)
        return int(cell[1]), int(cell[0])

    def _find_nearest_valid_cell(self, mask, start_rc):
        q = deque([tuple(start_rc)]); visited = {tuple(start_rc)}
        H, W = mask.shape
        while q:
            r, c = q.popleft()
            if 0 <= r < H and 0 <= c < W and mask[r, c]:
                return np.array([r, c])
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    q.append((nr, nc)); visited.add((nr, nc))
        return np.asarray(start_rc)

    def _bfs_dist_map(self, trav_mask, start_rc):
        H, W = trav_mask.shape
        dist_map = np.full((H, W), np.inf, dtype=np.float32)
        q = deque([start_rc])
        if 0 <= start_rc[0] < H and 0 <= start_rc[1] < W and trav_mask[start_rc]:
            dist_map[start_rc[0], start_rc[1]] = 0.0
        while q:
            r, c = q.popleft()
            base = dist_map[r, c]
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and trav_mask[nr, nc] and np.isinf(dist_map[nr, nc]):
                    dist_map[nr, nc] = base + 1.0
                    q.append((nr, nc))
        return dist_map

    def _bfs_path_rc(self, trav_mask, start_rc, goal_rc):
        H, W = trav_mask.shape
        parent = {tuple(start_rc): None}
        q = deque([tuple(start_rc)])
        if not (0 <= start_rc[0] < H and 0 <= start_rc[1] < W and trav_mask[start_rc[0], start_rc[1]]):
            return []
        while q:
            r, c = q.popleft()
            if (r, c) == tuple(goal_rc):
                path = []
                cur = (r, c)
                while cur is not None:
                    path.append(cur); cur = parent[cur]
                path.reverse(); return path
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and trav_mask[nr, nc] and (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c); q.append((nr, nc))
        return []

    def _bresenham_line_rc(self, r0, c0, r1, c1):
        points = []
        dr, dc = abs(r1-r0), abs(c1-c0)
        sr, sc = (1 if r1>=r0 else -1), (1 if c1>=c0 else -1)
        if dc > dr:
            err, r = dc//2, r0
            for c in range(c0, c1+sc, sc):
                points.append((r, c)); err -= dr
                if err < 0: r += sr; err += dc
        else:
            err, c = dr//2, c0
            for r in range(r0, r1+sr, sr):
                points.append((r, c)); err -= dc
                if err < 0: c += sc; err += dr
        return points

    def _eta_steps_and_path(self, trav_mask, map_info, start_world_xy, goal_world_xy):
        start_rc = self._world_to_cell_rc(start_world_xy, map_info)
        goal_rc  = self._world_to_cell_rc(goal_world_xy, map_info)
        if not trav_mask[start_rc[0], start_rc[1]]:
            start_rc = self._find_nearest_valid_cell(trav_mask, start_rc)
        if not trav_mask[goal_rc[0], goal_rc[1]]:
            goal_rc = self._find_nearest_valid_cell(trav_mask, goal_rc)
        rc_path = self._bfs_path_rc(trav_mask, start_rc, goal_rc)
        if not rc_path: return int(1e9), []
        steps = len(rc_path) / max(NODE_RESOLUTION, 1e-6)
        return int(round(steps)), rc_path

    def _nearest_known_free_to(self, world_xy, map_info):
        rr, cc = self._world_to_cell_rc(world_xy, map_info)
        belief = map_info.map
        H, W = belief.shape
        q = deque([(rr, cc)]); vis = {(rr, cc)}
        while q:
            r, c = q.popleft()
            if 0 <= r < H and 0 <= c < W and belief[r, c] == FREE: return (r, c)
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in vis:
                    vis.add((nr, nc)); q.append((nr, nc))
        return None

    # ========================= 可视化 =========================
    def plot_env(self, step):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 10), dpi=110)

        gs = GridSpec(N_AGENTS, 3, figure=fig, width_ratios=[2.5, 1.2, 1.2], wspace=0.15, hspace=0.1)
        ax_global = fig.add_subplot(gs[:, 0])
        ax_locals_obs = [fig.add_subplot(gs[i, 1]) for i in range(N_AGENTS)]
        ax_locals_pred = [fig.add_subplot(gs[i, 2]) for i in range(N_AGENTS)]
        agent_colors = plt.cm.get_cmap('cool', N_AGENTS)

        global_info = MapInfo(self.env.global_belief, self.env.belief_origin_x, self.env.belief_origin_y, self.env.cell_size)
        ax_global.set_title(f"Global View | Step {step}/{MAX_EPISODE_STEP}", fontsize=14, pad=10)
        ax_global.imshow(global_info.map, cmap='gray', origin='lower')
        ax_global.set_aspect('equal', adjustable='box')
        ax_global.set_axis_off()

        if self.robots and self.robots[0].pred_mean_map_info is not None:
            pred_mean = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
            belief = global_info.map
            unknown_mask = (belief == UNKNOWN)
            prob = np.zeros_like(pred_mean)
            prob[unknown_mask] = pred_mean[unknown_mask]
            ax_global.imshow(prob, cmap='magma', origin='lower', alpha=0.35)

        # 通信边
        groups = self._compute_groups_from_positions(self.env.robot_locations)
        for group in groups:
            for i_idx, i in enumerate(list(group)):
                for j in list(group)[i_idx+1:]:
                    p1 = self._world_to_cell_rc(self.robots[i].location, global_info)
                    p2 = self._world_to_cell_rc(self.robots[j].location, global_info)
                    ax_global.plot([p1[1], p2[1]], [p1[0], p2[0]], color="#33ff88", lw=2, alpha=0.8, zorder=5)

        # 机器人、轨迹、通讯圈
        for i, r in enumerate(self.robots):
            pos_cell = self._world_to_cell_rc(r.location, global_info)
            if hasattr(r, 'trajectory_x') and r.trajectory_x:
                traj_cells = [self._world_to_cell_rc(np.array([x, y]), global_info)
                              for x, y in zip(r.trajectory_x, r.trajectory_y)]
                ax_global.plot([c for rr, c in traj_cells], [rr for rr, c in traj_cells],
                               color=agent_colors(i), lw=1.5, zorder=3)
            comms_radius = patches.Circle((pos_cell[1], pos_cell[0]), COMMS_RANGE / CELL_SIZE,
                                          fc=(0, 1, 0, 0.05), ec=(0, 1, 0, 0.4), ls='--', lw=1.5, zorder=4)
            ax_global.add_patch(comms_radius)
            ax_global.plot(pos_cell[1], pos_cell[0], 'o', ms=10, mfc=agent_colors(i),
                           mec='white', mew=1.5, zorder=10)

        # 合同区域/中心与文本
        if self.contract is not None and self.contract.status == 'active':
            p_cell = self._world_to_cell_rc(self.contract.P, global_info)
            ax_global.plot(p_cell[1], p_cell[0], '*', ms=22, mfc='yellow', mec='white', mew=2, zorder=12)
            radius = patches.Circle((p_cell[1], p_cell[0]), self.contract.r / CELL_SIZE,
                                    fc=(1, 1, 0, 0.07), ec='yellow', ls='--', lw=2.0, zorder=11)
            ax_global.add_patch(radius)
            ax_global.text(5, 5,
                           f"RDV window: [{self.contract.t_min}, {self.contract.t_max}]  "
                           f"T_tar: {self.contract.meta.get('T_tar','-')}",
                           fontsize=10, color='yellow', ha='left', va='top')

        # 本地视角
        for i, r in enumerate(self.robots):
            ax_obs = ax_locals_obs[i]
            local_map_info = r.map_info
            ax_obs.set_title(f"Agent {i} View", fontsize=10, pad=5)
            ax_obs.imshow(local_map_info.map, cmap='gray', origin='lower')
            ax_obs.set_aspect('equal', adjustable='box')
            pos_cell_local = self._world_to_cell_rc(r.location, local_map_info)
            ax_obs.plot(pos_cell_local[1], pos_cell_local[0], 'o', ms=8, mfc=agent_colors(i),
                        mec='white', mew=1.5, zorder=10)
            if r.intent_seq:
                intent_world = [r.location] + r.intent_seq
                intent_cells = [self._world_to_cell_rc(pos, local_map_info) for pos in intent_world]
                ax_obs.plot([c for rr, c in intent_cells], [rr for rr, c in intent_cells],
                            'x--', c=agent_colors(i), lw=2, ms=6, zorder=8)
            ax_obs.set_axis_off()

            ax_pred = ax_locals_pred[i]
            ax_pred.set_title(f"Agent {i} Predicted (local)", fontsize=10, pad=5)
            ax_pred.set_aspect('equal', adjustable='box')
            ax_pred.set_axis_off()

            try:
                if r.pred_mean_map_info is not None or r.pred_max_map_info is not None:
                    pred_info = r.pred_mean_map_info if r.pred_mean_map_info is not None else r.pred_max_map_info
                    pred_local = r.get_updating_map(r.location, base=pred_info)
                    belief_local = r.get_updating_map(r.location, base=r.map_info)

                    ax_pred.imshow(pred_local.map, cmap='gray', origin='lower', vmin=0, vmax=255)
                    alpha_mask = (belief_local.map == FREE) * 0.45
                    ax_pred.imshow(belief_local.map, cmap='Blues', origin='lower', alpha=alpha_mask)

                    rc = get_cell_position_from_coords(r.location, pred_local)
                    ax_pred.plot(rc[0], rc[1], 'mo', markersize=8, zorder=6)
                else:
                    ax_pred.text(0.5, 0.5, 'No prediction', ha='center', va='center', fontsize=9)
            except Exception as e:
                ax_pred.text(0.5, 0.5, f'Pred plot err:\n{e}', ha='center', va='center', fontsize=8)

        out_path = os.path.join(self.run_dir, f"t{step:04d}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        self.env.frame_files.append(out_path)
