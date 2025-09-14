# worker.py — local planner training (RDV removed)
import os
import time
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from env import Env
from agent import Agent
from utils import *  # MapInfo, get_cell_position_from_coords, ...
from node_manager import NodeManager
from ground_truth_node_manager import GroundTruthNodeManager
from parameter import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


def make_gif_safe(frame_paths, out_path, duration_ms=120):
    frame_paths = [p for p in frame_paths if os.path.exists(p)]
    frame_paths.sort()
    if not frame_paths:
        return
    frames = []
    base_size = None
    for p in frame_paths:
        try:
            im = Image.open(p).convert("RGB")
            if base_size is None:
                base_size = im.size
            elif im.size != base_size:
                im = im.resize(base_size, Image.BILINEAR)
            frames.append(im)
        except Exception:
            continue
    if not frames:
        return
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


class Worker:
    def __init__(self, meta_agent_id, policy_net, predictor, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.node_manager = NodeManager(plot=self.save_image)
        self.robots = [Agent(i, policy_net, predictor, self.node_manager, device=device, plot=save_image)
                       for i in range(N_AGENTS)]
        self.gtnm = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info, device=device, plot=save_image)

        self.episode_buffer = [[] for _ in range(27)]
        self.perf_metrics = dict()

        # 通信缓存
        self.last_known_locations = [self.env.robot_locations.copy() for _ in range(N_AGENTS)]
        self.last_known_intents = [{aid: [] for aid in range(N_AGENTS)} for _ in range(N_AGENTS)]

        # 输出目录
        self.run_dir = os.path.join(
            gifs_path, f"run_g{self.global_step}_w{self.meta_agent_id}_{os.getpid()}_{int(time.time()*1000)}"
        )
        if self.save_image:
            os.makedirs(self.run_dir, exist_ok=True)
        self.env.frame_files = []

        # 预计算：地面真实 FREE 栅格总数（用于并集覆盖=100% 停止）
        gt_map = self.env.ground_truth_info.map
        self._gt_free_total = int(np.count_nonzero(gt_map == FREE)) if gt_map is not None else 0

    # critic 通道对齐
    def _match_intent_channels(self, obs_pack):
        n, m, e, ci, ce, ep = obs_pack
        need, got = NODE_INPUT_DIM, n.shape[-1]
        if got < need:
            pad = torch.zeros((n.shape[0], n.shape[1], need - got), dtype=n.dtype, device=n.device)
            n = torch.cat([n, pad], dim=-1)
        return [n, m, e, ci, ce, ep]

    # -------------------------------------------------------------------------
    # 主循环（无 RDV）
    # -------------------------------------------------------------------------
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

        if self.save_image:
            self.plot_env(step=0)

        for t in range(MAX_EPISODE_STEP):
            # ---------- 原策略：自由探索一步 ----------
            picks_raw, dists = [], []
            for i, r in enumerate(self.robots):
                obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_observation(obs, self._match_intent_channels(c_obs))
                nxt, _, act = r.select_next_waypoint(obs)
                r.save_action(act)
                picks_raw.append(nxt)
                dists.append(np.linalg.norm(nxt - r.location))

            # ---------- 冲突消解 & 推进环境 ----------
            picks = self.resolve_conflicts(picks_raw, dists)
            prev_max = self.env.get_agent_travel().max()
            prev_total = self.env.get_total_travel()
            for r, loc in zip(self.robots, picks):
                self.env.step(loc, r.id)
            self.env.max_travel_dist = self.env.get_agent_travel().max()
            delta_max = self.env.max_travel_dist - prev_max
            delta_total = self.env.get_total_travel() - prev_total

            # ---------- 通信同步（同组可见） ----------
            groups_after_move = self._compute_groups_from_positions(self.env.robot_locations)
            for g in groups_after_move:
                for i in g:
                    for j in g:
                        self.last_known_locations[i][j] = self.env.robot_locations[j].copy()
                        self.last_known_intents[i][j] = deepcopy(self.robots[j].intent_seq)

            # ---------- 图/预测/意图更新 ----------
            for i, r in enumerate(self.robots):
                r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
                r.update_predict_map()
                r.update_planning_state(self.last_known_locations[i], self.last_known_intents[i])
                self.last_known_intents[i][r.id] = deepcopy(r.intent_seq)

            # ---------- 奖励与终止 ----------
            # env.calculate_reward 兼容 team-only 或 (team, per_agent) 返回
            calc_ret = self.env.calculate_reward()
            if isinstance(calc_ret, tuple) and len(calc_ret) == 2:
                team_reward_env, per_agent_obs_rewards = calc_ret
            else:
                team_reward_env, per_agent_obs_rewards = float(calc_ret), [0.0] * N_AGENTS

            team_reward = (
                team_reward_env
                - ((delta_max / UPDATING_MAP_SIZE) * MAX_TRAVEL_COEF)
                - ((delta_total / UPDATING_MAP_SIZE) * TOTAL_TRAVEL_COEF)
            )

            # 并集合覆盖率=100% 或 utility 耗尽则结束
            if self._gt_free_total > 0:
                merged_count = int(np.count_nonzero(self.env.global_belief == FREE))
                coverage_ok = (merged_count / self._gt_free_total >= 0.995)
            else:
                coverage_ok = False

            total_util = sum(float(r.utility.sum()) for r in self.robots)
            utilities_empty = (total_util == 0.0)

            done = coverage_ok or utilities_empty
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

        # ---------- 总结 ----------
        self.perf_metrics.update({
            'travel_dist': self.env.get_total_travel(),
            'max_travel': self.env.get_max_travel(),
            'explored_rate': (int(np.count_nonzero(self.env.global_belief == FREE)) / self._gt_free_total
                              if self._gt_free_total > 0 else self.env.explored_rate),
            'success_rate': done
        })

        if self.save_image:
            make_gif_safe(self.env.frame_files, os.path.join(self.run_dir, f"ep_{self.global_step}.gif"))

        for r in self.robots:
            for k in range(len(self.episode_buffer)):
                self.episode_buffer[k] += r.episode_buffer[k]

    # -------------------------------------------------------------------------
    # 冲突消解（按“先到先得” + 节点邻居回退）
    # -------------------------------------------------------------------------
    def resolve_conflicts(self, picks, dists):
        picks = np.array(picks).reshape(-1, 2)
        order = list(np.argsort(dists))
        chosen_complex, resolved = set(), [None] * len(self.robots)

        for rid in order:
            robot = self.robots[rid]
            try:
                loc_key = np.around(robot.location, 1).tolist()
                neighbor_coords = sorted(
                    list(self.node_manager.nodes_dict.find(loc_key).data.neighbor_set),
                    key=lambda c: np.linalg.norm(np.array(c) - picks[rid])
                )
            except Exception:
                neighbor_coords = [robot.location.copy()]

            picked = None
            for cand in neighbor_coords:
                key = complex(cand[0], cand[1])
                if key not in chosen_complex:
                    picked = np.array(cand)
                    break
            resolved[rid] = picked if picked is not None else robot.location.copy()
            chosen_complex.add(complex(resolved[rid][0], resolved[rid][1]))
        return np.array(resolved).reshape(-1, 2)

    # -------------------------------------------------------------------------
    # 通信连通性（基于欧氏距离）
    # -------------------------------------------------------------------------
    def _compute_groups_from_positions(self, positions):
        n = len(positions)
        if n == 0:
            return []
        used = [False] * n
        groups = []
        for i in range(n):
            if used[i]:
                continue
            comp, q = [], [i]
            used[i] = True
            while q:
                u = q.pop()
                comp.append(u)
                for v in range(n):
                    if used[v]:
                        continue
                    if np.linalg.norm(np.asarray(positions[u]) - np.asarray(positions[v])) <= COMMS_RANGE + 1e-6:
                        used[v] = True
                        q.append(v)
            groups.append(tuple(sorted(comp)))
        return groups

    def plot_env(self, step):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import matplotlib.patches as patches

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 10), dpi=110)

        gs = GridSpec(N_AGENTS, 3, figure=fig, width_ratios=[2.5, 1.2, 1.2], wspace=0.15, hspace=0.1)
        ax_global = fig.add_subplot(gs[:, 0])
        ax_locals_obs = [fig.add_subplot(gs[i, 1]) for i in range(N_AGENTS)]
        ax_locals_pred = [fig.add_subplot(gs[i, 2]) for i in range(N_AGENTS)]
        agent_colors = plt.cm.get_cmap('cool', N_AGENTS)

        # --- Global panel ---
        global_info = MapInfo(self.env.global_belief, self.env.belief_origin_x, self.env.belief_origin_y,
                            self.env.cell_size)
        ax_global.set_title(f"Global View | Step {step}/{MAX_EPISODE_STEP}", fontsize=14, pad=10)
        ax_global.imshow(global_info.map, cmap='gray', origin='lower')
        ax_global.set_aspect('equal', adjustable='box')
        ax_global.set_axis_off()

        # 叠加预测未知区概率（仅在未知处着色）
        if self.robots and self.robots[0].pred_mean_map_info is not None:
            pred_mean = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
            belief = global_info.map
            unknown_mask = (belief == UNKNOWN)
            prob = np.zeros_like(pred_mean)
            prob[unknown_mask] = pred_mean[unknown_mask]
            ax_global.imshow(prob, cmap='magma', origin='lower', alpha=0.35)

        # 通信边（基于欧氏距离的组内连线）
        groups = self._compute_groups_from_positions(self.env.robot_locations)
        for group in groups:
            for i_idx, i in enumerate(list(group)):
                for j in list(group)[i_idx + 1:]:
                    r1c, r1r = self._world_to_cell_rc(self.robots[i].location, global_info)[1], \
                            self._world_to_cell_rc(self.robots[i].location, global_info)[0]
                    r2c, r2r = self._world_to_cell_rc(self.robots[j].location, global_info)[1], \
                            self._world_to_cell_rc(self.robots[j].location, global_info)[0]
                    ax_global.plot([r1c, r2c], [r1r, r2r], color="#33ff88", lw=2, alpha=0.8, zorder=5)

        # 机器人、轨迹、通讯圈
        for i, r in enumerate(self.robots):
            rr, cc = self._world_to_cell_rc(r.location, global_info)
            # 轨迹
            if hasattr(r, 'trajectory_x') and r.trajectory_x:
                traj_cells = [self._world_to_cell_rc(np.array([x, y]), global_info)
                            for x, y in zip(r.trajectory_x, r.trajectory_y)]
                ax_global.plot([c for rr_, c in traj_cells], [rr_ for rr_, c in traj_cells],
                            color=agent_colors(i), lw=1.5, zorder=3)
            # 通信半径
            comms_radius = patches.Circle((cc, rr), COMMS_RANGE / CELL_SIZE,
                                        fc=(0, 1, 0, 0.05), ec=(0, 1, 0, 0.4), ls='--', lw=1.5, zorder=4)
            ax_global.add_patch(comms_radius)
            # 当前位置
            ax_global.plot(cc, rr, 'o', ms=10, mfc=agent_colors(i),
                        mec='white', mew=1.5, zorder=10)

        # --- Local panels ---
        for i, r in enumerate(self.robots):
            # Local Obs
            ax_obs = ax_locals_obs[i]
            local_map_info = r.map_info
            ax_obs.set_title(f"Agent {i} View", fontsize=10, pad=5)
            ax_obs.imshow(local_map_info.map, cmap='gray', origin='lower')
            ax_obs.set_aspect('equal', adjustable='box')
            rr, cc = self._world_to_cell_rc(r.location, local_map_info)
            ax_obs.plot(cc, rr, 'o', ms=8, mfc=agent_colors(i),
                        mec='white', mew=1.5, zorder=10)
            if r.intent_seq:
                intent_world = [r.location] + r.intent_seq
                intent_cells = [self._world_to_cell_rc(pos, local_map_info) for pos in intent_world]
                ax_obs.plot([c for rr_, c in intent_cells], [rr_ for rr_, c in intent_cells],
                            'x--', c=agent_colors(i), lw=2, ms=6, zorder=8)
            ax_obs.set_axis_off()

            # Local Pred
            ax_pred = ax_locals_pred[i]
            ax_pred.set_title(f"Agent {i} Predicted (local)", fontsize=10, pad=5)
            ax_pred.set_aspect('equal', adjustable='box')
            ax_pred.set_axis_off()

            try:
                if r.pred_mean_map_info is not None or r.pred_max_map_info is not None:
                    pred_info = r.pred_mean_map_info if r.pred_mean_map_info is not None else r.pred_max_map_info
                    pred_local = r.get_updating_map(r.location, base=pred_info)
                    belief_local = r.get_updating_map(r.location, base=r.map_info)

                    # 背景：预测值；上覆：当前 belief（FREE 区半透明蓝）
                    ax_pred.imshow(pred_local.map, cmap='gray', origin='lower', vmin=0, vmax=255)
                    alpha_mask = (belief_local.map == FREE) * 0.45
                    ax_pred.imshow(belief_local.map, cmap='Blues', origin='lower', alpha=alpha_mask)

                    # FIX 1: 当前位姿（world -> (row,col) -> plot(col,row)）
                    rr_p, cc_p = self._world_to_cell_rc(r.location, pred_local)
                    ax_pred.plot(cc_p, rr_p, 'mo', markersize=8, zorder=6)

                    # （可选）FIX 2: 轨迹
                    if hasattr(r, 'trajectory_x') and r.trajectory_x:
                        traj_rc = [self._world_to_cell_rc(np.array([x, y]), pred_local)
                                for x, y in zip(r.trajectory_x, r.trajectory_y)]
                        tr = [rc_[0] for rc_ in traj_rc]
                        tc = [rc_[1] for rc_ in traj_rc]
                        ax_pred.plot(tc, tr, linewidth=2, zorder=1)

                    # （可选）FIX 3: 意图
                    if r.intent_seq:
                        intent_rc = [self._world_to_cell_rc(pos, pred_local) for pos in r.intent_seq]
                        ir = [rc_[0] for rc_ in intent_rc]
                        ic = [rc_[1] for rc_ in intent_rc]
                        ax_pred.plot(ic, ir, 'x--', lw=2, ms=6, zorder=8)
                else:
                    ax_pred.text(0.5, 0.5, 'No prediction', ha='center', va='center', fontsize=9)
            except Exception as e:
                ax_pred.text(0.5, 0.5, f'Pred plot err:\n{e}', ha='center', va='center', fontsize=8)

        out_path = os.path.join(self.run_dir, f"t{step:04d}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        self.env.frame_files.append(out_path)

