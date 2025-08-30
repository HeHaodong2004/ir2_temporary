# worker.py
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
from collections import deque

from env import Env
from agent import Agent
from utils import *
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

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False
    )

# --- 全局任务ID计数器 ---
_mission_id_counter = 0
def get_new_mission_id():
    global _mission_id_counter
    _mission_id_counter += 1
    return _mission_id_counter

class Mission:
    """一个独立的会合任务实例"""
    def __init__(self, P, T_meet, r_meet, participants, start_step, meta=None):
        self.id = get_new_mission_id()
        self.P = np.array(P, dtype=float)
        self.T_meet = int(T_meet)
        self.r_meet = float(r_meet)
        self.participants = set(participants)
        self.t0 = int(start_step)
        self.meta = meta if isinstance(meta, dict) else {}
        self.color = plt.cm.spring(self.id % 7) # 为不同任务分配不同颜色用于绘图


class RegroupScheduler:
    def __init__(self):
        self.active = False
        self.P = None
        self.r_meet = 0.0
        self.T_meet = -1
        self.meta = None
        self.host_id = 0
        self.t0 = 0
        # 新增：任务参与者集合
        self.participants = set()

    def propose(self, P, T_meet, r_meet, meta=None, host_id=0, start_step=0):
        self.active = True
        self.P = np.array(P, dtype=float)
        self.T_meet = int(T_meet)
        self.r_meet = float(r_meet)
        self.meta = meta if isinstance(meta, dict) else {}
        self.host_id = int(host_id)
        self.t0 = int(start_step)
        # 初始时，所有agent都参与
        self.participants = set(range(N_AGENTS))

    def clear(self):
        self.active = False
        self.P = None
        self.T_meet = -1
        self.r_meet = 0.0
        self.meta = None
        self.t0 = 0
        self.participants.clear()


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

        self.last_known_locations = [self.env.robot_locations.copy() for _ in range(N_AGENTS)]
        self.last_known_intents = [{aid: [] for aid in range(N_AGENTS)} for _ in range(N_AGENTS)]

        self.run_dir = os.path.join(
            gifs_path,
            f"run_g{self.global_step}_w{self.meta_agent_id}_{os.getpid()}_{int(time.time()*1000)}"
        )
        if self.save_image:
            os.makedirs(self.run_dir, exist_ok=True)
        self.env.frame_files = []
        
        self._rdv_paths_per_agent = [set() for _ in range(N_AGENTS)]
        self.scheduler = RegroupScheduler()
        self.disc_free_ts = []
        self.disc_occ_ts = []

    def _match_intent_channels(self, obs_pack):
        n, m, e, ci, ce, ep = obs_pack
        need = NODE_INPUT_DIM
        got = n.shape[-1]
        if got < need:
            pad = torch.zeros((n.shape[0], n.shape[1], need - got), dtype=n.dtype, device=n.device)
            n = torch.cat([n, pad], dim=-1)
        elif got > need:
            n = n[..., :need]
        return [n, m, e, ci, ce, ep]

    def run_episode(self):
        done = False

        # ========== 初始化：图与预测 ==========
        for i, r in enumerate(self.robots):
            map_i = self.env.get_agent_map(i)
            r.update_graph(map_i, self.env.robot_locations[i])
        for r in self.robots:
            r.update_predict_map()

        # ========== 初始化：视角缓存 ==========
        self.last_known_locations = [self.env.robot_locations.copy() for _ in range(N_AGENTS)]
        self.last_known_intents = [{aid: [] for aid in range(N_AGENTS)} for _ in range(N_AGENTS)]
        for i, r in enumerate(self.robots):
            r.update_planning_state(self.env.robot_locations)
            self.last_known_intents[r.id][r.id] = deepcopy(r.intent_seq)

        if self.save_image:
            self.plot_env(step=0)

        # ========== 主循环 ==========
        for t in range(MAX_EPISODE_STEP):
            rdv_reward_this_step = 0.0

            # ===== 1. 会合任务管理 (在决策前) =====
            groups = self.env._compute_comm_groups()
            is_disconnected = len(groups) > 1

            # 1.1 检查是否需要发起新的会合任务
            if not self.scheduler.active and is_disconnected:
                try:
                    # 使用新的picker (假设新的picker只需要worker实例)
                    P, r_meet, T_meet, meta = pick_rendezvous_point(self)
                    if P is not None:
                        # T_meet是绝对步数
                        self.scheduler.propose(P, T_meet, r_meet, meta=meta, host_id=0, start_step=t)
                except Exception as e:
                    print(f"[Error] Rendezvous point picking failed: {e}")

            # 1.2 处理意外相遇
            if self.scheduler.active:
                for group in groups:
                    participants_in_group = self.scheduler.participants.intersection(group)
                    if len(participants_in_group) > 1:
                        representative = min(participants_in_group)
                        released_agents = participants_in_group - {representative}
                        self.scheduler.participants -= released_agents

            # 1.3 检查任务成功或失败
            if self.scheduler.active:
                all_participants_arrived = False
                if not self.scheduler.participants:
                    all_participants_arrived = True
                else:
                    arrived_count = 0
                    for agent_id in self.scheduler.participants:
                        dist = np.linalg.norm(self.robots[agent_id].location - self.scheduler.P)
                        if dist <= self.scheduler.r_meet:
                            arrived_count += 1
                    if arrived_count == len(self.scheduler.participants):
                        all_participants_arrived = True

                if all_participants_arrived:
                    rdv_reward_this_step += float(R_MEET_SUCCESS)
                    self.scheduler.clear()
                elif t >= self.scheduler.T_meet + int(MEET_LATE_TOL):
                    rdv_reward_this_step -= float(R_MEET_LATE)
                    self.scheduler.clear()

            # ===== 2. 更新agent状态 (注入会合信息) =====
            if self.scheduler.active:
                total_duration = max(1, self.scheduler.T_meet - self.scheduler.t0)
                time_left = max(0, self.scheduler.T_meet - t)
                tl_norm = float(np.clip(time_left / total_duration, 0.0, 1.0))
            else:
                tl_norm = 0.0
            for ag in self.robots:
                ag.time_left_norm = tl_norm

            self._update_rdv_paths_per_agent()
            for i, ag in enumerate(self.robots):
                if self.scheduler.active and i in self.scheduler.participants:
                    ag.rdv_path_nodes_set = self._rdv_paths_per_agent[i]
                else:
                    ag.rdv_path_nodes_set = set()

            # ===== 3. Agent决策 =====
            picks, dists = [], []
            for i, r in enumerate(self.robots):
                obs = r.get_observation(
                    robot_locations=self.last_known_locations[i],
                    global_intents=self.last_known_intents[i]
                )
                c_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, robot_locations=self.env.robot_locations
                )
                c_obs = self._match_intent_channels(c_obs)
                r.save_observation(obs, c_obs)

                nxt, _, act = r.select_next_waypoint(obs)
                r.save_action(act)
                picks.append(nxt)
                dists.append(np.linalg.norm(nxt - r.location))

            # ===== 4. 冲突消解与执行 =====
            picks = self.resolve_conflicts(picks, dists)
            prev_max = self.env.get_agent_travel().max()
            prev_total = self.env.get_total_travel()

            for r, loc in zip(self.robots, picks):
                self.env.step(loc, r.id)

            curr_agent_travel = self.env.get_agent_travel()
            new_max = curr_agent_travel.max()
            delta_max = new_max - prev_max
            delta_total = self.env.get_total_travel() - prev_total
            self.env.max_travel_dist = new_max
            
            # ===== 5. 状态与信息更新 =====
            try:
                free_masks, occ_masks = self.env.pop_discovery_masks()
                self.disc_free_ts.append([float(m.sum()) * (CELL_SIZE ** 2) for m in free_masks])
                self.disc_occ_ts.append([float(m.sum()) * (CELL_SIZE ** 2) for m in occ_masks])
            except Exception: pass

            groups_after_move = self.env._compute_comm_groups()
            for g in groups_after_move:
                for i in g:
                    for j in g:
                        self.last_known_locations[i][j] = self.env.robot_locations[j].copy()
                        self.last_known_intents[i][j] = deepcopy(self.robots[j].intent_seq)

            for i, r in enumerate(self.robots):
                r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
                r.update_predict_map()
                r.update_planning_state(
                    robot_locations=self.last_known_locations[i],
                    intents_view=self.last_known_intents[i]
                )
                self.last_known_intents[i][r.id] = deepcopy(r.intent_seq)

            # ===== 6. 奖励计算 =====
            indiv_utility = []
            for r, picked in zip(self.robots, picks):
                try:
                    idx = np.argwhere(np.all(r.node_coords == picked, axis=1))[0, 0]
                    utility_term = r.utility[idx] / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
                    indiv_utility.append(float(utility_term))
                except Exception:
                    indiv_utility.append(0.0)

            team_reward_env, per_agent_obs_rewards = self.env.calculate_reward()
            max_penalty = (delta_max / UPDATING_MAP_SIZE) * MAX_TRAVEL_COEF
            total_penalty = (delta_total / UPDATING_MAP_SIZE) * TOTAL_TRAVEL_COEF
            
            # 将探索奖励、惩罚和本步的会合奖励合并
            team_reward = team_reward_env - (max_penalty + total_penalty) + rdv_reward_this_step
            
            # ===== 7. 终止判断 =====
            agent_rates = [np.count_nonzero(self.env.agent_beliefs[i] != UNKNOWN) / max(1, self.env.ground_truth.size) for i in range(N_AGENTS)]
            done = all(r >= 0.995 for r in agent_rates)
            if done:
                team_reward += 10.0

            # ===== 8. 保存缓冲 =====
            for i, r in enumerate(self.robots):
                indiv_total = indiv_utility[i] + team_reward + per_agent_obs_rewards[i]
                r.save_reward(indiv_total)
                r.save_done(done)
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
        self.perf_metrics['travel_dist'] = self.env.get_total_travel()
        self.perf_metrics['max_travel'] = self.env.get_max_travel()
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        try:
            free_m2, occ_m2 = self.env.get_discovered_area()
            bal_free = self.env.get_map_balance_stats("free")
            bal_both = self.env.get_map_balance_stats("both")
            self.perf_metrics.update({
                'disc_free_m2_per_agent': free_m2.tolist(),
                'disc_occ_m2_per_agent': occ_m2.tolist(),
                'disc_free_cv': bal_free['cv'],
                'disc_both_cv': bal_both['cv']
            })
        except Exception: pass
        
        if self.save_image:
            gif_out = os.path.join(self.run_dir, f"episode_{self.global_step}_w{self.meta_agent_id}.gif")
            make_gif_safe(self.env.frame_files, gif_out, duration_ms=120)

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
            curr_node_data = robot.node_manager.nodes_dict.find(robot.location.tolist()).data
            neighbor_coords = sorted(
                list(curr_node_data.neighbor_set), 
                key=lambda c: np.linalg.norm(np.array(c) - target)
            )
            picked = None
            for cand in neighbor_coords:
                key = complex(cand[0], cand[1])
                if key not in chosen_complex:
                    picked = cand
                    break
            if picked is None:
                picked = robot.location.copy()
            resolved[rid] = np.array(picked)
            chosen_complex.add(complex(picked[0], picked[1]))
        return np.array(resolved).reshape(-1, 2)
    
    # ===== 新增/修改的辅助函数 =====

    def _update_rdv_paths_per_agent(self):
        if not self.scheduler.active or self.scheduler.P is None:
            self._rdv_paths_per_agent = [set() for _ in range(N_AGENTS)]
            return

        a0 = self.robots[0]
        if a0.pred_mean_map_info is None: return

        pred_map = a0.pred_mean_map_info.map.astype(np.float32)
        belief_map = a0.map_info.map
        prob_free = pred_map / float(FREE)
        tau_free = 0.6
        traversable_mask = ((prob_free >= tau_free) | (belief_map == FREE)) & (belief_map != OCCUPIED)

        goal_rc = self._world_to_cell_rc(self.scheduler.P, a0.map_info)
        if not traversable_mask[goal_rc[0], goal_rc[1]]:
            goal_rc = self._find_nearest_valid_cell(traversable_mask, goal_rc)

        new_paths = [set() for _ in range(N_AGENTS)]
        for i, r in enumerate(self.robots):
            if i in self.scheduler.participants:
                start_rc = self._world_to_cell_rc(r.location, a0.map_info)
                if not traversable_mask[start_rc[0], start_rc[1]]:
                    start_rc = self._find_nearest_valid_cell(traversable_mask, start_rc)
                
                rc_path = self._bfs_path_rc(traversable_mask, start_rc, goal_rc)
                if rc_path:
                    new_paths[i] = self._cells_to_nodecoord_set(rc_path, a0.map_info)
        
        self._rdv_paths_per_agent = new_paths

    def _world_to_cell_rc(self, world_xy, map_info: MapInfo):
        cell = get_cell_position_from_coords(np.array(world_xy, dtype=float), map_info)
        return int(cell[1]), int(cell[0])

    def _cells_to_nodecoord_set(self, rc_path, map_info):
        if not rc_path: return set()
        coords = np.array([[map_info.map_origin_x + c * map_info.cell_size,
                            map_info.map_origin_y + r * map_info.cell_size] for r, c in rc_path])
        coords = np.around(coords, 1)
        return {tuple(c) for c in coords}

    def _find_nearest_valid_cell(self, mask, start_rc):
        q = deque([start_rc])
        visited = {start_rc}
        while q:
            r, c = q.popleft()
            if mask[r, c]: return r, c
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and (nr, nc) not in visited:
                    q.append((nr, nc))
                    visited.add((nr, nc))
        return start_rc

    def _bfs_path_rc(self, trav_mask, start_rc, goal_rc):
        H, W = trav_mask.shape
        q = deque([(start_rc, [start_rc])])
        visited = {start_rc}
        while q:
            (r, c), path = q.popleft()
            if (r, c) == goal_rc: return path
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and trav_mask[nr, nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [(nr, nc)]))
        return []

    def plot_env(self, step):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib import patheffects as pe
        from matplotlib.gridspec import GridSpec

        plt.switch_backend('agg')
        mpl.rcParams.update({
            "figure.facecolor": "#0d0f12", "axes.facecolor": "#0d0f12",
            "savefig.facecolor": "#0d0f12", "axes.edgecolor": "#181b20",
            "axes.linewidth": 1.2, "font.size": 11,
            "xtick.color": "#a8b3c7", "ytick.color": "#a8b3c7", "text.color": "#e6edf3",
        })
        agent_colors = ["#4cc9f0", "#f72585", "#43aa8b", "#ffca3a", "#b5179e", "#4895ef"]
        def ac(i): return agent_colors[i % len(agent_colors)]

        rows = max(N_AGENTS, 2)
        fig = plt.figure(figsize=(16, 9), dpi=120)
        gs = GridSpec(nrows=rows, ncols=2, width_ratios=[3, 1], wspace=0.06, hspace=0.05, figure=fig)
        ax_global = fig.add_subplot(gs[:, 0])
        ax_locals = [fig.add_subplot(gs[i, 1]) for i in range(rows)]

        # Left Panel: Global View
        gb = self.env.global_belief
        ax_global.imshow(gb, cmap='gray', vmin=0, vmax=255, zorder=0)
        ax_global.imshow((gb == UNKNOWN).astype(float), cmap='Greens', alpha=0.1, zorder=1)
        ax_global.set_title("Global View", pad=8)
        ax_global.set_axis_off()

        for r in self.robots:
            col = ac(r.id)
            if hasattr(r, 'trajectory_x') and len(r.trajectory_x) > 1:
                xs = (np.array(r.trajectory_x) - r.map_info.map_origin_x) / r.cell_size
                ys = (np.array(r.trajectory_y) - r.map_info.map_origin_y) / r.cell_size
                ax_global.plot(xs, ys, color=col, linewidth=1.5, alpha=0.9, zorder=2)
            cell = get_cell_position_from_coords(r.location, r.map_info)
            ax_global.plot(cell[0], cell[1], 'o', ms=8, mfc=col, mec='white', mew=1.2, zorder=6)

        groups = self.env._compute_comm_groups()
        for g in groups:
            for i_idx, i in enumerate(list(g)):
                for j in list(g)[i_idx+1:]:
                    p1 = get_cell_position_from_coords(self.robots[i].location, self.robots[i].map_info)
                    p2 = get_cell_position_from_coords(self.robots[j].location, self.robots[j].map_info)
                    ax_global.plot([p1[0], p2[0]], [p1[1], p2[1]], c="#66ffb3", lw=1.5, alpha=0.7, zorder=3)

        # Plot Rendezvous Info
        if self.scheduler.active and self.scheduler.P is not None:
            center_xy, r_meet = self.scheduler.P, self.scheduler.r_meet
            cc = get_cell_position_from_coords(center_xy, self.robots[0].map_info)
            ax_global.plot(cc[0], cc[1], '*', ms=18, mfc='#ff4d4f', mec='white', mew=2.0, zorder=12)
            r_pix = np.clip(r_meet / self.robots[0].cell_size, 2.0, 400.0)
            ring = patches.Circle((cc[0], cc[1]), r_pix, fill=False, ls='--', lw=2.0, ec='#ff4d4f', alpha=0.95, zorder=11)
            ax_global.add_patch(ring)
            for agent_id in self.scheduler.participants:
                r = self.robots[agent_id]
                p_agent = get_cell_position_from_coords(r.location, r.map_info)
                ax_global.plot([p_agent[0], cc[0]], [p_agent[1], cc[1]], c=ac(r.id), ls=':', lw=1.5, alpha=0.8, zorder=10)

        for r in self.robots:
            if hasattr(r, 'rdv_path_nodes_set') and r.rdv_path_nodes_set:
                pts = np.array(list(r.rdv_path_nodes_set)).reshape(-1, 2)
                cells = get_cell_position_from_coords(pts, r.map_info).reshape(-1, 2)
                ax_global.scatter(cells[:, 0], cells[:, 1], s=12, marker='s', c=ac(r.id), alpha=0.9, zorder=8, lw=0)
        
        # HUD
        hud_lines = [f"Step {step}/{MAX_EPISODE_STEP}", f"Explored: {self.env.explored_rate*100:.1f}%"]
        if self.scheduler.active:
            time_left = max(0, self.scheduler.T_meet - step)
            hud_lines.append(f"Mission: ON | T_left={time_left} | Participants: {len(self.scheduler.participants)}")
        else:
            hud_lines.append("Mission: OFF")
        txt = ax_global.text(5, 5, "\n".join(hud_lines), va='top', ha='left', zorder=20,
                             bbox=dict(boxstyle="round,pad=0.4", fc=(0,0,0,0.4), ec="#3a3f45"))

        # Right Panels: Local Views
        for i in range(N_AGENTS):
            r, ax = self.robots[i], ax_locals[i]
            ax.set_axis_off()
            try:
                ax.imshow(r.updating_map_info.map, cmap='gray', vmin=0, vmax=255)
                rc2 = get_cell_position_from_coords(r.location, r.updating_map_info)
                ax.plot(rc2[0], rc2[1], 'o', ms=7, mfc=ac(r.id), mec='white', mew=1.2)
                # Plot intents
                intents_view = self.last_known_intents[i]
                for aid, path in intents_view.items():
                    if not path: continue
                    start_pos = self.last_known_locations[i][aid]
                    full_path = [start_pos] + path
                    path_cells = [get_cell_position_from_coords(p, r.updating_map_info) for p in full_path]
                    ix, iy = zip(*path_cells)
                    ax.plot(ix, iy, ls='--', marker='x', ms=4, c=ac(aid), alpha=0.9)
                ax.set_title(f"Agent {r.id}", fontsize=10, pad=3)
            except Exception: pass
        for j in range(N_AGENTS, rows): ax_locals[j].set_visible(False)

        out_path = os.path.join(self.run_dir, f"t{step:04d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        self.env.frame_files.append(out_path)
