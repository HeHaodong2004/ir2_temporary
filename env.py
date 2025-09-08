# domains_cc/path_planning_rl/env/planning_env_coverage_runner.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional, Dict, Tuple, List
import matplotlib
matplotlib.use("Agg")

from domains_cc.worldCC import WorldCollisionChecker
from domains_cc.worldCCVisualizer import addGridToPlot, addXYThetaToPlot
from domains_cc.footprints.footprint import load_footprint_from_yaml_path
from domains_cc.dynamics.robot_dynamics import load_dynamics_from_yaml_path


class PathPlanningCoverageEnvRunner(gym.Env):
    """
    Coverage-as-tour（Runner 版）：
      - 外部注入 world_cc / start_xytheta（不读取 scen/map）
      - 分块取代表点并排序为一条“巡游”序列（order=nn2opt|snake）
      - 逐个作为 point-goal 子目标交给 PPO
      - 覆盖用射线投射（不可透墙）
      - “快速切换”：步数预算 / 无进展 / 连续碰撞 触发跳过当前子目标
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        # —— 与 runner 接口对齐：外部注入世界与起止位姿 —— 
        problem_index: int,            # 占位，与 rl_sas 接口兼容
        dynamics_config: str,
        footprint_config: str,
        max_steps: int = 4000,
        angle_tol: float = 0.10,
        n_lasers: int = 16,
        max_scan_dist: float = 10.0,
        hist_len: int = 6,
        start_xytheta: Optional[np.ndarray] = None,
        goal_xytheta: Optional[np.ndarray] = None,
        world_cc: Optional[WorldCollisionChecker] = None,
        constraints: list = None,

        # —— coverage 相关参数（给默认值，必要时可在 solver_config 里透传）——
        cover_radius_m: float = 2.0,
        cover_target_ratio: float = 0.95,
        subgoal_reach_tol_m: float = 1.0,
        subgoal_bonus: float = 5.0,
        sensor_range_m: Optional[float] = None,
        show_k_next: int = 20,
        tour_cell_m: float = 2.0,
        tour_start_from_row: str = "nearest",
        tour_pick: str = "center",
        order_strategy: str = "nn2opt",
        cover_n_rays: int = 180,
        goal_step_budget: int = 120,
        no_improve_patience: int = 25,
        min_progress_m: float = 0.10,
        collision_patience: int = 10,

        # 奖励/代价
        time_cost: float = -0.2,
        completion_bonus: float = 80.0,
        collision_penalty: float = -10.0,
        stuck_step_penalty: float = 0.6,
        stuck_cap: int = 5,
    ):
        super().__init__()
        assert world_cc is not None, "coverage-runner need world_cc"
        assert start_xytheta is not None, "coverage-runner need start_xytheta"

        # —— dynamics/footprint/world —— 
        self.dynamics, _ = load_dynamics_from_yaml_path(dynamics_config)
        self.footprint, _, _ = load_footprint_from_yaml_path(footprint_config)
        self.world_cc = world_cc
        raw_grid = self.world_cc.grid    # 注意：world_cc.grid 是 top-origin & grid[x,y]
        self.resolution = float(self.world_cc.resolution)

        grid_xy = self.world_cc.grid                         # grid[x,y]，bottom-origin
        self.free_mask = (grid_xy.T == 0).astype(np.uint8)   # → free_mask[row(Y), col(X)]
        self.H, self.W = self.free_mask.shape

        self._build_nav_mask(n_theta_samples=8, border_margin_m=self.resolution*2.0)

        # —— 起止位姿（goal 仅用于可视化/初始化，不是完成判据）——
        self.start_xytheta = np.array(start_xytheta, dtype=np.float32)
        self.goal_xytheta  = np.array(goal_xytheta if goal_xytheta is not None else start_xytheta, dtype=np.float32)
        self.start_state   = self.dynamics.get_state_from_xytheta(self.start_xytheta)
        self.goal_state    = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # —— 覆盖/子目标/快速切换参数 —— 
        self.cover_radius_m = float(cover_radius_m)
        self.cover_target_ratio = float(cover_target_ratio)
        self.subgoal_reach_tol_m = float(subgoal_reach_tol_m)
        self.subgoal_bonus = float(subgoal_bonus)
        self.cover_n_rays = int(cover_n_rays)
        self.sensor_range_m = float(sensor_range_m) if sensor_range_m is not None else float(max_scan_dist)

        self.node_strategy = "tour"
        self.show_k_next = int(show_k_next)
        self.tour_cell_m = float(tour_cell_m)
        self.tour_start_from_row = str(tour_start_from_row)
        self.tour_pick = str(tour_pick)
        self.order_strategy = str(order_strategy)

        self.goal_step_budget = int(goal_step_budget)
        self.no_improve_patience = int(no_improve_patience)
        self.min_progress_m = float(min_progress_m)
        self.collision_patience = int(collision_patience)

        # —— 奖励相关 —— 
        self.time_cost = float(time_cost)
        self.completion_bonus = float(completion_bonus)
        self.collision_penalty = float(collision_penalty)
        self.stuck_step_penalty = float(stuck_step_penalty)
        self.stuck_cap = int(stuck_cap)

        # —— 动作/观测空间，与 P2P policy 完全一致 —— 
        self.n_actions = self.dynamics.motion_primitives.shape[0]
        self.action_space = spaces.Discrete(self.n_actions)
        self.hist_len = int(hist_len)
        self.action_history = deque(maxlen=self.hist_len)

        self.n_lasers = int(n_lasers)
        self.max_scan_dist = float(max_scan_dist)
        self.laser_angles = np.linspace(-np.pi, np.pi, self.n_lasers, endpoint=False)

        self.observation_space = spaces.Dict({
            "scan":        spaces.Box(0.0, self.max_scan_dist, (self.n_lasers,), dtype=np.float32),
            "goal_vec":    spaces.Box(-np.inf, np.inf, (2,),       dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.n_actions),
            "hist":        spaces.MultiDiscrete([self.n_actions]*self.hist_len),
            "dist_grad":   spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
            "dist_phi":    spaces.Box(0.0, 1.0,   (1,), dtype=np.float32),
        })

        # —— 运行状态 —— 
        self.max_steps = int(max_steps)
        self.angle_tol = float(angle_tol)
        self.state: Optional[np.ndarray] = None
        self._steps = 0
        self._fig = None
        self._ax  = None
        self.constraints = list(constraints) if constraints is not None else []

        # —— 内部势场/梯度/子目标/覆盖 —— 
        self.dist_map = None
        self.grad_x = None
        self.grad_y = None
        self.max_potential = 0.0
        self.collision_streak = 0
        self.subgoal_count = 0
        self._goal_reachable_from_robot = True

        self.high_nodes_seq: List[Tuple[int,int]] = []
        self._nodes_used = np.zeros((self.H, self.W), dtype=np.uint8)

        self._no_progress_steps = 0
        self._best_phi_this_goal = np.inf
        self._best_goal_dist = np.inf
        self._collision_steps = 0
        self._goal_steps = 0

        # 时间步长（供 rl_sas 使用）
        self.dt_default = float(self.dynamics.motion_primitives[0, -1])

        self._init_fields_and_tour()

    # ====== 与 rl_sas 兼容的“runner 接口” ======
    def set_constraints(self, constraints):
        # coverage 先不在环境层面处理约束（rl_sas 会在 action mask 中处理）
        self.constraints = list(constraints) if constraints is not None else []

    def move(self, old_state, action):
        traj = self.dynamics.get_next_states(old_state)[action, :, :3]
        return traj[-1]

    def action_masks(self) -> np.ndarray:
        assert self.state is not None
        trajs = self.dynamics.get_next_states(self.state)[:, :, :3]
        mask = np.zeros(self.n_actions, dtype=bool)
        for i in range(self.n_actions):
            if not self.world_cc.isValid(self.footprint, trajs[i]).all():
                continue
            mask[i] = True
        return mask

    # ====== reset/step/render ======
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state.copy()
        self._steps = 0
        self.action_history.clear()
        for _ in range(self.hist_len):
            self.action_history.append(0)
        self.collision_streak = 0
        self.subgoal_count = 0
        self.covered = np.zeros((self.H, self.W), dtype=np.uint8)

        if options:
            if "cover_radius_m" in options:
                self.cover_radius_m = float(options["cover_radius_m"])
            if "sensor_range_m" in options:
                self.sensor_range_m = float(options["sensor_range_m"])
                self.max_scan_dist  = float(options["sensor_range_m"])

        self._mark_coverage(self.state[:2])
        self._build_high_level_nodes_tour()
        self._pick_first_reachable_goal()
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.state is not None
        cur = self.state.copy()

        traj = self.dynamics.get_next_states(cur)[action, :, :3]
        valid = self.world_cc.isValid(self.footprint, traj).all()
        info = {
            "collision": not valid, "reached": False,
            "covered_ratio": 0.0, "subgoals": self.subgoal_count,
            "next_nodes_world": []
        }

        # 碰撞/正常推进 + 基础奖励
        if not valid:
            self.collision_streak += 1
            extra = - self.stuck_step_penalty * min(max(self.collision_streak - 1, 0), self.stuck_cap)
            reward = self.collision_penalty + self.time_cost + extra
            self.action_history.append(action)
        else:
            self.state = traj[-1]
            self.action_history.append(action)
            if self.collision_streak > 0:
                self.collision_streak = 0

            prev_phi = self._interpolate_dist(cur[:2])
            new_phi  = self._interpolate_dist(self.state[:2])
            delta_potential = prev_phi - new_phi
            reward = delta_potential + self.time_cost

            # 更新覆盖
            self._mark_coverage(self.state[:2])

        self._steps += 1
        self._goal_steps += 1

        # === 停滞统计（进展用“势能/欧氏距离改善”双准则） ===
        goal_dist = float(np.hypot(self.goal_state[0] - self.state[0],
                                   self.goal_state[1] - self.state[1]))
        cur_phi = float(self._interpolate_dist(self.state[:2]))
        if valid:
            phi_improved  = (self._best_phi_this_goal - cur_phi) > 1e-3
            dist_improved = (self._best_goal_dist     - goal_dist) > self.min_progress_m
            if phi_improved or dist_improved:
                self._no_progress_steps = 0
                self._best_phi_this_goal = min(self._best_phi_this_goal, cur_phi)
                self._best_goal_dist     = min(self._best_goal_dist, goal_dist)
            else:
                self._no_progress_steps += 1
            self._collision_steps = 0
        else:
            self._no_progress_steps += 1
            self._collision_steps += 1

        # === 快速切换触发：步数预算 / 无进展 / 连续碰撞 ===
        def _try_advance_goal():
            tried = 0
            while tried < 64:
                next_rc = self._pick_next_goal_rc()
                if next_rc is None:
                    return False
                self._set_goal_by_rc(*next_rc)
                rr, cc = self._world_to_grid_rc(self.state[:2])
                if self._goal_reachable_from_robot and np.isfinite(self.dist_map[rr, cc]):
                    self.subgoal_count += 1
                    info["subgoals"] = self.subgoal_count
                    return True
                tried += 1
            return False

        if (self._goal_steps >= self.goal_step_budget) or \
           (self._no_progress_steps >= self.no_improve_patience) or \
           (self._collision_steps >= self.collision_patience):
            _ = _try_advance_goal()
            # 重置统计
            self._no_progress_steps = 0
            self._best_phi_this_goal = np.inf
            self._best_goal_dist = np.inf
            self._collision_steps = 0
            self._goal_steps = 0

        # === REPLACE 这段“到达判定” ===
        # 旧：dxg, dyg = self.state[0]-self.goal_state[0] ...
        # 新：看当前 waypoint
        if getattr(self, "_guide_wp_xy", None) and self._guide_idx < len(self._guide_wp_xy):
            wx, wy = self._guide_wp_xy[self._guide_idx]
            d_wp = float(np.hypot(self.state[0]-wx, self.state[1]-wy))
            reached_wp = (d_wp < self.subgoal_reach_tol_m)
            if reached_wp:
                self._guide_idx += 1
                # 全部 waypoint 完成，认为“到达子目标”
                reached = (self._guide_idx >= len(self._guide_wp_xy))
            else:
                reached = False
        else:
            dxg = self.state[0] - self.goal_state[0]
            dyg = self.state[1] - self.goal_state[1]
            reached = valid and (np.hypot(dxg, dyg) < self.subgoal_reach_tol_m)
        if reached:
            reward += self.subgoal_bonus
            info["reached"] = True
            _ = _try_advance_goal()
            self._no_progress_steps = 0
            self._best_phi_this_goal = np.inf
            self._best_goal_dist = np.inf
            self._collision_steps = 0
            self._goal_steps = 0

        # 覆盖率
        total_free = int(np.count_nonzero(self.free_mask == 1))
        covered_free = int(np.count_nonzero((self.free_mask == 1) & (self.covered == 1)))
        cover_ratio = covered_free / max(total_free, 1)
        info["covered_ratio"] = cover_ratio

        done = (cover_ratio >= self.cover_target_ratio)
        truncated = False
        if not done and self._steps >= self.max_steps:
            truncated = True
        if done:
            reward += self.completion_bonus

        # 未来K个节点（可视化）
        if self.high_nodes_seq:
            remain = []
            count = 0
            for (r, c) in self.high_nodes_seq:
                if count >= self.show_k_next:
                    break
                if self._nodes_used[r, c] == 1 or self.covered[r, c] == 1:
                    continue
                wx, wy = (c + 0.5)*self.resolution, (r + 0.5)*self.resolution
                remain.append([wx, wy]); count += 1
            info["next_nodes_world"] = remain

        return self._get_obs(), reward, done, truncated, info

    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6,6))
        ax = self._ax; ax.clear()
        addGridToPlot(self.world_cc, ax)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.start_xytheta)
        addXYThetaToPlot(self.world_cc, ax, self.footprint, self.state[:3])

        # 势场 & 覆盖热力
        if self.dist_map is not None:
            H, W = self.dist_map.shape
            dm = self.dist_map / (self.max_potential + 1e-6)
            ax.imshow(dm, origin='lower',
                      extent=[0, W*self.resolution, 0, H*self.resolution],
                      cmap='viridis', alpha=0.45)
        if hasattr(self, "covered"):
            cov = np.ma.masked_where(self.covered == 0, self.covered)
            ax.imshow(cov, origin='lower',
                      extent=[0, self.W*self.resolution, 0, self.H*self.resolution],
                      cmap='autumn', alpha=0.35)

        # 未来 K 个节点（可视化）
        if self.high_nodes_seq:
            xs, ys = [], []; plotted = 0
            for (r, c) in self.high_nodes_seq:
                if plotted >= self.show_k_next: break
                if self._nodes_used[r, c] == 1 or self.covered[r, c] == 1: continue
                wx, wy = (c + 0.5)*self.resolution, (r + 0.5)*self.resolution
                ax.plot(wx, wy, marker='o', markersize=3)
                ax.text(wx+0.05, wy+0.05, f"{plotted+1}", fontsize=7)
                xs.append(wx); ys.append(wy); plotted += 1
            if len(xs) >= 2: ax.plot(xs, ys, linewidth=1.0)

        ax.set_xlim(0, self.W*self.resolution); ax.set_ylim(0, self.H*self.resolution)
        ax.set_aspect('equal'); ax.set_title(f"Step {self._steps} | Subgoals {self.subgoal_count} | order={self.order_strategy}")

        if mode == "rgb_array":
            self._fig.canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            buf = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        else:
            import time as _t; _t.sleep(1/self.metadata["render_fps"])
        self._plot_high_level_nodes(ax, show_labels=False, show_path=True, k_next=None)

    # ====== 初始化：构建巡游节点并选择首个可达子目标 ======
    def _init_fields_and_tour(self):
        # 先做一次覆盖和巡游节点构建，在 reset 时也会再做一遍
        self.covered = np.zeros((self.H, self.W), dtype=np.uint8)
        self._mark_coverage(self.start_xytheta[:2])
        self._build_high_level_nodes_tour()
        self._pick_first_reachable_goal()

    # ---------- helpers ----------
    def _world_to_grid_rc(self, xy: np.ndarray) -> Tuple[int, int]:
        r = int(np.floor(xy[1] / self.resolution))
        c = int(np.floor(xy[0] / self.resolution))
        r = np.clip(r, 0, self.H-1)
        c = np.clip(c, 0, self.W-1)
        return r, c

    def _mark_coverage(self, xy: np.ndarray):
        """可见性覆盖：射线投射，不透墙（在 bottom-origin free_mask 上）"""
        rad_cells = int(np.ceil(self.cover_radius_m / self.resolution))
        cx = float(xy[0] / self.resolution)  # 列方向
        cy = float(xy[1] / self.resolution)  # 行方向
        angles = np.linspace(0.0, 2*np.pi, self.cover_n_rays, endpoint=False)
        for ang in angles:
            dx = np.cos(ang); dy = np.sin(ang)
            for step in range(rad_cells + 1):
                gx = int(np.floor(cx + dx * step))
                gy = int(np.floor(cy + dy * step))
                if not (0 <= gy < self.H and 0 <= gx < self.W):
                    break
                if self.free_mask[gy, gx] == 0:
                    break
                self.covered[gy, gx] = 1

    def _compute_distance_map(self, grid: np.ndarray, goal_rc: np.ndarray) -> np.ndarray:
        """
        grid: 1=free, 0=obs ；goal_rc=(r,c)
        返回到 goal 的 8 邻域 geodesic 距离（轴向=1, 对角=√2）。不可达为 +inf
        """
        import heapq
        H, W = grid.shape
        free = (grid == 1)
        INF = np.inf
        dist = np.full((H, W), INF, dtype=np.float32)

        gr, gc = int(goal_rc[0]), int(goal_rc[1])
        if not (0 <= gr < H and 0 <= gc < W):
            return dist
        if not free[gr, gc]:
            # 把目标弹到最近 free（BFS 4 邻域即可）
            from collections import deque
            vis = np.zeros_like(free, dtype=np.uint8)
            dq = deque([(np.clip(gr,0,H-1), np.clip(gc,0,W-1))]); vis[dq[0]] = 1
            found = None
            while dq and found is None:
                r0, c0 = dq.popleft()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rn, cn = r0+dr, c0+dc
                    if 0<=rn<H and 0<=cn<W and not vis[rn,cn]:
                        if free[rn,cn]: found=(rn,cn); break
                        vis[rn,cn]=1; dq.append((rn,cn))
            if found is None: return dist
            gr, gc = found

        dist[gr, gc] = 0.0
        pq = [(0.0, gr, gc)]
        # 8 邻域 + 代价
        N8 = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
            (-1,-1,2**0.5),(-1,1,2**0.5),(1,-1,2**0.5),(1,1,2**0.5)]
        while pq:
            d, r, c = heapq.heappop(pq)
            if d != dist[r, c]: continue
            for dr, dc, w in N8:
                rn, cn = r + dr, c + dc
                if 0 <= rn < H and 0 <= cn < W and free[rn, cn]:
                    nd = d + w
                    if nd < dist[rn, cn]:
                        dist[rn, cn] = nd
                        heapq.heappush(pq, (nd, rn, cn))
        return dist

    def _build_grid_gradient(self, dist_map: np.ndarray, free_mask: np.ndarray):
        H, W = dist_map.shape
        gx = np.zeros_like(dist_map, dtype=np.float32)
        gy = np.zeros_like(dist_map, dtype=np.float32)
        neighs = [(-1,0),(1,0),(0,-1),(0,1)]
        for i in range(H):
            for j in range(W):
                if free_mask[i, j] == 0 or not np.isfinite(dist_map[i, j]):
                    continue
                best_d = dist_map[i, j]
                bi, bj = i, j
                for di, dj in neighs:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < H and 0 <= nj < W and free_mask[ni, nj] == 1 and np.isfinite(dist_map[ni, nj]):
                        if dist_map[ni, nj] < best_d:
                            best_d = dist_map[ni, nj]
                            bi, bj = ni, nj
                if (bi != i) or (bj != j):
                    vx = (bj - j); vy = (bi - i)
                    norm = (vx*vx + vy*vy) ** 0.5
                    if norm > 0:
                        gx[i, j] = vx / norm
                        gy[i, j] = vy / norm
        return gx, gy

    # ---------- tour 节点生成 + 排序 ----------
    def _tour_pick_in_cell(self, points: List[Tuple[int,int]], ri: int, cj: int, cell: int) -> Optional[Tuple[int,int]]:
        if not points:
            return None
        if self.tour_pick == "center":
            r0 = ri*cell; c0 = cj*cell
            r_center = r0 + min(cell, self.H - r0)/2.0
            c_center = c0 + min(cell, self.W - c0)/2.0
            best, best_d2 = None, 1e18
            for (r, c) in points:
                d2 = (r - r_center)**2 + (c - c_center)**2
                if d2 < best_d2:
                    best, best_d2 = (r, c), d2
            return best
        else:  # medoid
            pts = np.array(points, dtype=int)
            dsum = np.abs(pts[:,None,:] - pts[None,:,:]).sum(axis=2).sum(axis=1)
            idx = int(np.argmin(dsum))
            return tuple(map(int, pts[idx]))

    # === REPLACE: 高层节点仅从“起点连通的 nav_mask==1”里选，并做轻度评分 ===
    def _build_high_level_nodes_tour(self):
        cell = max(1, int(round(self.tour_cell_m / self.resolution)))
        Hc = int(np.ceil(self.H / cell)); Wc = int(np.ceil(self.W / cell))

        # 只在起点可达区域取点
        rs0, cs0 = self._world_to_grid_rc(self.start_xytheta[:2])
        reach = self._reachable_mask_from((rs0, cs0))

        buckets = [[[] for _ in range(Wc)] for __ in range(Hc)]
        for r in range(self.H):
            if not np.any(self.nav_mask[r, :]): continue
            ri = r // cell
            for c in range(self.W):
                if self.nav_mask[r, c] != 1 or reach[r, c] != 1:  # 必须可通行且与起点连通
                    continue
                cj = c // cell
                buckets[ri][cj].append((r, c))

        # 每个 supercell 选一个代表点：越居中/越空旷越好
        def pick_rep(points, ri, cj):
            if not points: return None
            r0 = ri*cell; c0 = cj*cell
            r_center = r0 + min(cell, self.H - r0)/2.0
            c_center = c0 + min(cell, self.W - c0)/2.0
            best, best_score = None, -1e18
            for (r, c) in points:
                # 简单清晰：离 supercell 中心越近 + 8 邻域 nav 可行邻居越多
                d2 = (r - r_center)**2 + (c - c_center)**2
                neigh_ok = 0
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        if dr==0 and dc==0: continue
                        rn, cn = r+dr, c+dc
                        if 0<=rn<self.H and 0<=cn<self.W and self.nav_mask[rn,cn]==1:
                            neigh_ok += 1
                score = (+ 0.1*neigh_ok) - 0.001*d2
                if score > best_score:
                    best, best_score = (r, c), score
            return best

        reps = []
        for ri in range(Hc):
            for cj in range(Wc):
                rep = pick_rep(buckets[ri][cj], ri, cj)
                if rep is not None:
                    reps.append(rep)

        # geodesic 最近邻串起，再做一次 2-opt
        def dist_rc(a, b):
            return float(np.hypot(a[0]-b[0], a[1]-b[1]))
        if not reps:
            self.high_nodes_seq = []
            self._nodes_used[:] = 0
            return

        # 从“离起点 geodesic 最近”的开始
        dmap0 = self._compute_distance_map(self.nav_mask, np.array([rs0, cs0]))
        reps = [rc for rc in reps if np.isfinite(dmap0[rc[0], rc[1]])]
        if not reps:
            self.high_nodes_seq = []
            self._nodes_used[:] = 0
            return

        reps_sorted = sorted(reps, key=lambda rc: dmap0[rc[0], rc[1]])
        order = [reps_sorted[0]]
        pool = set(reps_sorted[1:])
        while pool:
            last = order[-1]
            # 用 geodesic 距离最近邻
            dmap = self._compute_distance_map(self.nav_mask, np.array([last[0], last[1]]))
            best, best_d = None, 1e18
            for rc in list(pool):
                d = dmap[rc[0], rc[1]]
                if d < best_d:
                    best, best_d = rc, d
            if best is None or not np.isfinite(best_d):
                # 退化：用欧氏最近
                best = min(pool, key=lambda rc: dist_rc(last, rc))
            order.append(best); pool.remove(best)

        # 2-opt（欧氏）去抖
        def two_opt(route):
            N = len(route)
            if N < 4: return route
            improved = True
            while improved:
                improved = False
                for i in range(N-3):
                    for k in range(i+2, N-1):
                        a,b = route[i], route[i+1]
                        c,d = route[k], route[k+1]
                        if dist_rc(a,c) + dist_rc(b,d) + 1e-9 < dist_rc(a,b) + dist_rc(c,d):
                            route[i+1:k+1] = reversed(route[i+1:k+1])
                            improved = True
            return route

        self.high_nodes_seq = two_opt(order)
        self._nodes_used[:] = 0


    # === REPLACE: 设定子目标，并基于 nav_mask 生成几何引导 ===
    def _set_goal_by_rc(self, r: int, c: int):
        self.goal_xytheta = np.array([c*self.resolution, r*self.resolution, 0.0], dtype=np.float32)
        self.goal_state   = self.dynamics.get_state_from_xytheta(self.goal_xytheta)

        # 使用 nav_mask 生成 geodesic 场（更匹配可通行域）
        dist_arr = self._compute_distance_map(self.nav_mask, np.array([r, c]))

        # 可达性标记（供奖励塑形/观测）
        self._goal_reachable_from_robot = True
        if self.state is not None:
            rr, cc = self._world_to_grid_rc(self.state[:2])
            self._goal_reachable_from_robot = np.isfinite(dist_arr[rr, cc])

        # 构梯度场（只在 nav_mask==1 上有效）
        unreachable = np.isinf(dist_arr)
        reachable   = ~unreachable
        self.max_potential = float(np.max(dist_arr[reachable])) if np.any(reachable) else 0.0
        dm = dist_arr.copy()
        dm[unreachable] = self.max_potential
        self.dist_map = dm
        self.grad_x, self.grad_y = self._build_grid_gradient(self.dist_map, self.nav_mask)
        valid = (self.nav_mask == 1) & (~unreachable)
        self.grad_x[~valid] = 0.0; self.grad_y[~valid] = 0.0

        # === A* 微路径（保存为世界系 waypoints） ===
        rr0, cc0 = self._world_to_grid_rc(self.state[:2]) if self.state is not None else self._world_to_grid_rc(self.start_xytheta[:2])
        path_rc = self._astar_on_nav_mask((rr0, cc0), (r, c))
        self._guide_wp_xy = []     # [ [x,y], ... ]
        self._guide_idx = 0
        if path_rc is not None and len(path_rc) >= 2:
            for (pr, pc) in path_rc:
                x = (pc + 0.5) * self.resolution
                y = (pr + 0.5) * self.resolution
                self._guide_wp_xy.append([float(x), float(y)])

        # reset per-goal stats
        self._no_progress_steps = 0
        self._best_phi_this_goal = np.inf
        self._best_goal_dist = np.inf
        self._collision_steps = 0
        self._goal_steps = 0

    def _pick_first_reachable_goal(self):
        # 预选“可达”的首目标（一次就跳过不可达/已覆盖的）
        while True:
            first_rc = self._pick_next_goal_rc()
            if first_rc is None:
                # 找不到就把当前位置作为临时目标，避免空场
                self._set_goal_by_rc(*self._world_to_grid_rc(self.start_xytheta[:2]))
                break
            r, c = first_rc
            dist_arr = self._compute_distance_map(self.nav_mask, np.array([r, c]))
            rr, cc = self._world_to_grid_rc(self.start_xytheta[:2])
            if np.isfinite(dist_arr[rr, cc]):
                self._set_goal_by_rc(r, c)
                self.subgoal_count += 1
                break
            # 不可达则继续挑下一个（已在 _pick_next_goal_rc 标为 used）

    # === REPLACE: 观测里用“当前引导 waypoint”而不是最终子目标 ===
    def _get_obs(self) -> Dict:
        x, y, theta = self.state[:3]

        scans = np.full(self.n_lasers, self.max_scan_dist, dtype=np.float32)
        gw_x, gw_y = self.world_cc.grid.shape
        max_steps = int(self.sensor_range_m / self.resolution)
        for idx, ang in enumerate(self.laser_angles):
            dx = np.cos(theta+ang); dy = np.sin(theta+ang)
            for s in range(max_steps):
                px = x + dx*s*self.resolution
                py = y + dy*s*self.resolution
                ix = int(np.floor(px/self.resolution))
                iy = int(np.floor(py/self.resolution))
                if not (0 <= ix < gw_x and 0 <= iy < gw_y) or self.world_cc.grid[ix, iy] != 0:
                    scans[idx] = s * self.resolution
                    break

        # === 当前引导点（若无，则回退到真实子目标）
        if getattr(self, "_guide_wp_xy", None) and self._guide_idx < len(self._guide_wp_xy):
            tx, ty = self._guide_wp_xy[self._guide_idx]
        else:
            tx, ty = self.goal_state[0], self.goal_state[1]

        # 机体坐标
        dxg = tx - x; dyg = ty - y
        gx =  dxg*np.cos(-theta) - dyg*np.sin(-theta)
        gy =  dxg*np.sin(-theta) + dyg*np.cos(-theta)
        goal_vec = np.array([gx, gy], dtype=np.float32)

        gx_w = self._interpolate_grad(self.grad_x, self.state[:2])
        gy_w = self._interpolate_grad(self.grad_y, self.state[:2])
        gx_r = gx_w*np.cos(-theta) - gy_w*np.sin(-theta)
        gy_r = gx_w*np.sin(-theta) + gy_w*np.cos(-theta)

        phi = self._interpolate_dist(self.state[:2])
        phi_norm = phi / (self.max_potential + 1e-6)

        hist = np.array(self.action_history, dtype=np.int64)
        return {
            "scan":        scans,
            "goal_vec":    goal_vec,
            "action_mask": self.action_masks(),
            "hist":        hist,
            "dist_grad":   np.array([gx_r, gy_r], dtype=np.float32),
            "dist_phi":    np.array([phi_norm], dtype=np.float32),
        }


    # ---------- interpolation ----------
    def _interpolate_dist(self, xy: np.ndarray) -> float:
        x, y = xy
        i_f = np.clip(y/self.resolution, 0, self.H-1)
        j_f = np.clip(x/self.resolution, 0, self.W-1)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = min(i0+1, self.H-1), min(j0+1, self.W-1)
        di, dj = i_f - i0, j_f - j0
        d00 = self.dist_map[i0, j0]; d10 = self.dist_map[i1, j0]
        d01 = self.dist_map[i0, j1]; d11 = self.dist_map[i1, j1]
        return (
            d00*(1-di)*(1-dj)
          + d10*di*(1-dj)
          + d01*(1-di)*dj
          + d11*di*dj
        )

    def _interpolate_grad(self, grad_map: np.ndarray, xy: np.ndarray) -> float:
        x, y = xy
        i_f = np.clip(y/self.resolution, 0, self.H-1)
        j_f = np.clip(x/self.resolution, 0, self.W-1)
        i0, j0 = int(np.floor(i_f)), int(np.floor(j_f))
        i1, j1 = min(i0+1, self.H-1), min(j0+1, self.W-1)
        di, dj = i_f - i0, j_f - j0
        g00 = grad_map[i0, j0]; g10 = grad_map[i1, j0]
        g01 = grad_map[i0, j1]; g11 = grad_map[i1, j1]
        return (
            g00*(1-di)*(1-dj)
          + g10*di*(1-dj)
          + g01*(1-di)*dj
          + g11*di*dj
        )

    # ---------- runtime setters（可选） ----------
    def set_cover_radius(self, r_m: float):
        self.cover_radius_m = float(r_m)

    def set_sensor_range(self, r_m: float):
        self.sensor_range_m = float(r_m)
        self.max_scan_dist  = float(r_m)

    # ---------- export helper（可选） ----------
    def get_high_level_nodes_world(self):
        res = []
        for (r, c) in self.high_nodes_seq:
            x = (c + 0.5) * self.resolution
            y = (r + 0.5) * self.resolution
            res.append([float(x), float(y)])
        return res

    def export_nodes_json(self, path: str):
        import json
        nodes = self.get_high_level_nodes_world()
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"nodes_xy": nodes}, f, ensure_ascii=False, indent=2)
        return path

    # ====== restore：支持 rl_sas 的缓存回退 ======
    def restore(self, state: np.ndarray, steps: int, action_history: Optional[np.ndarray] = None):
        """
        把环境硬重置到任意节点（用于 rl_sas 回退后继续滚）：
          - state: [x, y, theta]
          - steps: 已经走过的离散步数（用于恢复时间 t = steps * dt_default）
          - action_history: 动作序列（长度<=hist_len）；用于恢复策略的历史依赖
        返回：一帧新的 observation（原始 obs，不是归一化后的）
        """
        # 位置/朝向与步数
        self.state = np.array(state, dtype=np.float32).copy()
        self._steps = int(steps)

        # 恢复动作历史：先用 0 填满，再把传入历史的尾部接上
        self.action_history.clear()
        for _ in range(self.hist_len):
            self.action_history.append(0)
        if action_history is not None:
            tail = list(map(int, action_history[-self.hist_len:]))
            for a in tail:
                self.action_history.append(a)

        # 注意：coverage 的内部统计（如 _no_progress_steps/_goal_steps 等）
        # 这里不强制清零，保留原值以便“回退后继续滚”的直觉；如需改为清零，可在此处重置。
        return self._get_obs()

    def _plot_high_level_nodes(self, ax, show_labels=False, show_path=True, k_next=None):
        """
        在 ax 上可视化 high-level 节点：
        - 绿色：未覆盖 & 未使用（未来会访问）
        - 灰色：已覆盖（covered==1）
        - 橙色：已标记使用（_nodes_used==1），但尚未覆盖（比如刚被选中过/正在路上）
        参数：
        show_labels: 是否给每个“未覆盖&未使用”的节点标序号
        show_path:   是否把“未覆盖&未使用”的节点按访问顺序连线
        k_next:      仅显示未来前 k 个（不传则显示全部 future 节点）
        """
        if not self.high_nodes_seq:
            return

        xs_future, ys_future = [], []
        xs_used, ys_used = [], []
        xs_cov, ys_cov = [], []

        count_future = 0
        for (r, c) in self.high_nodes_seq:
            x = (c + 0.5) * self.resolution
            y = (r + 0.5) * self.resolution

            if self.covered[r, c] == 1:
                xs_cov.append(x); ys_cov.append(y)
                continue

            if self._nodes_used[r, c] == 1:
                # 已被取用过（可能正在进行/或曾是目标）但还没被覆盖
                xs_used.append(x); ys_used.append(y)
                continue

            # 未来将访问的节点
            if (k_next is None) or (count_future < k_next):
                xs_future.append(x); ys_future.append(y)
                count_future += 1

        # 画覆盖热力在先（若需要）
        # 已在 render() 里做过，这里只画节点
        if xs_cov:
            ax.scatter(xs_cov, ys_cov, s=12, marker='o', alpha=0.5, label='covered nodes', color='grey')
        if xs_used:
            ax.scatter(xs_used, ys_used, s=18, marker='^', alpha=0.8, label='used (picked) nodes', color='orange')
        if xs_future:
            ax.scatter(xs_future, ys_future, s=20, marker='o', alpha=0.9, label='future nodes', color='green')
            if show_path and len(xs_future) >= 2:
                ax.plot(xs_future, ys_future, linewidth=1.2, alpha=0.9)

            if show_labels:
                for i, (x, y) in enumerate(zip(xs_future, ys_future), start=1):
                    ax.text(x+0.05, y+0.05, str(i), fontsize=7)

        ax.legend(loc='upper right', fontsize=8)

    def _manhattan_dt(self, free_mask: np.ndarray) -> np.ndarray:
        """
        到最近障碍的 L1 距离（格子数）的两遍扫描近似；free=1 才有值，障碍为 0。
        """
        H, W = free_mask.shape
        INF = 10**9
        d = np.full((H, W), INF, dtype=np.int32)
        d[free_mask == 0] = 0

        # forward
        for i in range(H):
            for j in range(W):
                if d[i, j] == 0: 
                    continue
                if i > 0: d[i, j] = min(d[i, j], d[i-1, j] + 1)
                if j > 0: d[i, j] = min(d[i, j], d[i, j-1] + 1)
        # backward
        for i in range(H-1, -1, -1):
            for j in range(W-1, -1, -1):
                if i+1 < H: d[i, j] = min(d[i, j], d[i+1, j] + 1)
                if j+1 < W: d[i, j] = min(d[i, j], d[i, j+1] + 1)

        d[free_mask == 0] = 0
        return d

    def _visibility_count(self, r: int, c: int, rad_cells: int, n_rays: int) -> int:
        """
        从格子 (r,c) 出发，按 n_rays 条射线、半径 rad_cells，统计可见 free 栅格数量（不透墙）。
        与 _mark_coverage 一致的几何近似，用于代表点评分。
        """
        H, W = self.H, self.W
        cx = c + 0.5
        cy = r + 0.5
        cnt = 0
        for ang in np.linspace(0.0, 2*np.pi, int(n_rays), endpoint=False):
            dx = np.cos(ang); dy = np.sin(ang)
            for step in range(rad_cells + 1):
                gx = int(np.floor(cx + dx * step))
                gy = int(np.floor(cy + dy * step))
                if not (0 <= gy < H and 0 <= gx < W): break
                if self.free_mask[gy, gx] == 0: break
                cnt += 1
        return cnt

    def _build_nav_mask(self, n_theta_samples: int = 8, border_margin_m: float = 0.0, require_all: bool = True):
        """
        生成 nav_mask：
        - 采样 n_theta_samples 个朝向，用 world_cc.isValid 检查足迹放置安全
        - require_all=True 表示“所有采样朝向都安全”才记为可通行（更保守，不贴墙）
            若你想放宽，可传 require_all=False（任一朝向可行即可）
        - border_margin_m 给边界留白，避免贴边
        另外生成 clearance_cells（到障碍的 L1 近似距离，单位=格子）
        """
        H, W = self.H, self.W
        xs = (np.arange(W) + 0.5) * self.resolution
        ys = (np.arange(H) + 0.5) * self.resolution
        XX, YY = np.meshgrid(xs, ys)                          # (H,W)
        thetas = np.linspace(-np.pi, np.pi, int(n_theta_samples), endpoint=False)

        valid_any = np.zeros((H, W), dtype=bool)
        valid_all = np.ones((H, W), dtype=bool)

        # 分朝向批量检查（避免一次性 H*W*theta 的超大数组）
        for th in thetas:
            # 世界坐标下的 pose 列表：[(x,y,th), ...]
            xytheta = np.stack([XX.ravel(), YY.ravel(), np.full(H*W, th, dtype=np.float32)], axis=1)
            ok = self.world_cc.isValid(self.footprint, xytheta).reshape(H, W)
            valid_any |= ok
            valid_all &= ok

        nav_core = valid_all if require_all else valid_any
        nav = (self.free_mask == 1) & nav_core

        # 边界留白
        margin = int(np.ceil(border_margin_m / self.resolution))
        if margin > 0:
            nav[:margin, :]  = False
            nav[-margin:, :] = False
            nav[:, :margin]  = False
            nav[:, -margin:] = False

        self.nav_mask = nav.astype(np.uint8)
        self.clearance_cells = self._manhattan_dt(self.free_mask)

    def _reachable_mask_from(self, start_rc):
        """
        在 nav_mask 上做 BFS，返回起点连通区域的掩码（uint8: 1=可达, 0=不可达）。
        注意：使用 while+popleft 的标准写法，避免 deque 在迭代时被修改。
        """
        from collections import deque
        H, W = self.H, self.W

        vis = np.zeros((H, W), dtype=np.uint8)
        sr, sc = int(start_rc[0]), int(start_rc[1])

        # 起点越界或起点不可站立，直接返回全 0
        if not (0 <= sr < H and 0 <= sc < W) or self.nav_mask[sr, sc] == 0:
            return vis

        dq = deque()
        dq.append((sr, sc))
        vis[sr, sc] = 1

        # 8 邻域连通（与 A* / geodesic 使用一致）
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                    (-1,-1),(-1,1),(1,-1),(1,1)]

        while dq:
            r, c = dq.popleft()
            for dr, dc in neighbors:
                rn, cn = r + dr, c + dc
                if 0 <= rn < H and 0 <= cn < W and vis[rn, cn] == 0 and self.nav_mask[rn, cn] == 1:
                    vis[rn, cn] = 1
                    dq.append((rn, cn))

        return vis

    # === NEW: 在 nav_mask 上跑 A*，返回 rc 路径（含起终点） ===
    def _astar_on_nav_mask(self, start_rc, goal_rc):
        import heapq, math
        H, W = self.H, self.W
        if self.nav_mask[start_rc] == 0 or self.nav_mask[goal_rc] == 0:
            return None
        def h(rc1, rc2):
            dr = abs(rc1[0]-rc2[0]); dc = abs(rc1[1]-rc2[1])
            # Octile heuristic
            return (max(dr, dc) - min(dr, dc)) + (2**0.5) * min(dr, dc)
        N8 = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
            (-1,-1,2**0.5),(-1,1,2**0.5),(1,-1,2**0.5),(1,1,2**0.5)]
        openq = []
        g = {start_rc: 0.0}
        parent = {start_rc: None}
        heapq.heappush(openq, (h(start_rc, goal_rc), 0.0, start_rc))
        seen = set()
        while openq:
            f, gcost, cur = heapq.heappop(openq)
            if cur in seen: continue
            seen.add(cur)
            if cur == goal_rc:
                # 回溯
                path = []
                node = cur
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            r, c = cur
            for dr, dc, w in N8:
                rn, cn = r+dr, c+dc
                nxt = (rn, cn)
                if not (0<=rn<H and 0<=cn<W): continue
                if self.nav_mask[rn, cn] == 0: continue
                ng = g[cur] + w
                if ng < g.get(nxt, np.inf):
                    g[nxt] = ng
                    parent[nxt] = cur
                    heapq.heappush(openq, (ng + h(nxt, goal_rc), ng, nxt))
        return None

    def _pick_next_goal_rc(self) -> Optional[Tuple[int, int]]:
        """
        从 high_nodes_seq 里挑下一个子目标 (r,c)：
        - 跳过：已被标记 used 的 / 已覆盖的 / nav_mask==0 的
        - 选到一个后，立刻把 _nodes_used[r,c] 置 1
        - 若没有可用节点则返回 None
        """
        if not self.high_nodes_seq:
            return None

        for (r, c) in self.high_nodes_seq:
            # 已用过
            if self._nodes_used[r, c] == 1:
                continue
            # 该代表点已经在运行中被覆盖了
            if self.covered[r, c] == 1:
                self._nodes_used[r, c] = 1
                continue
            # 不在可通行导航掩码上（考虑足迹/朝向），也跳过
            if getattr(self, "nav_mask", None) is not None and self.nav_mask[r, c] != 1:
                self._nodes_used[r, c] = 1
                continue

            # 选定
            self._nodes_used[r, c] = 1
            return int(r), int(c)

        return None
