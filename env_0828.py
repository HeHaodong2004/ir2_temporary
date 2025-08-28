import os
import random
import numpy as np
from copy import deepcopy
from skimage import io
from skimage.measure import block_reduce

from sensor import sensor_work
from utils import *
from parameter import *

class Env:
    def __init__(self, episode_index, plot=False, test=False):
        self.episode_index = episode_index
        self.plot = plot
        self.test = test

        # 读取地图（保持你单机的数据集结构）
        self.ground_truth, initial_cell, self.map_path = self.import_ground_truth(episode_index)
        self.cell_size = CELL_SIZE
        self.ground_truth_size = np.shape(self.ground_truth)

        # —— 多机信念图 —— #
        self.global_belief  = np.ones(self.ground_truth_size) * UNKNOWN
        self.agent_beliefs  = [np.ones_like(self.global_belief) * UNKNOWN for _ in range(N_AGENTS)]

        # 坐标原点（与单机一致）
        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)

        # 传感器与指标
        self.sensor_range = SENSOR_RANGE
        self.explored_rate = 0.0
        self.done = False
        self.total_travel_dist = 0.0
        # —— 新增：逐机器人累计路程与最大路程 —— #
        self.agent_travel_dists = np.zeros(N_AGENTS, dtype=float)
        self.max_travel_dist = 0.0

        # 初次观测：先让起点更新一次
        self.global_belief = sensor_work(initial_cell,
                                         round(self.sensor_range / self.cell_size),
                                         self.global_belief,
                                         self.ground_truth)
        for i in range(N_AGENTS):
            self.agent_beliefs[i] = deepcopy(self.global_belief)

        # 产生多机器人初始位姿（从 FREE 的更新窗口里采样 N_AGENTS 个）
        tmp_info = MapInfo(self.agent_beliefs[0], self.belief_origin_x, self.belief_origin_y, self.cell_size)
        free_nodes, _ = get_updating_node_coords(np.array([0.0, 0.0]), tmp_info)
        choice = np.random.choice(free_nodes.shape[0], N_AGENTS, replace=False)
        self.robot_locations = np.array(free_nodes[choice])

        # 起点也做一次感知
        for cell in get_cell_position_from_coords(self.robot_locations, tmp_info).reshape(-1, 2):
            self.global_belief = sensor_work(cell, round(self.sensor_range / self.cell_size),
                                             self.global_belief, self.ground_truth)
        for i in range(N_AGENTS):
            self.agent_beliefs[i] = deepcopy(self.global_belief)

        # belief_info（默认给 agent0，用 get_agent_map(i) 拿其它 agent 的）
        self.belief_info = MapInfo(self.agent_beliefs[0], self.belief_origin_x, self.belief_origin_y, self.cell_size)
        # ground truth for critic
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        self.global_frontiers = get_frontier_in_map(self.belief_info)
        if self.plot:
            self.frame_files = []

    # ---------- dataset ----------
    def import_ground_truth(self, episode_index):
        map_dir = f'maps_second' if not self.test else f'dataset/maps_eval'
        map_list = []
        for root, _, files in os.walk(map_dir):
            for f in files:
                map_list.append(os.path.join(root, f))
        if not self.test:
            rng = random.Random(1)
            rng.shuffle(map_list)

        idx = episode_index % len(map_list)
        gt = (io.imread(map_list[idx], 1)).astype(int)
        gt = block_reduce(gt, (2, 2), np.min)

        # 起点cell（与单机一致）
        robot_cell = np.nonzero(gt == 208)
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])

        gt = (gt > 150) | ((gt <= 80) & (gt >= 50))
        gt = gt * 254 + 1
        return gt, robot_cell, map_list[idx]

    # ---------- comms ----------
    def _compute_comm_groups(self):
        """按 COMMS_RANGE 求连通分量（多跳可连）。"""
        adj = np.zeros((N_AGENTS, N_AGENTS), dtype=bool)
        for i in range(N_AGENTS):
            for j in range(i+1, N_AGENTS):
                if np.linalg.norm(self.robot_locations[i] - self.robot_locations[j]) <= COMMS_RANGE:
                    adj[i, j] = adj[j, i] = True
        groups, unseen = [], set(range(N_AGENTS))
        while unseen:
            r = unseen.pop()
            stack = [r]; comp = {r}
            while stack:
                u = stack.pop()
                for v in range(N_AGENTS):
                    if v in unseen and adj[u, v]:
                        unseen.remove(v)
                        comp.add(v)
                        stack.append(v)
            groups.append(comp)
        return groups

    def _merge_agent_beliefs(self, groups):
        """组内像素级合并：Known 覆盖 Unknown。"""
        for g in groups:
            merged = np.ones_like(self.global_belief) * UNKNOWN
            for i in g:
                known = (self.agent_beliefs[i] != UNKNOWN)
                merged[known] = self.agent_beliefs[i][known]
            for i in g:
                self.agent_beliefs[i] = merged.copy()

    # ---------- step ----------
    def step(self, next_waypoint, agent_id):
        old = self.robot_locations[agent_id]
        dist = np.linalg.norm(next_waypoint - old)
        self.total_travel_dist += dist
        # —— 新增：累计每个体的路程 —— #
        self.agent_travel_dists[agent_id] += dist

        # 1) 位置更新
        self.robot_locations[agent_id] = next_waypoint

        # 2) 观测写回 global + own
        cell = get_cell_position_from_coords(next_waypoint, self.belief_info)
        self.global_belief = sensor_work(cell, round(self.sensor_range / self.cell_size),
                                         self.global_belief, self.ground_truth)
        self.agent_beliefs[agent_id] = sensor_work(cell, round(self.sensor_range / self.cell_size),
                                                   self.agent_beliefs[agent_id], self.ground_truth)

        # 3) 组内合并
        groups = self._compute_comm_groups()
        self._merge_agent_beliefs(groups)

        # 4) 刷新该 agent 的 belief_info（其余由 get_agent_map 获取）
        self.belief_info = MapInfo(self.agent_beliefs[agent_id],
                                   self.belief_origin_x, self.belief_origin_y, self.cell_size)

    # ---------- metrics ----------
    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.global_belief == FREE) / np.sum(self.ground_truth == FREE)

    def calculate_reward(self):
        """团队奖励：全局前沿减少量"""
        self.evaluate_exploration_rate()
        binfo = MapInfo(self.global_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        global_frontiers = get_frontier_in_map(binfo)
        if len(global_frontiers) == 0:
            delta = len(self.global_frontiers)
        else:
            observed = self.global_frontiers - global_frontiers
            delta = len(observed)
        self.global_frontiers = global_frontiers
        return delta / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

    # ---------- helpers ----------
    def get_agent_map(self, agent_id):
        return MapInfo(self.agent_beliefs[agent_id],
                       self.belief_origin_x, self.belief_origin_y, self.cell_size)

    def get_total_travel(self):
        return self.total_travel_dist

    # —— 新增：暴露 per-agent 和 max 路程 —— #
    def get_agent_travel(self):
        return self.agent_travel_dists.copy()

    def get_max_travel(self):
        return float(self.max_travel_dist)
