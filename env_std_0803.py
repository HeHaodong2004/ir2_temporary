import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy
import numpy as np

from sensor import sensor_work
from parameter import *
from utils import *


class Env:
    def __init__(self, episode_index, plot=False):
        # —— 原有初始化 —— #
        self.episode_index = episode_index
        self.plot = plot
        self.ground_truth, initial_cell = self.import_ground_truth(episode_index)
        self.ground_truth_size = np.shape(self.ground_truth)
        self.cell_size = CELL_SIZE

        self.robot_belief = np.ones(self.ground_truth_size) * UNKNOWN
        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)

        self.sensor_range = SENSOR_RANGE
        self.explored_rate = 0
        self.done = False
        
        self.total_travel_dist = 0.0 

        # —— 新增：全局 Belief + 每个 agent 各自的 Belief —— #
        self.global_belief = deepcopy(self.robot_belief)
        self.agent_beliefs = [
            deepcopy(self.robot_belief) for _ in range(N_AGENTS)
        ]
        
        # —— 新增：初始化 discovery_map，-1 表示“尚未被任何 agent 首次观测” —— #
        #    discovery_map[i,j] 将存储第一个看到 (i,j) 的 agent_id
        self.discovery_map = np.full(self.ground_truth_size, -1, dtype=int)
        
        # 原有的初始感知，改写为同时更新 global 和 agent_beliefs[*]
        self.global_belief = sensor_work(initial_cell,
                                         self.sensor_range / self.cell_size,
                                         self.global_belief,
                                         self.ground_truth)
        for i in range(N_AGENTS):
            self.agent_beliefs[i] = deepcopy(self.global_belief)

        # 构造 belief_info 暂时指向 agent 0（后面 step 调用会覆盖）
        self.belief_info = MapInfo(self.agent_beliefs[0],
                                   self.belief_origin_x,
                                   self.belief_origin_y,
                                   self.cell_size)

        # 其余起点选取、二次感知逻辑保持不变……
        free, _ = get_updating_node_coords(np.array([0.0, 0.0]), self.belief_info)
        choice = np.random.choice(free.shape[0], N_AGENTS, replace=True)
        starts = free[choice]
        self.robot_locations = np.array(starts)

        for robot_cell in get_cell_position_from_coords(self.robot_locations, self.belief_info).reshape(-1,2):
            self.global_belief = sensor_work(robot_cell,
                                             self.sensor_range / self.cell_size,
                                             self.global_belief,
                                             self.ground_truth)
            for i in range(N_AGENTS):
                self.agent_beliefs[i] = deepcopy(self.global_belief)

        self.old_belief = deepcopy(self.global_belief)
        self.global_frontiers = get_frontier_in_map(self.belief_info)

        if self.plot:
            self.frame_files = []

    def import_ground_truth(self, episode_index):
        map_dir = f'maps_train'
        map_list = os.listdir(map_dir)
        map_index = episode_index % np.size(map_list)
        ground_truth = (io.imread(map_dir + '/' + map_list[map_index], 1)).astype(int)

        ground_truth = block_reduce(ground_truth, (2,2), np.min)

        robot_cell = np.array(np.nonzero(ground_truth == 208))
        robot_cell = np.array([robot_cell[1, 10], robot_cell[0, 10]])

        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1

        return ground_truth, robot_cell


    def update_robot_location(self, robot_location):
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])

    def update_robot_belief(self, robot_cell):
        self.robot_belief = sensor_work(robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)

    def calculate_reward(self):
        reward = 0

        global_frontiers = get_frontier_in_map(self.belief_info)
        if len(global_frontiers) == 0:
            delta_num = len(self.global_frontiers)
        else:
            observed_frontiers = self.global_frontiers - global_frontiers
            delta_num = len(observed_frontiers)

        reward += delta_num / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        return reward

    def check_done(self):
        if np.sum(self.ground_truth == 255) - np.sum(self.robot_belief == 255) <= 250:
            self.done = True

    def evaluate_exploration_rate(self):
        #self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)
        b = self.global_belief       # <- 改成用 global_belief
        self.explored_rate = np.sum(b == FREE) / np.sum(self.ground_truth == FREE)

    # ——— 私有工具：计算通信分组 ——— #
    def _compute_comm_groups(self):
        adj = np.zeros((N_AGENTS, N_AGENTS), dtype=bool)
        for i in range(N_AGENTS):
            for j in range(i+1, N_AGENTS):
                d = np.linalg.norm(self.robot_locations[i] - self.robot_locations[j])
                if d <= COMMS_RANGE:
                    adj[i,j] = adj[j,i] = True

        groups = []
        unseen = set(range(N_AGENTS))
        while unseen:
            root = unseen.pop()
            stack = [root]
            group = {root}
            while stack:
                u = stack.pop()
                for v in range(N_AGENTS):
                    if v in unseen and adj[u,v]:
                        unseen.remove(v)
                        group.add(v)
                        stack.append(v)
            groups.append(group)
        return groups


    # ——— 私有工具：组内 Belief 合并 ——— #
    def _merge_agent_beliefs(self, groups):
        for group in groups:
            # 开始时全 UNKNOWN
            merged = np.ones_like(self.global_belief) * UNKNOWN
            # 像素级 OR：任何 member 知道的就写入 merged
            for i in group:
                known = (self.agent_beliefs[i] != UNKNOWN)
                merged[known] = self.agent_beliefs[i][known]
            # 广播回组内每个 agent
            for i in group:
                self.agent_beliefs[i] = merged.copy()


    # ——— 修改 step：更新 Belief→合并→刷新 belief_info ——— #
    def step(self, next_waypoint, agent_id):
        self.evaluate_exploration_rate()
        
        # 1) update own location & 距离累加
        old_loc = self.robot_locations[agent_id]
        dist = np.linalg.norm(next_waypoint - old_loc)
        self.total_travel_dist += dist
        self.robot_locations[agent_id] = next_waypoint

        # 2) compute cell & sensor sweep
        cell = get_cell_position_from_coords(next_waypoint, self.belief_info)

        # —— 新增：记录该 agent 更新前的局部信念 —— #
        old_belief = self.agent_beliefs[agent_id].copy()

        # 执行感知：更新 global_belief & agent_beliefs[agent_id]
        self.global_belief = sensor_work(cell,
                                         round(self.sensor_range / self.cell_size),
                                         self.global_belief,
                                         self.ground_truth)
        self.agent_beliefs[agent_id] = sensor_work(cell,
                                                   round(self.sensor_range / self.cell_size),
                                                   self.agent_beliefs[agent_id],
                                                   self.ground_truth)

        # —— 新增：在 discovery_map 上打标签 —— #
        # 找到新看到的格子：之前是 UNKNOWN，现在变成已知
        newly_seen = (old_belief == UNKNOWN) & (self.agent_beliefs[agent_id] != UNKNOWN)
        # 只给第一次被观测的格子打标签
        mask = newly_seen & (self.discovery_map == -1)
        self.discovery_map[mask] = agent_id

        # 3) 通信分组 + 组内合并
        groups = self._compute_comm_groups()
        self._merge_agent_beliefs(groups)

        # 4) 刷新提供给 agent 的 belief_info
        self.belief_info = MapInfo(self.agent_beliefs[agent_id],
                                   self.belief_origin_x,
                                   self.belief_origin_y,
                                   self.cell_size)

        # 5) （原有 reward/terminal 逻辑按需放这里）
        reward = 0
        # … calculate_reward / check_done …

        return reward
    
    def get_agent_map(self, agent_id):
        """
        返回给 Agent 构图用的 MapInfo 对象，
        用的是该 agent 的本地 belief。
        """
        return MapInfo(
            self.agent_beliefs[agent_id],
            self.belief_origin_x,
            self.belief_origin_y,
            self.cell_size
        )

    def get_total_travel(self):
        """
        返回整个 episode 中所有 agent 的累计行驶距离。
        """
        return self.total_travel_dist
