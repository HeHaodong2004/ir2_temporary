import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from env import Env
from agent import Agent
from utils import *
from node_manager import NodeManager
from parameter import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class MultiAgentWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        # 环境和节点管理器
        self.env = Env(global_step, plot=self.save_image)
        self.n_agent = N_AGENTS
        self.node_manager = NodeManager(plot=self.save_image)

        # 创建 agents
        self.robot_list = [
            Agent(i, policy_net, self.node_manager, self.device, self.save_image)
            for i in range(N_AGENTS)
        ]

        # 初始化“最后已知位置”缓存
        self.last_known_locations = [
            self.env.robot_locations.copy()
            for _ in range(N_AGENTS)
        ]

        # 初始化“最后已知 intent”缓存
        initial_intents = {r.id: deepcopy(r.intent_seq) for r in self.robot_list}
        self.last_known_intents = [
            deepcopy(initial_intents)
            for _ in range(N_AGENTS)
        ]

        # 关键时刻判断阈值
        self.UTILITY_THRESHOLD = UTILITY_RANGE * MIN_UTILITY

        # 缓存与性能指标
        self.episode_buffer = []
        self.perf_metrics = {}
        for _ in range(15):
            self.episode_buffer.append([])

    def is_critical(self, agent_id):
        """
        判断 agent 是否处于“关键时刻”：
        1) 周围节点最大 utility 超过阈值（“路口”）
        2) 与任意队友的距离超过 0.9 * COMMS_RANGE（“快要失联”）
        """
        robot = self.robot_list[agent_id]
        if robot.utility is not None and np.max(robot.utility) > self.UTILITY_THRESHOLD:
            return True

        dists = np.linalg.norm(
            self.env.robot_locations - self.env.robot_locations[agent_id],
            axis=1
        )
        if np.max(dists) > 0.9 * COMMS_RANGE:
            return True

        return False

    def run_episode(self):
        done = False

        # —— 初始构图：根据各自 last_known_locations 构图 —— #
        for i, robot in enumerate(self.robot_list):
            map_info_i = self.env.get_agent_map(i)
            robot.update_graph(map_info_i, self.env.robot_locations[i])
            robot.update_planning_state(self.last_known_locations[i])

        for step in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            next_node_index_list = []

            # 1) 每个 agent 用缓存的位置和 intents 构建观测并选 action
            for i, robot in enumerate(self.robot_list):
                obs = robot.get_observation(self.last_known_intents[i])
                robot.save_observation(obs)

                next_loc, next_idx, act_idx = robot.select_next_waypoint(obs)
                robot.save_action(act_idx)

                node = robot.node_manager.nodes_dict.find((robot.location[0], robot.location[1]))
                neigh = np.array(node.data.neighbor_list)
                assert (next_loc[0] + 1j*next_loc[1]) in (neigh[:,0] + 1j*neigh[:,1])
                assert not np.all(next_loc == robot.location)

                selected_locations.append(next_loc)
                dist_list.append(np.linalg.norm(next_loc - robot.location))
                next_node_index_list.append(next_idx)

            selected_locations = np.vstack(selected_locations)
            arriving_seq = np.argsort(dist_list)
            ordered_locs = selected_locations[arriving_seq]

            # 2) 碰撞解决
            for j, loc in enumerate(ordered_locs):
                solved = ordered_locs[:j]
                while (loc[0] + 1j*loc[1]) in (solved[:,0] + 1j*solved[:,1]):
                    aid = arriving_seq[j]
                    near = self.robot_list[aid].node_manager.nodes_dict.nearest_neighbors(loc.tolist(), 25)
                    for nd in near:
                        c = nd.data.coords
                        if (c[0] + 1j*c[1]) in (solved[:,0] + 1j*solved[:,1]):
                            continue
                        loc = c
                        break
                    ordered_locs[j] = loc
                    selected_locations[aid] = loc

            # 3) 环境 step：Belief 更新 + 组内合并
            for robot, loc in zip(self.robot_list, selected_locations):
                self.env.step(loc, robot.id)

            # 4) 有条件地更新 last_known_intents：
            #    — 组内且关键时刻：同步真实 intent_seq
            #    — 组内但非关键：置空列表
            #    — 组外：不触及，保留上次缓存
            groups = self.env._compute_comm_groups()
            for group in groups:
                for i in group:
                    crit_i = self.is_critical(i)
                    for j in group:
                        crit_j = self.is_critical(j)
                        if crit_i or crit_j:
                            self.last_known_intents[i][j] = deepcopy(self.robot_list[j].intent_seq)
                        else:
                            self.last_known_intents[i][j] = []

            # 5) 重建图 & 计算奖励
            reward_list = []
            for idx, (robot, nxt_idx) in enumerate(zip(self.robot_list, next_node_index_list)):
                map_info_i = self.env.get_agent_map(idx)
                robot.update_graph(map_info_i, self.env.robot_locations[idx])
                robot.update_planning_state(self.last_known_locations[idx])
                r_ind = robot.utility[nxt_idx] / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
                reward_list.append(r_ind)

            total_util = sum(r.utility.sum() for r in self.robot_list)
            if total_util == 0:
                done = True

            team_r = self.env.calculate_reward() - 0.5
            if done:
                team_r += 10

            # 6) 保存 reward & done
            for i, (robot, r) in enumerate(zip(self.robot_list, reward_list)):
                robot.save_reward(r + team_r)
                robot.update_planning_state(self.last_known_locations[i])
                robot.save_done(done)

            if self.save_image:
                self.plot_env(step)

            if done:
                break

        # 7) 收集 metrics、buffer、gif
        self.perf_metrics['travel_dist'] = self.env.get_total_travel()
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        for i, robot in enumerate(self.robot_list):
            obs = robot.get_observation(self.last_known_intents[i])
            robot.save_next_observations(obs)
            for j in range(len(self.episode_buffer)):
                self.episode_buffer[j] += robot.episode_buffer[j]

        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def plot_env(self, step):
        plt.switch_backend('agg')
        color_list = ['r', 'b', 'g', 'y']
        fig, axes = plt.subplots(1, 2 + self.n_agent, figsize=(5 * (2 + self.n_agent), 5))

        # Panel 1: global belief + traj + intent
        ax0 = axes[0]
        ax0.imshow(self.env.global_belief, cmap='gray')
        ax0.axis('off')
        for robot in self.robot_list:
            c = color_list[robot.id]
            cell = get_cell_position_from_coords(robot.location, robot.map_info)
            ax0.plot(cell[0], cell[1], c+'o', markersize=12, zorder=5)
            ax0.plot(
                (np.array(robot.trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                (np.array(robot.trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size,
                c, linewidth=2, zorder=1
            )
        for robot in self.robot_list:
            if not robot.intent_seq:
                continue
            cells = [get_cell_position_from_coords(np.array(p), robot.map_info) for p in robot.intent_seq]
            xs = [(x - robot.map_info.map_origin_x) / robot.cell_size for x, y in cells]
            ys = [(y - robot.map_info.map_origin_y) / robot.cell_size for x, y in cells]
            ax0.plot(xs, ys, '--x', label=f'Intent {robot.id}', zorder=2)
        ax0.legend(loc='upper right', fontsize='small')
        ax0.set_title('Global Belief + Traj + Intent')

        # Panel 2: agent0 graph
        ax1 = axes[1]
        agent0 = self.robot_list[0]
        ax1.imshow(agent0.map_info.map, cmap='gray')
        ax1.axis('off')
        for coords in agent0.node_coords:
            node = self.node_manager.nodes_dict.find(coords.tolist()).data
            for neigh in node.neighbor_list[1:]:
                mid = (np.array(neigh) + coords) / 2
                ax1.plot(
                    ([coords[0], mid[0]] - agent0.map_info.map_origin_x) / agent0.map_info.cell_size,
                    ([coords[1], mid[1]] - agent0.map_info.map_origin_y) / agent0.map_info.cell_size,
                    'tan', zorder=1
                )
        ax1.set_title('Agent 0 Graph')

        # Panels 3...: each agent's local obs + traj + intent
        for i, robot in enumerate(self.robot_list):
            ax = axes[2 + i]
            obs_map = robot.updating_map_info.map
            ax.imshow(obs_map, cmap='gray')
            # traj
            traj_cells = [
                get_cell_position_from_coords(np.array([x,y]), robot.updating_map_info)
                for x, y in zip(robot.trajectory_x, robot.trajectory_y)
            ]
            if traj_cells:
                tx, ty = zip(*traj_cells)
                ax.plot(tx, ty, color_list[robot.id], linewidth=2, zorder=1)
            # current pos
            cur = get_cell_position_from_coords(robot.location, robot.updating_map_info)
            ax.plot(cur[0], cur[1], color_list[robot.id]+'o', markersize=8, zorder=2)
            # intent
            icoords = [
                get_cell_position_from_coords(np.array(p), robot.updating_map_info)
                for p in robot.intent_seq
            ]
            if icoords:
                ix, iy = zip(*icoords)
                ax.scatter(ix, iy, marker='x', s=50, zorder=3)
            ax.set_title(f'Agent {robot.id} Obs+Traj+Intent')
            ax.axis('off')

        fig.suptitle(f'Step {step} | Explored: {self.env.explored_rate:.3g} | Dist: {self.env.get_total_travel():.2f}')
        plt.tight_layout()
        out = f'{gifs_path}/{self.global_step}_{step}_obs.png'
        plt.savefig(out, dpi=150)
        plt.close(fig)
        self.env.frame_files.append(out)
