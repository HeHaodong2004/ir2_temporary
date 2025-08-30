# rendezvous_picker.py
import numpy as np
import math
from collections import deque
from parameter import *
from utils import get_cell_position_from_coords

# 函数签名改变：不再接收worker，而是直接接收机器人列表
def pick_rendezvous_point(robots_for_mission, current_step):
    """
    V3: 接口优化的Rendezvous Point Picker.
    - 直接接收一个机器人列表，而不是整个worker对象。
    """
    # --- 0. 准备工作 ---
    if not robots_for_mission:
        return None, 0.0, 0, {'reason': 'no_robots_for_mission'}
    a0 = robots_for_mission[0] # 以列表中的第一个机器人为参考
    
    if a0.pred_mean_map_info is None:
        return None, 0.0, 0, {'reason': 'no_prediction_map'}

    # ... 此后的所有代码与您之前的版本完全相同 ...
    pred_map = a0.pred_mean_map_info.map.astype(np.float32)
    belief_map = a0.map_info.map
    H, W = pred_map.shape
    cell_size = float(a0.cell_size)

    prob_free = pred_map / float(FREE)
    tau_free = 0.6
    traversable_mask = ((prob_free >= tau_free) | (belief_map == FREE)) & (belief_map != OCCUPIED)
    unknown_mask = (belief_map == UNKNOWN)
    
    cand_rows, cand_cols = np.where(unknown_mask & traversable_mask)
    if cand_rows.size == 0:
        return None, 0.0, 0, {'reason': 'no_candidate_points_in_unknown_area'}
    
    num_candidates = min(cand_rows.size, 200)
    indices = np.random.choice(cand_rows.size, num_candidates, replace=False)
    candidate_cells = np.stack([cand_rows[indices], cand_cols[indices]], axis=1)

    dist_maps_meters = []
    # 迭代传入的机器人列表
    for r in robots_for_mission:
        start_cell = _world_to_cell_rc(r.location, a0.map_info)
        if not traversable_mask[start_cell[0], start_cell[1]]:
            start_cell = _find_nearest_valid_cell(traversable_mask, np.array(start_cell))
        dist_map_steps = _bfs_dist_map(traversable_mask, tuple(start_cell))
        dist_maps_meters.append(dist_map_steps * cell_size)

    best_candidate = None
    max_score = -np.inf
    W_INFO_GAIN = 0.7
    W_TRAVEL_COST = 0.3
    info_radius_pixels = int(RENDEZVOUS_INFO_RADIUS_M / cell_size)

    for r_cand, c_cand in candidate_cells:
        r_min, r_max = max(0, r_cand - info_radius_pixels), min(H, r_cand + info_radius_pixels)
        c_min, c_max = max(0, c_cand - info_radius_pixels), min(W, c_cand + info_radius_pixels)
        info_gain_score = unknown_mask[r_min:r_max, c_min:c_max].sum()
        if info_gain_score < 5: continue

        travel_distances = [dist_map[r_cand, c_cand] for dist_map in dist_maps_meters]
        if any(not np.isfinite(d) for d in travel_distances): continue
        max_travel_dist = np.max(travel_distances)
        
        map_diagonal = math.hypot(H * cell_size, W * cell_size)
        norm_info_gain = info_gain_score / (math.pi * info_radius_pixels**2 + 1e-6)
        norm_travel_cost = max_travel_dist / (map_diagonal + 1e-6)
        score = W_INFO_GAIN * norm_info_gain - W_TRAVEL_COST * norm_travel_cost
        
        if score > max_score:
            max_score = score
            best_candidate = {'cell': (r_cand, c_cand), 'max_dist': max_travel_dist, 'score': score}

    if best_candidate is None:
        return None, 0.0, 0, {'reason': 'no_candidate_with_info_gain'}

    best_cell = best_candidate['cell']
    center_xy = _cell_to_world(best_cell, a0.map_info)
    r_meet = float(MEET_RADIUS_FRAC * COMMS_RANGE)
    steps_to_reach = best_candidate['max_dist'] / NODE_RESOLUTION
    buffer_steps = MEET_BUFFER_ALPHA * steps_to_reach + MEET_BUFFER_BETA
    # 直接使用传入的 current_step
    T_meet = int(current_step + steps_to_reach + buffer_steps)
    meta = {'score': best_candidate['score']}
    
    return center_xy, r_meet, T_meet, meta

# --- Helper Functions (No changes needed here) ---
def _world_to_cell_rc(world_xy, map_info):
    cell = get_cell_position_from_coords(np.array(world_xy, dtype=float), map_info)
    return int(cell[1]), int(cell[0])

def _cell_to_world(rc, map_info):
    r, c = rc
    x = map_info.map_origin_x + c * map_info.cell_size
    y = map_info.map_origin_y + r * map_info.cell_size
    return np.array([x, y])

def _find_nearest_valid_cell(mask, start_rc):
    q = deque([tuple(start_rc)])
    visited = {tuple(start_rc)}
    while q:
        r, c = q.popleft()
        if mask[r, c]: return np.array([r, c])
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and (nr, nc) not in visited:
                q.append((nr, nc))
                visited.add((nr, nc))
    return start_rc

def _bfs_dist_map(traversable_mask, start_rc):
    H, W = traversable_mask.shape
    dist_map = np.full((H, W), np.inf, dtype=np.float32)
    q = deque([(start_rc, 0)])
    if 0 <= start_rc[0] < H and 0 <= start_rc[1] < W:
        dist_map[start_rc[0], start_rc[1]] = 0
    else: return dist_map
    while q:
        (r, c), dist = q.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and traversable_mask[nr, nc] and np.isinf(dist_map[nr, nc]):
                dist_map[nr, nc] = dist + 1
                q.append(((nr, nc), dist + 1))
    return dist_map
