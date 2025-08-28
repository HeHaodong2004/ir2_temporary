# rendezvous_picker.py
# -*- coding: utf-8 -*-
import math
import heapq
from collections import deque
import numpy as np

from parameter import *
from utils import get_cell_position_from_coords, MapInfo, get_frontier_in_map


# =========================================================
# 公共入口：挑一个“面向未来”的 rendezvous 点（世界坐标）
# =========================================================
def pick_rendezvous_point(worker,
                          H_max_meter: float = 100.0,     # 允许更远一些（未来会合）
                          r_meet_frac: float = 0.4,
                          H_post_meter: float = 20.0,
                          sync_tol_meter: float = 18.0,   # 到达时间差限制（米）
                          target_frac: float = 0.60,      # 远距偏好：目标均值距离 = target_frac * map_diag
                          cand_opts: dict = None,
                          debug: bool = False):
    """
    仅用于“计算可视化的 RDV 点”，不干扰你们原有策略。
    - 候选：预测 Free & 当前 UNKNOWN（= 外面），可选是否只贴前沿（默认关闭）
    - 距离：在“预测 Free 栅格”上对每个 agent 跑一次 BFS，得到米级整图距离场
    - 目标：方向对齐 + 远距偏好 + 同步友好 + 不确定度增益 + 局部通畅
    """
    # ---- 0) 取一份 viewer 作为“预测世界”的坐标系（用 0 号 agent） ----
    a0 = worker.robots[0]
    if a0.pred_mean_map_info is None:
        return None, 0.0, {'reason': 'no_pred'}

    pred = a0.pred_mean_map_info.map.astype(np.float32)  # [1,255]
    belief = a0.map_info.map
    H, W = pred.shape
    cell = float(a0.cell_size)

    # ---- 1) 构造“预测可通行”栅格（traversable）----
    # 预测 free: p_free >= tau_free；同时把“已知 Free”也视为可走（避免门口被卡住）
    cfg = dict(tau_free=0.60, near_frontier_only=False, max_band_m=12.0)
    if isinstance(cand_opts, dict): cfg.update(cand_opts)

    p_free = pred / float(FREE)
    traversable = (p_free >= cfg['tau_free']) | (belief == FREE)
    # 明确把已知占据剔除
    traversable &= (belief != OCCUPIED)

    # ---- 2) 候选：只在 “预测 Free 且 当前 UNKNOWN” 的位置，均匀抽样 ----
    unknown = (belief == UNKNOWN)
    cand_mask = (p_free >= float(cfg['tau_free'])) & unknown

    if cfg['near_frontier_only']:
        band_cells = max(1, int(round(cfg['max_band_m'] / cell)))
        cand_mask &= _unknown_edge_band(unknown, band_cells)

    rr, cc = np.where(cand_mask)
    if rr.size == 0:
        return None, 0.0, {'reason': 'no_candidates_after_pred_unknown', 'cfg': cfg}

    # 均匀抽样，控制上限
    topk = 160
    N = int(rr.size)
    if N > topk:
        step = max(1, int(math.sqrt(N / float(topk))))
        sel = (rr % step == 0) & (cc % step == 0)
        rr2, cc2 = rr[sel], cc[sel]
        if rr2.size == 0:
            idx = np.random.choice(N, size=topk, replace=False)
            rr2, cc2 = rr[idx], cc[idx]
    else:
        rr2, cc2 = rr, cc

    cand_cells = np.stack([rr2, cc2], axis=1)  # row, col
    cand_world = _cells_to_world(cand_cells, a0.map_info)

    # ---- 3) 为每个 agent 预计算：在“预测可通行栅格”上的 BFS 距离场（米） ----
    dist_maps = []
    for r in worker.robots:
        sr, sc = _world_to_cell_rc(r.location, a0.map_info)
        if not _in_bounds(sr, sc, H, W) or not traversable[sr, sc]:
            # 起点不在可行网格上：就近修正（找最近的 traversable）
            sr, sc = _nearest_free_rc(traversable, sr, sc)
            if sr is None:
                # 彻底不可行
                return None, 0.0, {'reason': 'agent_not_on_traversable'}
        D = _bfs_dist_map(traversable, (sr, sc))
        dist_maps.append(D * cell)  # 转米

    # 地图对角线（米），设一个“理想会合半径”
    map_diag_m = math.hypot(H * cell, W * cell)
    D_target = target_frac * map_diag_m
    sigma = max(1e-6, 0.25 * D_target)  # 远距偏好带宽

    # ---- 4) 不确定度图（只在 UNKNOWN 内：U = p*(1-p)） ----
    U = (p_free * (1.0 - p_free) * unknown.astype(np.float32))

    # ---- 5) 评估候选 ----
    items = []
    rej_sync, rej_hmax, rej_unreach = 0, 0, 0

    # 预取每个 agent 的“意图方向”向量
    headings = []
    for r in worker.robots:
        h = _heading_from_intent(r)
        headings.append(h)  # np.array([dx,dy]) or None

    for (rr, cc), p in zip(cand_cells, cand_world):
        # 多 agent 预测网格距离（米）
        t_list = []
        unreached = False
        for D in dist_maps:
            d = D[rr, cc]
            if not np.isfinite(d):
                unreached = True
                break
            t_list.append(d)
        if unreached:
            rej_unreach += 1
            continue
        t_list = np.array(t_list, dtype=float)

        # 1) 同步友好：时间差 + Jain
        t_spread = float(t_list.max() - t_list.min())
        if t_spread > sync_tol_meter:
            rej_sync += 1
            continue

        denom = (len(t_list) * np.sum(t_list ** 2) + 1e-9)
        jain = float((np.sum(t_list) ** 2) / denom)
        t_mean = float(np.mean(t_list))
        if t_list.max() > H_max_meter:
            rej_hmax += 1
            continue

        # 2) 远距偏好（均值距离靠近 D_target 最优）
        S_far = math.exp(-((t_mean - D_target) ** 2) / (2.0 * sigma * sigma))

        # 3) 方向对齐：candidate 相对方向 与 intent 方向越一致越好
        S_align = _alignment_score(p, worker, headings)

        # 4) 会合后信息增益：候选周围 H_post_meter 半径内累积 U
        S_gain = _gain_local_U(U, a0.map_info, p, radius_m=H_post_meter)

        # 5) 局部通畅度：候选邻域内 traversable 的密度
        S_clear = _local_clear_density(traversable, (rr, cc), radius_cells=max(2, int(round(4.0 / cell))))

        # 6) 平均时间也纳入（越短越好，但不是目标）
        S_time = math.exp(-math.log(5.0) * (t_mean / max(H_max_meter, 1e-6)))

        # 7) 汇总（未来导向权重）
        total = (0.35 * S_align +
                 0.25 * S_far   +
                 0.20 * (0.6 * jain + 0.4 * S_time) +
                 0.15 * _norm01_scalar(S_gain) +
                 0.05 * S_clear)

        items.append(dict(
            p=np.array(p, dtype=float),
            rc=(int(rr), int(cc)),
            t_list=t_list, t_mean=t_mean, t_spread=t_spread, jain=jain,
            S_align=S_align, S_far=S_far, S_time=S_time,
            S_gain=S_gain, S_clear=S_clear,
            total=total
        ))

    if not items:
        return None, 0.0, {
            'reason': 'no_feasible_candidates',
            'debug': {'reject_unreach': rej_unreach, 'reject_sync': rej_sync, 'reject_hmax': rej_hmax},
            'cfg': dict(cfg, target_frac=target_frac, H_max=H_max_meter, sync_tol=sync_tol_meter)
        }

    # 选最佳
    best = max(items, key=lambda d: d['total'])
    center_xy = best['p'].copy()
    r_meet = float(r_meet_frac * COMMS_RANGE)
    meta = dict(
        total=float(best['total']),
        lat_est=float(best['t_list'].max()),
        t_mean=float(best['t_mean']),
        t_spread=float(best['t_spread']),
        jain=float(best['jain']),
        S_align=float(best['S_align']),
        S_far=float(best['S_far']),
        S_time=float(best['S_time']),
        S_gain=float(_norm01_scalar(best['S_gain'])),
        S_clear=float(best['S_clear']),
        cfg=dict(cfg, target_frac=target_frac, H_max=H_max_meter, sync_tol=sync_tol_meter)
    )
    # —— 估算到达步数（把米转成步：用 NODE_RESOLUTION 作为‘每步行进米数’的标定）——
    step_meters = float(NODE_RESOLUTION)  # 每步≈走一个图节点间距
    ETA_list = [float(d) / max(1e-6, step_meters) for d in best['t_list']]
    ETA_max  = max(ETA_list)
    ETA_mean = float(np.mean(ETA_list))

    # 安全余量：alpha*均值 + beta（步）
    T_buffer = float(MEET_BUFFER_ALPHA) * ETA_mean + float(MEET_BUFFER_BETA)
    T_meet   = int(math.ceil(getattr(worker, 'global_step', 0) + ETA_max + T_buffer))

    # 原 return 替换为（多一个 T_meet）：
    return center_xy, r_meet, T_meet, meta


# =========================================================
# 工具函数（候选、距离、打分）
# =========================================================

def _in_bounds(r, c, H, W):
    return (0 <= r < H) and (0 <= c < W)


def _world_to_cell_rc(world_xy, map_info: MapInfo):
    """世界坐标 -> (row, col) 索引（与 numpy 索引一致：先 y 后 x）"""
    cell = get_cell_position_from_coords(np.array(world_xy, dtype=float), map_info)
    return int(cell[1]), int(cell[0])


def _cells_to_world(rcs, map_info: MapInfo):
    """(row,col) 数组 -> 世界坐标"""
    r = rcs[:, 0].astype(np.float32)
    c = rcs[:, 1].astype(np.float32)
    x = map_info.map_origin_x + c * map_info.cell_size
    y = map_info.map_origin_y + r * map_info.cell_size
    return np.stack([x, y], axis=1)


def _nearest_free_rc(traversable, r, c):
    """若 (r,c) 不可走，在小范围内找最近可走点；找不到返回 (None,None)"""
    H, W = traversable.shape
    if _in_bounds(r, c, H, W) and traversable[r, c]:
        return r, c
    # 扩环搜索
    for rad in range(1, 16):
        r0, r1 = max(0, r - rad), min(H - 1, r + rad)
        c0, c1 = max(0, c - rad), min(W - 1, c + rad)
        found = np.argwhere(traversable[r0:r1 + 1, c0:c1 + 1])
        if found.size > 0:
            rr, cc = found[0]
            return r0 + int(rr), c0 + int(cc)
    return None, None


def _bfs_dist_map(trav_mask, start_rc):
    """在 trav_mask 上做 4邻域 BFS，输出到每个栅格的步数（不可达为 +inf）。"""
    H, W = trav_mask.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    sr, sc = start_rc
    if not _in_bounds(sr, sc, H, W) or not trav_mask[sr, sc]:
        return dist
    q = deque()
    dist[sr, sc] = 0.0
    q.append((sr, sc))
    while q:
        r, c = q.popleft()
        d = dist[r, c] + 1.0
        # 4-neighbors
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if _in_bounds(nr, nc, H, W) and trav_mask[nr, nc] and d < dist[nr, nc]:
                dist[nr, nc] = d
                q.append((nr, nc))
    return dist


def _unknown_edge_band(unknown_mask, band_cells):
    """UNKNOWN 的“边界扩张”带。"""
    unk = unknown_mask.astype(bool)
    inner = np.roll(unk, 1, 0) & np.roll(unk, -1, 0) & np.roll(unk, 1, 1) & np.roll(unk, -1, 1)
    edge = (unk & (~inner))
    band = edge.copy()
    for _ in range(max(0, int(band_cells) - 1)):
        band = band | np.roll(band, 1, 0) | np.roll(band, -1, 0) | np.roll(band, 1, 1) | np.roll(band, -1, 1)
        band = band & unk
    return band


def _heading_from_intent(agent):
    """从 agent.intent_seq 估计方向向量（世界坐标）。返回单位向量或 None。"""
    try:
        path = agent.intent_seq if isinstance(agent.intent_seq, list) else []
        if len(path) >= 1:
            p_last = np.array(path[-1], dtype=float)
            v = p_last - np.array(agent.location, dtype=float)
        else:
            return None
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else None
    except Exception:
        return None


def _alignment_score(p_world, worker, headings):
    """candidate 与各 agent 意图方向的对齐得分（0~1）。"""
    scores = []
    P = np.array(p_world, dtype=float)
    for r, h in zip(worker.robots, headings):
        v = P - np.array(r.location, dtype=float)
        nv = float(np.linalg.norm(v))
        if nv < 1e-6 or h is None:
            continue
        v /= nv
        s = float(np.dot(v, h))  # [-1,1]
        scores.append(max(0.0, s))  # 只奖励方向一致
    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))


def _gain_local_U(U, map_info: MapInfo, p_world, radius_m=20.0):
    """candidate 周围半径内的不确定度累积（越大越好）。"""
    try:
        r0, c0 = _world_to_cell_rc(p_world, map_info)
        R = int(max(1, round(radius_m / float(map_info.cell_size))))
        H, W = U.shape
        r1, r2 = max(0, r0 - R), min(H, r0 + R + 1)
        c1, c2 = max(0, c0 - R), min(W, c0 + R + 1)
        patch = U[r1:r2, c1:c2]
        # 圆形掩码
        yy, xx = np.ogrid[-(r0 - r1):(r2 - r0), -(c0 - c1):(c2 - c0)]
        mask = (yy * yy + xx * xx) <= (R * R)
        return float(patch[mask].sum())
    except Exception:
        return 0.0


def _local_clear_density(traversable, rc, radius_cells=6):
    """候选附近 Free 密度（0~1）。"""
    r0, c0 = rc
    H, W = traversable.shape
    r1, r2 = max(0, r0 - radius_cells), min(H, r0 + radius_cells + 1)
    c1, c2 = max(0, c0 - radius_cells), min(W, c0 + radius_cells + 1)
    patch = traversable[r1:r2, c1:c2].astype(np.float32)
    # 圆形区域
    yy, xx = np.ogrid[-(r0 - r1):(r2 - r0), -(c0 - c1):(c2 - c0)]
    mask = (yy * yy + xx * xx) <= (radius_cells * radius_cells)
    if mask.sum() <= 1:
        return 0.0
    return float(patch[mask].mean())


def _norm01_scalar(x):
    """标量转 0~1（鲁棒归一化）"""
    if not np.isfinite(x):
        return 0.0
    # 经验上 S_gain 会随半径/场景变化很大，这里做一个对数压缩再映射
    x = max(1e-9, float(x))
    y = math.log1p(x)  # 缓和大值
    # 粗略映射到 0~1（经验常数）
    return float(1.0 - math.exp(-y))


# =========================
# 说明：
# - 想更“远一些”：把 target_frac 调大（如 0.7~0.8），或把 H_max_meter 提大（如 150）
# - 想更“贴合趋势”：S_align 已经对齐 intent；也可以把上面的总分权重里 0.35 再加大
# - 如果想“只在前沿带附近的未知区”选，把 cand_opts 里 near_frontier_only=True
#
# 示例调用（在 worker.run_episode 第 6.5 步）：
# self.debug_rdv = pick_rendezvous_point(
#     self,
#     H_max_meter=140.0,
#     r_meet_frac=0.4,
#     H_post_meter=24.0,
#     sync_tol_meter=16.0,
#     target_frac=0.7,
#     cand_opts=dict(tau_free=0.58, near_frontier_only=False, max_band_m=9999.0)
# )
# =========================================================
