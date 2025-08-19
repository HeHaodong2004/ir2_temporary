# intent_map_viz.py
import os
import numpy as np
import matplotlib.pyplot as plt

# ------------- 基础工具：把世界坐标落到 grid 索引 -----------------
def _to_cell(p_xy, map_info):
    """p_xy: np.array([x,y]) in world; return (cx, cy) int index on map_info.map"""
    cx, cy = None, None
    try:
        from utils import get_cell_position_from_coords
        cx, cy = get_cell_position_from_coords(np.array(p_xy, dtype=float), map_info)
    except Exception:
        pass
    return cx, cy

def _bresenham_line(p0, p1):
    """在格点上画线段，返回所有整数栅格坐标 (x,y) 列表。"""
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    points = []
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x += sx
        if e2 <= dx:
            err += dx; y += sy
    return points

def _make_disk_kernel(radius_cells):
    r = max(0, int(round(radius_cells)))
    if r == 0:
        return np.ones((1,1), dtype=np.float32)
    y, x = np.ogrid[-r:r+1, -r:r+1]
    mask = (x*x + y*y) <= r*r
    k = np.zeros((2*r+1, 2*r+1), dtype=np.float32)
    k[mask] = 1.0
    k /= k.sum()
    return k

# ------------- 主计算：把 intent_seq 转成概率图 -----------------
def build_intent_prob_map(agent, base_map_info, last_known_pos=None,
                          tube_radius_m=0.8, time_decay=0.85, close_loop=False):
    """
    返回与 base_map_info.map 同尺寸的 [0,1] 概率图。
    - agent.intent_seq: 世界坐标序列（长度=INTENT_HORIZON）
    - last_known_pos: 该 agent 的已知起点（世界坐标）。若为 None 则用 agent.location
    - tube_radius_m: 将离散意图“刷粗”的管径半径（米）
    - time_decay: 时间折减系数，越靠后的步权重越小
    - close_loop: True 时把最后一点与起点相连（一般 False）
    """
    H, W = base_map_info.map.shape
    prob = np.zeros((H, W), dtype=np.float32)

    # 起点与路径
    start_xy = np.array(last_known_pos if last_known_pos is not None else agent.location, dtype=float)
    path = [np.array(p, dtype=float) for p in (agent.intent_seq if agent.intent_seq else [])]
    if len(path) == 0:
        # 没有预测则只在当前位置打一个小核
        cx, cy = _to_cell(start_xy, base_map_info)
        if cx is not None and 0 <= cx < W and 0 <= cy < H:
            k = _make_disk_kernel(max(1, int(round(tube_radius_m / base_map_info.cell_size))))
            kh, kw = k.shape
            y0 = max(0, cy - kh//2); y1 = min(H, cy + kh//2 + 1)
            x0 = max(0, cx - kw//2); x1 = min(W, cx + kw//2 + 1)
            ky0 = kh//2 - (cy - y0); ky1 = ky0 + (y1 - y0)
            kx0 = kw//2 - (cx - x0); kx1 = kx0 + (x1 - x0)
            prob[y0:y1, x0:x1] += k[ky0:ky1, kx0:kx1]
        if prob.max() > 0: prob /= prob.max()
        return prob

    # 将 path 的世界坐标映射为 grid 索引序列
    cells = []
    # 拼接起点
    c0 = _to_cell(start_xy, base_map_info)
    if c0[0] is not None:
        cells.append(c0)
    for p in path:
        c = _to_cell(p, base_map_info)
        if c[0] is not None:
            cells.append(c)

    if len(cells) < 1:
        return prob

    # 构造时间权重
    #   第 i 段（cells[i] -> cells[i+1]）赋予 weight = time_decay**i
    #   同时用盘核把线段刷粗
    kernel = _make_disk_kernel(max(1, int(round(tube_radius_m / base_map_info.cell_size))))
    kh, kw = kernel.shape

    seg_cnt = len(cells) - 1 + (1 if (close_loop and len(cells) > 1) else 0)
    for i in range(seg_cnt):
        a = cells[i]
        b = cells[(i+1) % len(cells)] if (i+1 < len(cells)) else cells[-1]
        w = (time_decay ** i)

        line = _bresenham_line(a, b)
        for (x, y) in line:
            if not (0 <= x < W and 0 <= y < H): 
                continue
            # 将盘核加到 (y,x) 周围
            y0 = max(0, y - kh//2); y1 = min(H, y + kh//2 + 1)
            x0 = max(0, x - kw//2); x1 = min(W, x + kw//2 + 1)
            ky0 = kh//2 - (y - y0); ky1 = ky0 + (y1 - y0)
            kx0 = kw//2 - (x - x0); kx1 = kx0 + (x1 - x0)
            prob[y0:y1, x0:x1] += w * kernel[ky0:ky1, kx0:kx1]

    # 归一化到 [0,1]
    mx = prob.max()
    if mx > 1e-8:
        prob /= mx
    return prob

# ------------- 可视化：叠加在预测图 / 局部窗口上 -----------------
def draw_intent_probability_maps(worker, step=0, use_pred_mean=True, local_window=True,
                                 tube_radius_m=0.8, time_decay=0.85, save_name=None):
    """
    为 worker.robots 的每个 agent 绘制 intent 概率图。
    - use_pred_mean: True 用 pred_mean_map_info，False 用 pred_max_map_info
    - local_window:  True 画“局部更新窗口”；False 画全图
    - tube_radius_m, time_decay: 与计算函数一致
    - 输出到 worker.run_dir / intent_maps_tXXXX.png
    """
    robots = worker.robots
    env = worker.env

    n = len(robots)
    cols = n
    rows = 1
    plt.switch_backend('agg')
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n == 1:
        axes = [axes]

    for i, r in enumerate(robots):
        ax = axes[i]
        ax.axis('off')

        base_map = r.pred_mean_map_info if use_pred_mean else r.pred_max_map_info
        if base_map is None:
            # 没预测就用 belief 兜底
            base_map = r.map_info

        # last_known_pos：与 worker 的“该 viewer 视角下的”队友位置保持一致；
        # 这里只对自己画概率，所以用自身当前位置即可
        last_known_pos = r.location

        # 计算全图概率
        prob_full = build_intent_prob_map(
            agent=r, base_map_info=base_map,
            last_known_pos=last_known_pos,
            tube_radius_m=tube_radius_m,
            time_decay=time_decay
        )

        if local_window:
            # 裁剪到局部窗口（与训练/可视化一致）
            local = r.get_updating_map(r.location, base=base_map)
            # 将全图概率裁到相同窗口
            from utils import get_cell_position_from_coords
            oidx = get_cell_position_from_coords(np.array([local.map_origin_x, local.map_origin_y]), base_map)
            tidx = get_cell_position_from_coords(
                np.array([local.map_origin_x + (local.map.shape[1]-1)*local.cell_size,
                          local.map_origin_y + (local.map.shape[0]-1)*local.cell_size]), base_map
            )
            y0, y1 = int(oidx[1]), int(tidx[1])
            x0, x1 = int(oidx[0]), int(tidx[0])
            y0 = max(0, y0); x0 = max(0, x0)
            y1 = min(base_map.map.shape[0]-1, y1)
            x1 = min(base_map.map.shape[1]-1, x1)
            prob_show = prob_full[y0:y1+1, x0:x1+1]
            # 背景：预测图
            ax.imshow(local.map, cmap='gray', vmin=0, vmax=255)
            # 概率热力叠加
            im = ax.imshow(prob_show, cmap='hot', alpha=0.65, vmin=0.0, vmax=1.0)
            # 当前位置
            rc = _to_cell(r.location, local); 
            if rc[0] is not None:
                ax.plot(rc[0], rc[1], 'mo', markersize=10, zorder=5)
            ax.set_title(f'Agent {r.id} Intent Prob (local)')
        else:
            # 全局展示
            ax.imshow(base_map.map, cmap='gray', vmin=0, vmax=255)
            im = ax.imshow(prob_full, cmap='hot', alpha=0.65, vmin=0.0, vmax=1.0)
            rc = _to_cell(r.location, base_map)
            if rc[0] is not None:
                ax.plot(rc[0], rc[1], 'mo', markersize=10, zorder=5)
            ax.set_title(f'Agent {r.id} Intent Prob (global)')

        # 画出已知“intent 折线”（便于对比）
        if r.intent_seq:
            try:
                # 将 intent 映射到当前显示坐标系
                ref_map = r.get_updating_map(r.location, base=(r.pred_mean_map_info if local_window else base_map)) if local_window else base_map
                pts = []
                # 起点
                c0 = _to_cell(r.location, ref_map)
                if c0[0] is not None: pts.append(c0)
                for p in r.intent_seq:
                    c = _to_cell(p, ref_map)
                    if c[0] is not None: pts.append(c)
                if len(pts) >= 2:
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    ax.plot(xs, ys, 'c--', linewidth=1.5, marker='x', markersize=5, zorder=6)
            except Exception:
                pass

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('Intent probability (normalized)')

    out_name = save_name or f"intent_maps_t{int(step):04d}.png"
    out_path = os.path.join(worker.run_dir, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    # 同时把帧路径也记录（可选，便于合成 gif）
    if hasattr(worker.env, 'frame_files'):
        worker.env.frame_files.append(out_path)
    print(f"[intent-viz] saved: {out_path}")
