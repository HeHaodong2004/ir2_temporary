# ===== FAST Advanced MC (chained + recenter + downsample + vector line + gaussian) =====
import os, numpy as np, torch, matplotlib.pyplot as plt
from skimage.draw import line as sk_line
from skimage.transform import resize
from skimage.filters import gaussian
from utils import get_cell_position_from_coords
from parameter import UPDATING_MAP_SIZE, N_AGENTS

def _to_cell_xy_fast(p_xy, map_info):
    try:
        x, y = get_cell_position_from_coords(np.array(p_xy, dtype=float), map_info)
        return int(x), int(y)
    except Exception:
        return None, None

def _recenter_minimal(node_inputs, n_node, coords_np, center_idx):
    # feats: rel_xy(2) + utility(1) + pred_prob(1) + guidepost(1) + guidepost2(1) + occupancy(1) + intent_mask(N_AGENTS)
    rel = node_inputs[..., :2]
    occ = node_inputs[..., 6:7]
    center = coords_np[center_idx]
    delta = coords_np - center
    scale = (UPDATING_MAP_SIZE / 2.0)
    rel[0, :n_node, 0] = torch.tensor(delta[:n_node, 0] / scale, dtype=rel.dtype, device=rel.device)
    rel[0, :n_node, 1] = torch.tensor(delta[:n_node, 1] / scale, dtype=rel.dtype, device=rel.device)
    occ.zero_()
    occ[0, center_idx, 0] = -1.0
    return node_inputs

@torch.no_grad()
def build_intent_prob_map_mc_adv_fast(
    agent,
    base_map_info,
    robot_locations_view=None,
    global_intents_view=None,
    num_samples=384,
    segments=4,
    steps_per_segment=5,
    temperature=0.95,
    epsilon=0.08,
    time_decay=0.985,
    downsample=2,        # 2 或 4；1 表示不开
    blur_sigma_cells=2.0 # 管径≈(2*σ)cells，配合 downsample 更快
):
    H, W = base_map_info.map.shape
    # 下采样网格尺寸
    hS = H//downsample if downsample>1 else H
    wS = W//downsample if downsample>1 else W
    heat = np.zeros((hS, wS), dtype=np.float32)

    obs = agent.get_observation(robot_locations=robot_locations_view,
                                global_intents=global_intents_view)
    node_inputs, node_padding_mask, edge_mask, ci, ce, ep = obs
    device = next(agent.policy_net.parameters()).device
    n_node = agent.node_coords.shape[0]
    coords_np = agent.node_coords.copy()
    adj = agent.adjacent_matrix.copy()
    ksize = ce.size(1)

    # 采样
    for s in range(num_samples):
        ci_s = ci.clone(); ce_s = ce.clone(); ep_s = ep.clone()
        seq = [int(ci_s[0,0,0].item())]
        for seg in range(segments):
            for t in range(steps_per_segment):
                logits = agent.policy_net(node_inputs, node_padding_mask, edge_mask, ci_s, ce_s, ep_s)
                probs = torch.softmax(logits / max(1e-6, temperature), dim=1).squeeze(0)
                if epsilon>0:
                    K = probs.shape[0]
                    probs = (1-epsilon)*probs + epsilon*(torch.ones_like(probs)/K)
                    probs = probs / probs.sum()
                a = torch.multinomial(probs, 1).item()
                next_idx = int(ce_s[0, a, 0].item())
                seq.append(next_idx)

                # recenter（只改 rel_xy/occupancy）
                _recenter_minimal(node_inputs, n_node, coords_np, next_idx)

                # 刷新邻居
                neigh = np.argwhere(adj[next_idx] == 0).reshape(-1)
                ce_np = torch.tensor(neigh, dtype=torch.long, device=device).view(1, -1, 1)
                if ce_np.size(1) < ksize:
                    ce_s = torch.nn.functional.pad(ce_np, (0,0,0, ksize - ce_np.size(1)), value=0)
                    ep_s = torch.cat([
                        torch.zeros(1,1,ce_np.size(1), dtype=torch.int16, device=device),
                        torch.ones (1,1,ksize - ce_np.size(1), dtype=torch.int16, device=device)
                    ], dim=-1)
                else:
                    ce_s = ce_np[:, :ksize, :]
                    ep_s = torch.zeros(1,1,ksize, dtype=torch.int16, device=device)
                ci_s = torch.tensor([[[next_idx]]], device=device, dtype=torch.long)

        # 把整条链“细线”绘制到下采样网格上（之后统一做高斯模糊）
        for t in range(1, len(seq)):
            i0 = seq[t-1]; i1 = seq[t]
            p0 = coords_np[i0]; p1 = coords_np[i1]
            c0 = _to_cell_xy_fast(p0, base_map_info)
            c1 = _to_cell_xy_fast(p1, base_map_info)
            if c0[0] is None or c1[0] is None: continue
            # 下采样坐标
            x0, y0 = c0[0]//downsample, c0[1]//downsample
            x1, y1 = c1[0]//downsample, c1[1]//downsample
            rr, cc = sk_line(y0, x0, y1, x1)  # 注意先 y 后 x
            w = (time_decay ** (t-1))
            rr = np.clip(rr, 0, hS-1); cc = np.clip(cc, 0, wS-1)
            heat[rr, cc] += w

    # 高斯“管径”模糊（在小网格上，极快）
    if blur_sigma_cells and blur_sigma_cells>0:
        sigma = blur_sigma_cells
        heat = gaussian(heat, sigma=sigma, preserve_range=True)

    # 归一化并上采样到原图大小
    m = heat.max()
    if m>1e-8: heat = heat / m
    if downsample>1:
        heat_full = resize(heat, (H, W), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
    else:
        heat_full = heat.astype(np.float32)
    return heat_full

def draw_intent_probability_maps_mc_adv_fast(
    worker, step=0, use_pred_mean=True,
    num_samples=384, segments=4, steps_per_segment=5,
    temperature=0.95, epsilon=0.08,
    time_decay=0.985, downsample=2, blur_sigma_cells=2.0,
    ema_alpha=0.90, save_name=None
):
    robots = worker.robots
    n = len(robots)
    plt.switch_backend('agg')
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]
    last_locs = getattr(worker, 'last_known_locations', None)
    last_ints = getattr(worker, 'last_known_intents', None)
    im_ref = None

    # 以 0 号 agent 的底图规格为基准
    r0 = robots[0]
    base0 = r0.pred_mean_map_info if use_pred_mean else (r0.pred_max_map_info or r0.map_info)
    H, W = base0.map.shape
    # 初始化/取出 EMA 存储
    if not hasattr(worker, "__intent_heatmaps_fast__"):
        worker.__intent_heatmaps_fast__ = {i: np.zeros((H, W), dtype=np.float32) for i in range(n)}

    for i, r in enumerate(robots):
        ax = axes[i]; ax.axis('off')
        base_map = r.pred_mean_map_info if use_pred_mean else (r.pred_max_map_info or r.map_info)
        view_locs = last_locs[r.id] if last_locs is not None else None
        view_ints = last_ints[r.id] if last_ints is not None else None

        prob_new = build_intent_prob_map_mc_adv_fast(
            agent=r, base_map_info=base_map,
            robot_locations_view=view_locs, global_intents_view=view_ints,
            num_samples=num_samples, segments=segments, steps_per_segment=steps_per_segment,
            temperature=temperature, epsilon=epsilon, time_decay=time_decay,
            downsample=downsample, blur_sigma_cells=blur_sigma_cells
        )

        # EMA
        heat = worker.__intent_heatmaps_fast__[i]
        if heat.shape != prob_new.shape:
            worker.__intent_heatmaps_fast__[i] = np.zeros_like(prob_new)
            heat = worker.__intent_heatmaps_fast__[i]
        heat[:] = ema_alpha * heat + (1 - ema_alpha) * prob_new

        disp = heat.copy()
        m = disp.max()
        if m > 1e-8: disp /= m

        ax.imshow(base_map.map, cmap='gray', vmin=0, vmax=255)
        im_ref = ax.imshow(disp, cmap='hot', alpha=0.65, vmin=0.0, vmax=1.0)
        cx, cy = _to_cell_xy_fast(r.location, base_map)
        if cx is not None:
            ax.plot(cx, cy, 'mo', markersize=8)
        ax.set_title(f'Agent {r.id} Intent Prob (MC-FAST, EMA)')

    # 轻量显示：去掉 colorbar（如需可保留）
    out = save_name or f"intent_mc_fast_global_t{int(step):04d}.jpg"
    path = os.path.join(worker.run_dir, out)
    plt.savefig(path, dpi=110, bbox_inches='tight', quality=90)  # jpg 更小更快
    plt.close(fig)
    if hasattr(worker.env, 'frame_files'):
        worker.env.frame_files.append(path)
    print(f"[intent-viz] MC-FAST global saved: {path}")
