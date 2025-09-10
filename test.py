# test.py
import os
import sys
import yaml
import time
import json
import torch
import random
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from model import PolicyNet
from worker import Worker
from parameter import *
from mapinpaint.networks import Generator
from mapinpaint.evaluator import Evaluator

from attention_viz import AttnRecorder  


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_predictor(device):

    config_path = f'{generator_path}/config.yaml'
    ckpts = sorted([f for f in os.listdir(generator_path) if f.startswith('gen') and f.endswith('.pt')])
    if not ckpts:
        raise FileNotFoundError(f"No generator checkpoint found in {generator_path}")
    checkpoint_path = os.path.join(generator_path, ckpts[0])

    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    generator = Generator(config['netG'], USE_GPU)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    predictor = Evaluator(config, generator, USE_GPU, N_GEN_SAMPLE)
    print(f"[Predictor] Map predictor loaded from {checkpoint_path}")
    return predictor


def maybe_load_policy_weights(policy_net: PolicyNet, device):

    ckpt_path = os.path.join(model_path, 'checkpoint.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'policy_model' in ckpt:
            policy_net.load_state_dict(ckpt['policy_model'])
            print(f"[Policy] Loaded policy weights from {ckpt_path}")
        else:
            print(f"[Policy] checkpoint.pth found but no 'policy_model' key; skipping.")
    else:
        print(f"[Policy] No checkpoint found at {ckpt_path}; using randomly initialized policy.")


def infer_gif_path(worker: Worker):

    return os.path.join(worker.run_dir, f"episode_{worker.global_step}_w{worker.meta_agent_id}.gif")


def _fmt_nan(x):
    try:
        return f"{float(x):.6f}"
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Evaluate DRL planner (single-process) and export GIF/metrics.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-gif", action="store_true", default=True, help="Save GIFs during evaluation")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"],
                        help="Force device (default: auto from parameter.py)")
    parser.add_argument("--metrics-out", type=str, default=None,
                        help="Optional: path to save a JSON metrics summary (full data including time series)")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Optional: path to save a CSV (default: checkpoints/.../test_metrics.csv)")
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')

    print(f"[Info] Using device: {device}")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(gifs_path, exist_ok=True)

    set_global_seed(args.seed)

    policy_net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    maybe_load_policy_weights(policy_net, device)

    recorder = AttnRecorder()
    recorder.register(policy_net)

    predictor = load_predictor(device=device)

    all_metrics = []
    for ep in range(1, args.episodes + 1):
        save_image = bool(args.save_gif)

        worker = Worker(meta_agent_id=0,
                        policy_net=policy_net,
                        predictor=predictor,
                        global_step=ep,
                        device=device,
                        save_image=save_image,
                        attn_recorder=recorder)

        t0 = time.time()
        worker.run_episode()
        dt = time.time() - t0

        pm = worker.perf_metrics
        total_travel  = float(pm.get('travel_dist', np.nan))
        max_travel    = float(pm.get('max_travel', np.nan))
        explored_rate = float(pm.get('explored_rate', np.nan))
        success       = bool(pm.get('success_rate', False))

        disc_free_mean_m2 = pm.get('disc_free_mean_m2', np.nan)
        disc_free_std_m2  = pm.get('disc_free_std_m2', np.nan)
        disc_free_cv      = pm.get('disc_free_cv', np.nan)
        disc_both_mean_m2 = pm.get('disc_both_mean_m2', np.nan)
        disc_both_std_m2  = pm.get('disc_both_std_m2', np.nan)
        disc_both_cv      = pm.get('disc_both_cv', np.nan)
        disc_free_per_agent = pm.get('disc_free_m2_per_agent', None)
        disc_occ_per_agent  = pm.get('disc_occ_m2_per_agent', None)

        gif_path = infer_gif_path(worker) if save_image else None
        if gif_path and os.path.exists(gif_path):
            gif_msg = gif_path
        elif gif_path:
            gif_msg = f"(scheduled at {gif_path}, but file not found—check logs)"
        else:
            gif_msg = "(disabled)"

        print(f"\n=== Episode {ep}/{args.episodes} ===")
        print(f"Elapsed: {dt:.2f}s")
        print(f"Total travel distance: {total_travel:.3f}")
        print(f"Max travel distance (per-agent): {max_travel:.3f}")
        print(f"Explored rate: {explored_rate:.4f}")
        print(f"Success flag: {success}")
        if not (np.isnan(disc_both_cv) if isinstance(disc_both_cv, float) else False):
            print("[Balance] discovered FREE: mean={:.2f} m^2, std={:.2f} m^2, CV={:.3f}".format(
                float(disc_free_mean_m2), float(disc_free_std_m2), float(disc_free_cv)))
            print("[Balance] discovered FREE+OCC: mean={:.2f} m^2, std={:.2f} m^2, CV={:.3f}".format(
                float(disc_both_mean_m2), float(disc_both_std_m2), float(disc_both_cv)))
            if disc_free_per_agent is not None:
                print("[Per-agent] discovered FREE m^2:", [f"{float(x):.1f}" for x in disc_free_per_agent])
            if disc_occ_per_agent is not None:
                print("[Per-agent] discovered OCC  m^2:", [f"{float(x):.1f}" for x in disc_occ_per_agent])
        else:
            print("[Balance] discovery stats unavailable (did you wire up Env.pop_discovery_masks / get_discovered_area / get_map_balance_stats?).")
        print(f"GIF: {gif_msg}")

        row = {
            "episode": ep,
            "elapsed_sec": dt,
            "total_travel": total_travel,
            "max_travel": max_travel,
            "explored_rate": explored_rate,
            "success": success,
            "gif": gif_path if gif_path and os.path.exists(gif_path) else None,

            "disc_free_mean_m2": float(disc_free_mean_m2) if disc_free_mean_m2 is not None else None,
            "disc_free_std_m2":  float(disc_free_std_m2)  if disc_free_std_m2  is not None else None,
            "disc_free_cv":      float(disc_free_cv)      if disc_free_cv      is not None else None,
            "disc_both_mean_m2": float(disc_both_mean_m2) if disc_both_mean_m2 is not None else None,
            "disc_both_std_m2":  float(disc_both_std_m2)  if disc_both_std_m2  is not None else None,
            "disc_both_cv":      float(disc_both_cv)      if disc_both_cv      is not None else None,

            "disc_free_m2_per_agent": disc_free_per_agent,
            "disc_occ_m2_per_agent":  disc_occ_per_agent,
            "disc_free_ts": pm.get("disc_free_ts", None),
            "disc_occ_ts":  pm.get("disc_occ_ts",  None),
        }
        all_metrics.append(row)

    if args.metrics_out:
        out_dir = os.path.dirname(os.path.abspath(args.metrics_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\n[Metrics] Saved metrics to {args.metrics_out}")

    csv_path = args.csv_out if args.csv_out else os.path.join(model_path, "test_metrics.csv")
    try:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "elapsed_sec", "total_travel", "max_travel",
                "explored_rate", "success", "gif",
                "disc_free_mean_m2", "disc_free_std_m2", "disc_free_cv",
                "disc_both_mean_m2", "disc_both_std_m2", "disc_both_cv"
            ])
            for r in all_metrics:
                writer.writerow([
                    r["episode"], f"{r['elapsed_sec']:.3f}",
                    f"{(r['total_travel'] if r['total_travel'] is not None else np.nan):.6f}",
                    f"{(r['max_travel'] if r['max_travel'] is not None else np.nan):.6f}",
                    f"{(r['explored_rate'] if r['explored_rate'] is not None else np.nan):.6f}",
                    int(r["success"]), r["gif"] if r["gif"] else "",
                    _fmt_nan(r.get("disc_free_mean_m2")),
                    _fmt_nan(r.get("disc_free_std_m2")),
                    _fmt_nan(r.get("disc_free_cv")),
                    _fmt_nan(r.get("disc_both_mean_m2")),
                    _fmt_nan(r.get("disc_both_std_m2")),
                    _fmt_nan(r.get("disc_both_cv")),
                ])
        print(f"[Metrics] CSV saved to {csv_path}")
    except Exception as e:
        print(f"[Metrics] Failed to write CSV: {e}")


if __name__ == "__main__":
    main()
