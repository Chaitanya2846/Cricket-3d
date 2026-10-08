import argparse, json, os
import numpy as np
from geometry import *
from detect_stumps import detect, aggregate, save_check_image
from calibrate import build_correspondences, calibrate
from ball_io import load_ball
from trajectory import fit_trajectory, simulate, NAMES
from metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--redo-stumps", action="store_true"); ap.add_argument("--mc", type=int, default=0,
                    help="Monte-Carlo runs for uncertainty (perturb calibration + pixel noise)")
    ap.add_argument("--no-render", action="store_true"); a = ap.parse_args()
    cfg = load_cfg(a.config); os.makedirs("outputs", exist_ok=True); os.makedirs("data/interim", exist_ok=True)
    fps, size = cfg["fps"], cfg["size"]

    sp = "data/interim/stumps.json"
    if a.redo_stumps or not os.path.exists(sp):
        st = aggregate(detect(cfg["video"], cfg["model"], *cfg["frames"], cfg["stump_conf"]))
        json.dump(st, open(sp, "w"), indent=2); save_check_image(cfg["video"], st, "outputs/stump_check.png")
    st = json.load(open(sp))
    W, P = build_correspondences(st, cfg.get("extra_points"))
    cal = calibrate(W, P, size, cfg["camera_prior"]); json.dump(cal, open("outputs/calibration.json", "w"), indent=2)
    print(f"[calib] rmse={cal['rmse_px']:.2f}px  C={np.round(cal['C'], 2)}  f={cal['f']:.0f}  hfov={cal['hfov_deg']:.0f}deg")

    frames, obs = load_ball(cfg["ball_csv"], size, cfg.get("ball_frame_offset", 0))
    print(f"[ball] {len(frames)} labelled frames: {frames[0]}..{frames[-1]}")
    fit = fit_trajectory(frames, obs, cal, fps)
    metrics, info = compute_metrics(fit["theta"], fit["ts"], fps)
    print(f"[fit] reprojection rms={fit['rms_px']:.2f}px  bounce~frame {fit['bounce_candidate']}")
    print("[theta]", {n: round(float(v), 3) for n, v in zip(NAMES, fit["theta"])})

    unc = {}
    if a.mc:
        rng = np.random.default_rng(0); runs = []
        for _ in range(a.mc):
            c2 = dict(cal); c2["C"] = (np.array(cal["C"]) + rng.normal(0, 0.10, 3)).tolist()
            c2["f"] = cal["f"] * (1 + rng.normal(0, 0.02))
            o2 = obs + rng.normal(0, 1.0, obs.shape)
            f2 = fit_trajectory(frames, o2, c2, fps, verbose=False, candidates=[list(frames).index(fit["bounce_candidate"])])
            m2, _ = compute_metrics(f2["theta"], f2["ts"], fps); runs.append(m2)
        for k in ("speed_mph", "length_m", "swing_dev_cm", "spin_deg", "line_x_at_stumps_m"):
            v = [r[k] for r in runs if k in r]
            if v: unc[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        print("[uncertainty]", json.dumps(unc, indent=1))

    gt = cfg.get("ground_truth", {})
    out = {"metrics": metrics, "ground_truth": gt, "uncertainty": unc, "theta": dict(zip(NAMES, map(float, fit["theta"]))),
           "fit_rms_px": fit["rms_px"], "calibration_rmse_px": cal["rmse_px"]}
    json.dump(out, open("outputs/result.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    if gt: print(f"[compare] speed {metrics['speed_mph']:.1f} mph vs GT {gt.get('speed_mph')} mph | "
                 f"spin {metrics.get('spin_deg', float('nan')):.1f} deg vs GT {gt.get('spin_deg')} deg")
    if not a.no_render:
        from render import render_overlay_video, render_virtual_view
        render_overlay_video(cfg["video"], cal, fit, metrics, info, fps, "outputs/overlay.mp4")
        render_virtual_view(cal, fit, metrics, info, "outputs/virtual_view.png")
        print("-> outputs/overlay.mp4, outputs/virtual_view.png")


if __name__ == "__main__":
    main()
