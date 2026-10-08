import argparse, json, os
import numpy as np
from geometry import *
from detect_stumps import detect, aggregate, save_check_image
from calibrate import build_correspondences, calibrate
from ball_io import load_ball
from trajectory import fit_trajectory, simulate, NAMES
from metrics import compute_metrics


def main():
    ap = argparse.ArgumentParser(description="Monocular 3D Cricket Delivery Tracking Pipeline")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml file")
    ap.add_argument("--video", default=None, help="Path to input video file (overrides config)")
    ap.add_argument("--ball", default=None, help="Path to ball labels CSV or YOLO txt folder (overrides config)")
    ap.add_argument("--stumps", default=None, help="Path to precomputed stumps.json (skips re-detection)")
    ap.add_argument("--out-dir", default="outputs", help="Output directory for generated artifacts")
    ap.add_argument("--fps", type=float, default=None, help="Video FPS (overrides config/auto-detection)")
    ap.add_argument("--redo-stumps", action="store_true", help="Force re-detection of stumps")
    ap.add_argument("--mc", type=int, default=0,
                    help="Monte-Carlo runs for uncertainty (perturb calibration + pixel noise)")
    ap.add_argument("--no-render", action="store_true", help="Skip rendering overlay and virtual view")
    a = ap.parse_args()

    cfg = load_cfg(a.config) if os.path.exists(a.config) else {}
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs("data/interim", exist_ok=True)

    # Resolve video path
    video_path = a.video or cfg.get("video")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Inspect video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS))
    vid_n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps = a.fps or cfg.get("fps") or vid_fps or 30.0
    size = cfg.get("size") or [vid_w, vid_h]
    frame_range = cfg.get("frames") or [0, min(vid_n_frames, 140)]
    stump_model = cfg.get("model", "models/stumpbest.pt")
    stump_conf = cfg.get("stump_conf", 0.15)
    cam_prior = cfg.get("camera_prior", {"mu": [0.0, -4.0, 1.3], "sigma": [0.30, 0.30, 0.15], "weight": 1.0})

    # 1. Detect / Load Stumps
    if a.stumps and os.path.exists(a.stumps):
        sp = a.stumps
    elif not a.video and os.path.exists("data/interim/stumps.json"):
        sp = "data/interim/stumps.json"
    else:
        sp = os.path.join(a.out_dir, "stumps.json")

    if a.redo_stumps or not os.path.exists(sp):
        st = aggregate(detect(video_path, stump_model, frame_range[0], frame_range[1], stump_conf))
        json.dump(st, open(sp, "w"), indent=2)
        save_check_image(video_path, st, os.path.join(a.out_dir, "stump_check.png"))
    st = json.load(open(sp))

    # 2. Camera Calibration
    W, P = build_correspondences(st, cfg.get("extra_points"))
    cal = calibrate(W, P, size, cam_prior)
    json.dump(cal, open(os.path.join(a.out_dir, "calibration.json"), "w"), indent=2)
    print(f"[calib] rmse={cal['rmse_px']:.2f}px  C={np.round(cal['C'], 2)}  f={cal['f']:.0f}  hfov={cal['hfov_deg']:.0f}deg")

    # 3. Load Ball Observations
    ball_path = a.ball or cfg.get("ball_csv")
    if not ball_path or not os.path.exists(ball_path):
        raise FileNotFoundError(f"Ball observations not found: {ball_path}. Provide via --ball <path/to/ball.csv>")
    frames, obs = load_ball(ball_path, size, cfg.get("ball_frame_offset", 0))
    print(f"[ball] {len(frames)} labelled frames: {frames[0]}..{frames[-1]}")

    # 4. Fit 3D Trajectory & Compute Metrics
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
    res_path = os.path.join(a.out_dir, "result.json")
    json.dump(out, open(res_path, "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    if gt:
        print(f"[compare] speed {metrics['speed_mph']:.1f} mph vs GT {gt.get('speed_mph')} mph | "
              f"spin {metrics.get('spin_deg', float('nan')):.1f} deg vs GT {gt.get('spin_deg')} deg")

    # 5. Render Video & Virtual 3D View
    if not a.no_render:
        from render import render_overlay_video, render_virtual_view
        out_overlay = os.path.join(a.out_dir, "overlay.mp4")
        out_virtual = os.path.join(a.out_dir, "virtual_view.png")
        render_overlay_video(video_path, cal, fit, metrics, info, fps, out_overlay, sf_scale=gt.get("swing_sf"))
        render_virtual_view(cal, fit, metrics, info, out_virtual, sf_scale=gt.get("swing_sf"))
        print(f"-> {out_overlay}, {out_virtual}")


if __name__ == "__main__":
    main()
