"""Batch evaluation of monocular 3D cricket delivery tracking across all clips."""
import sys, os, json, glob, cv2, numpy as np, pandas as pd
sys.path.insert(0, "src")

from geometry import *
from detect_stumps import detect, aggregate, save_check_image
from calibrate import build_correspondences, calibrate
from ball_io import load_ball
from trajectory import fit_trajectory, simulate, NAMES
from metrics import compute_metrics
from render import render_overlay_video, render_virtual_view

def process_clip(clip_id, clip_dir, stump_boxes_override=None):
    print("=" * 60)
    print(f"PROCESSING {clip_id.upper()} ({clip_dir})")
    print("=" * 60)

    video_path = os.path.join(clip_dir, "video.mp4")
    ball_csv = os.path.join(clip_dir, "ball.csv")
    gt_path = os.path.join(clip_dir, "gt.json")
    out_dir = os.path.join(clip_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    gt = json.load(open(gt_path)) if os.path.exists(gt_path) else {}

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = [width, height]
    print(f"[video] {width}x{height} @ {fps:.1f} fps, {n_frames} frames")

    # 1. Stumps
    stumps_path = os.path.join(clip_dir, "stumps.json")
    if stump_boxes_override:
        st = stump_boxes_override
    elif os.path.exists(stumps_path):
        st = json.load(open(stumps_path))
    else:
        dets = detect(video_path, "models/stumpbest.pt", 0, n_frames, 0.15)
        st = aggregate(dets)

    json.dump(st, open(stumps_path, "w"), indent=2)
    save_check_image(video_path, st, os.path.join(out_dir, "stump_check.png"))
    print(f"[stumps] near used: {st.get('near_frames_used')}, far used: {st.get('far_frames_used')}")

    # 2. Calibration
    W, P = build_correspondences(st)
    camera_prior = {"mu": [0.0, -4.0, 1.3], "sigma": [0.30, 0.30, 0.15], "weight": 1.0}
    cal = calibrate(W, P, size, camera_prior)
    cal_path = os.path.join(out_dir, "calibration.json")
    json.dump(cal, open(cal_path, "w"), indent=2)
    print(f"[calib] rmse={cal['rmse_px']:.2f}px  C={np.round(cal['C'], 2)}  f={cal['f']:.0f}  hfov={cal['hfov_deg']:.1f}deg")

    # 3. Ball labels
    frames, obs = load_ball(ball_csv, size, offset=0)
    print(f"[ball] {len(frames)} labels: frames {frames[0]}..{frames[-1]}")

    # 4. Trajectory fit
    fit = fit_trajectory(frames, obs, cal, fps, verbose=False)
    metrics, info = compute_metrics(fit["theta"], fit["ts"], fps)
    print(f"[fit] reprojection rms={fit['rms_px']:.2f}px  bounce~frame {fit['bounce_candidate']}")

    # 5. Result
    out_result = {
        "clip_id": clip_id,
        "metrics": metrics,
        "ground_truth": gt,
        "fit_rms_px": fit["rms_px"],
        "calibration_rmse_px": cal["rmse_px"],
        "theta": dict(zip(NAMES, map(float, fit["theta"]))),
        "C": cal["C"],
        "f": cal["f"]
    }
    json.dump(out_result, open(os.path.join(out_dir, "result.json"), "w"), indent=2)

    # 6. Render
    render_overlay_video(video_path, cal, fit, metrics, info, fps, os.path.join(out_dir, "overlay.mp4"))
    render_virtual_view(cal, fit, metrics, info, os.path.join(out_dir, "virtual_view.png"), size=(size[0], size[1]))
    print(f"[render] generated overlay.mp4 and virtual_view.png")

    gt_spd = gt.get("speed_mph", gt.get("speed_kph", 0))
    gt_spin = gt.get("spin_deg", float("nan"))
    gt_swing = gt.get("swing_sf", float("nan"))
    print(f"[summary] Speed: {metrics['speed_mph']:.1f} mph (GT: {gt_spd}) | Spin: {metrics.get('spin_deg', 0):.1f} deg (GT: {gt_spin}) | Swing dev: {metrics.get('swing_dev_cm', 0):.1f} cm (GT: {gt_swing} sf)")

    return out_result


def main():
    clips_dir = "data/clips"
    results = []

    # Clip 1 stump specification (from our verified detection/alignment)
    c1_stumps = json.load(open("data/interim/stumps.json"))
    res1 = process_clip("clip_1", os.path.join(clips_dir, "clip_1"), stump_boxes_override=c1_stumps)
    results.append(res1)

    # Clip 2 (has automatic near + far detections)
    # Box: near = [296.64, 869.69, 390.19, 1112.0], far = [346.86, 588.85, 369.4, 644.99]
    nb2 = [296.64, 869.69, 390.19, 1112.0]
    fb2 = [346.86, 588.85, 369.4, 644.99]
    wn2 = nb2[2] - nb2[0]
    wf2 = fb2[2] - fb2[0]
    c2_stumps = {
        "near": [
            [float(nb2[0] + 0.16 * wn2), float(nb2[1]), float(nb2[3] - 0.05 * (nb2[3] - nb2[1]))],
            [float((nb2[0] + nb2[2]) / 2.0), float(nb2[1]), float(nb2[3] - 0.05 * (nb2[3] - nb2[1]))],
            [float(nb2[2] - 0.16 * wn2), float(nb2[1]), float(nb2[3] - 0.05 * (nb2[3] - nb2[1]))],
        ],
        "near_frames_used": 103,
        "far": [
            [float(fb2[0] + 0.16 * wf2), float(fb2[1]), float(fb2[3] - 0.05 * (fb2[3] - fb2[1]))],
            [float((fb2[0] + fb2[2]) / 2.0), float(fb2[1]), float(fb2[3] - 0.05 * (fb2[3] - fb2[1]))],
            [float(fb2[2] - 0.16 * wf2), float(fb2[1]), float(fb2[3] - 0.05 * (fb2[3] - fb2[1]))],
        ],
        "far_frames_used": 100
    }
    res2 = process_clip("clip_2", os.path.join(clips_dir, "clip_2"), stump_boxes_override=c2_stumps)
    results.append(res2)

    # Clip 3 (1080x1350 @ 30fps)
    nb3 = [482.22, 944.66, 584.62, 1240.6]
    fb3 = [539.07, 718.1, 576.06, 805.59]
    wn3 = nb3[2] - nb3[0]
    wf3 = fb3[2] - fb3[0]
    c3_stumps = {
        "near": [
            [float(nb3[0] + 0.16 * wn3), float(nb3[1]), float(nb3[3] - 0.05 * (nb3[3] - nb3[1]))],
            [float((nb3[0] + nb3[2]) / 2.0), float(nb3[1]), float(nb3[3] - 0.05 * (nb3[3] - nb3[1]))],
            [float(nb3[2] - 0.16 * wn3), float(nb3[1]), float(nb3[3] - 0.05 * (nb3[3] - nb3[1]))],
        ],
        "near_frames_used": 12,
        "far": [
            [float(fb3[0] + 0.16 * wf3), float(fb3[1]), float(fb3[3] - 0.05 * (fb3[3] - fb3[1]))],
            [float((fb3[0] + fb3[2]) / 2.0), float(fb3[1]), float(fb3[3] - 0.05 * (fb3[3] - fb3[1]))],
            [float(fb3[2] - 0.16 * wf3), float(fb3[1]), float(fb3[3] - 0.05 * (fb3[3] - fb3[1]))],
        ],
        "far_frames_used": 17
    }
    res3 = process_clip("clip_3", os.path.join(clips_dir, "clip_3"), stump_boxes_override=c3_stumps)
    results.append(res3)

    # Clip 4 (1080x1350 @ 30fps)
    nb4 = [482.29, 943.87, 583.48, 1239.8]
    fb4 = [540.31, 718.39, 579.66, 799.63]
    wn4 = nb4[2] - nb4[0]
    wf4 = fb4[2] - fb4[0]
    c4_stumps = {
        "near": [
            [float(nb4[0] + 0.16 * wn4), float(nb4[1]), float(nb4[3] - 0.05 * (nb4[3] - nb4[1]))],
            [float((nb4[0] + nb4[2]) / 2.0), float(nb4[1]), float(nb4[3] - 0.05 * (nb4[3] - nb4[1]))],
            [float(nb4[2] - 0.16 * wn4), float(nb4[1]), float(nb4[3] - 0.05 * (nb4[3] - nb4[1]))],
        ],
        "near_frames_used": 12,
        "far": [
            [float(fb4[0] + 0.16 * wf4), float(fb4[1]), float(fb4[3] - 0.05 * (fb4[3] - fb4[1]))],
            [float((fb4[0] + fb4[2]) / 2.0), float(fb4[1]), float(fb4[3] - 0.05 * (fb4[3] - fb4[1]))],
            [float(fb4[2] - 0.16 * wf4), float(fb4[1]), float(fb4[3] - 0.05 * (fb4[3] - fb4[1]))],
        ],
        "far_frames_used": 8
    }
    res4 = process_clip("clip_4", os.path.join(clips_dir, "clip_4"), stump_boxes_override=c4_stumps)
    results.append(res4)

    # Write grand comparison table
    summary_path = "outputs/batch_summary.json"
    json.dump(results, open(summary_path, "w"), indent=2)
    print("\n" + "=" * 80)
    print("ALL 4 CLIPS PROCESSED SUCCESSFULLY! SUMMARY TABLE:")
    print("=" * 80)
    rows = []
    for r in results:
        m = r["metrics"]
        gt = r["ground_truth"]
        rows.append({
            "Clip": r["clip_id"],
            "Speed (Our)": f"{m['speed_mph']:.1f} mph",
            "Speed (GT)": f"{gt.get('speed_mph', round(gt.get('speed_kph', 0)/1.60934, 1))} mph",
            "Spin (Our)": f"{m.get('spin_deg', 0):.1f}°",
            "Spin (GT)": f"{gt.get('spin_deg', 0):.1f}°",
            "Length": m.get("length_class", "?"),
            "Swing Dev": f"{m.get('swing_dev_cm', 0):.1f} cm",
            "Swing GT": f"{gt.get('swing_sf', 0)} sf",
            "Fit RMS": f"{r['fit_rms_px']:.2f} px"
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
