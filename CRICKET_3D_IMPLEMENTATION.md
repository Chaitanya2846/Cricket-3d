# Monocular 3D Cricket Delivery Tracking — Implementation Guide

Goal: from one phone video (camera behind the bowler's stumps) + stump detections + labelled ball 2D points, recover the 3D ball path and show **Speed, Swing, Spin, Length, Line** in a FullTrack-style view.

> Every code block that starts with `# FILE: <path>` is extracted automatically into your project by one command in Phase 0. You do not need to copy-paste files one by one.

---

## 0. What I found in your uploads

| File | What it is | Consequence |
|---|---|---|
| `Full_Track_video001.mp4` | 360×640, **30 fps**, 4.0 s, 120 frames. A screen-recording of the FullTrack app. Frames ~0–100 are the raw footage, and the last frames switch to the app's own 3D replay. | Ball is only a few pixels wide, and 30 fps means ~0.75 m of travel per frame at 52 mph. Use only the raw-footage frames for calibration/tracking. The replay frame shows the **ground truth: Swing 0.2 sf, Spin 3.6°, Speed 52 mph**. |
| `stumpbest_pt.zip` | An **unzipped PyTorch checkpoint** (folder `last/…`), not a normal `.pt`. Ultralytics **YOLO11s, `DetectionModel`, task `detect`, one class `stump`**, trained with ultralytics 8.4.149. | It outputs **bounding boxes only (no keypoints)**. Stump top = top-centre of box, stump bottom = bottom-centre. The folder name `last` suggests this is `last.pt`, not `best.pt`. Phase 0 re-zips it into a loadable `stumpbest.pt` (I also attach a ready-made one). |
| `1791386218084_image.png` | FullTrack output reference: Swing **6.8 sf**, Spin **0.7°**, Speed **76 mph**, red trajectory, blue line from bounce to stumps. | Target look for Phase 7. |

Two things from the frames that shape the design:
- **Two sets of stumps** are visible: near (bowler's end) and far (batter's end, partly hidden by the batter). Near stumps give accurate pixels; far stumps are small and noisier but are what gives depth.
- The setup (FullTrack's video): **tripod ~1.3 m high, 4 m behind the bowler's stumps, phone tilted slightly down.** These become *soft priors* in calibration, not hard constants.

---

## 1. Method in one page

**World frame (metres):** origin = base of the middle stump at the bowler's end; `X` = right (as seen from the camera), `Y` = along the pitch toward the batter, `Z` = up. Batter's stumps are at `Y = 20.12`.

**Calibration (Phase 3):** 3D↔2D correspondences = base and top of each stump at both ends (up to 12 points) → solve camera pose + focal length by least squares, with soft priors from the FullTrack setup (`x≈0, y≈−4, z≈1.3 m`).

**Why not just a homography?** A homography only maps the *ground plane*, and the ball is in the air. The stumps are only 0.23 m wide, so lateral geometry is weakly constrained. We use a full pose (R, C, f) and treat the ground plane only for the bounce.

**3D trajectory (Phase 5):** one pixel ray per frame cannot give depth, so we fit a **physical model** whose projection must match all ball pixels:
- Flight: gravity + quadratic drag + constant lateral acceleration `ax` (swing).
- Bounce on `Z = ball radius`: restitution `e`, horizontal speed retention `kh`, horizontal direction change `delta` (spin/seam deviation).
- 10 parameters fitted to ~20–40 pixel measurements, with the bounce frame found by trying every candidate frame.

**Metrics (Phase 6):** speed (mph), swing (lateral deviation and angle in air), spin/deviation (° after bounce), length (bounce distance from batter's stumps), line and height at the stumps.

**Honest limits:** monocular depth is ambiguous, 30 fps is coarse, and "sf" (swing unit) and FullTrack's exact definitions are proprietary. We compute physically meaningful quantities, then calibrate a scale to FullTrack's numbers using several clips (Phase 8).

---

## 2. Phase 0 — Project setup

Python 3.10+ (CPU is enough; a GPU only speeds up stump detection).

```bash
mkdir cricket3d && cd cricket3d
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -U pip
pip install ultralytics opencv-python numpy scipy pandas matplotlib pyyaml tqdm pytest
mkdir -p data/raw data/interim data/labels models src tools tests outputs
```

Put your files in place (adjust the source paths):

```bash
cp ~/Downloads/Full_Track_video001.mp4 data/raw/
cp ~/Downloads/CRICKET_3D_IMPLEMENTATION.md .
cp ~/Downloads/1791386218084_image.png data/raw/reference_output.png
```

**Model file.** Easiest: use the `stumpbest.pt` attached with this guide:

```bash
cp ~/Downloads/stumpbest.pt models/stumpbest.pt
```

Or rebuild it from your zip (PyTorch `.pt` files are zip archives, and yours was unzipped):

```bash
unzip stumpbest_pt.zip -d /tmp/stump_unz
python - <<'EOF'
import zipfile, os
src="/tmp/stump_unz"; out="models/stumpbest.pt"
with zipfile.ZipFile(out,"w",zipfile.ZIP_STORED) as z:
    for root,_,files in os.walk(src):
        for f in sorted(files):
            p=os.path.join(root,f); z.write(p, os.path.relpath(p,src))
print("wrote",out)
EOF
```

Verify it loads:

```bash
python -c "from ultralytics import YOLO; m=YOLO('models/stumpbest.pt'); print(m.task, m.names)"
```
Expected: `detect {0: 'stump'}`.

**Extract all code from this guide into the project** (one command):

```bash
python - <<'EOF'
import re, pathlib
t = open("CRICKET_3D_IMPLEMENTATION.md", encoding="utf8").read()
for m in re.finditer(r"```(\w+)\n(?:#|//) FILE: (\S+)\n(.*?)```", t, re.S):
    p = pathlib.Path(m.group(2)); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(m.group(3), encoding="utf8"); print("wrote", p)
EOF
```

Final layout:

```
cricket3d/
├── config.yaml
├── data/raw/            video, reference image
├── data/interim/        stumps.json, frames
├── data/labels/         ball.csv (your labelled data)
├── models/stumpbest.pt
├── src/                 geometry, detect_stumps, calibrate, ball_io, trajectory, metrics, render, run_pipeline
├── tests/test_synthetic.py
└── outputs/             calibration.json, result.json, overlay.mp4, virtual_view.png
```

### Config

```yaml
# FILE: config.yaml
video: data/raw/Full_Track_video001.mp4
model: models/stumpbest.pt
fps: 30
size: [360, 640]            # width, height of the video
frames: [0, 100]            # raw-footage range [start, end); exclude the app's 3D replay
ball_csv: data/labels/ball.csv
ball_frame_offset: 0        # set to -1 if your labels are 1-based
stump_conf: 0.25

# FullTrack setup (soft priors): camera 4 m behind bowler's stumps, 1.3 m high
camera_prior:
  mu: [0.0, -4.0, 1.3]
  sigma: [0.30, 0.30, 0.15]   # metres; widen if you suspect a different setup
  weight: 1.0

# Optional extra ground-truth points: world (x,y,z) in metres -> pixel (u,v).
# Example: far popping-crease/return-crease intersections read off any frame.
extra_points: []
#  - {name: far_pop_left, world: [-1.32, 18.90, 0.0], px: [150.0, 250.0]}

ground_truth:               # for the validation phase (from the app's replay)
  speed_mph: 52
  swing_sf: 0.2
  spin_deg: 3.6
```

---

## 3. Phase 1 — Check the inputs and find the replay cut

```bash
ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames -of default=nw=1 data/raw/Full_Track_video001.mp4
mkdir -p data/interim/frames
ffmpeg -loglevel error -i data/raw/Full_Track_video001.mp4 -vsync 0 data/interim/frames/%04d.png
ls data/interim/frames | wc -l      # expect 120
```

Look at a few frames (`0001.png`, `0060.png`, `0110.png`). Frame index 110 is the app's 3D replay. Set `frames: [0, N]` in `config.yaml` so that `N` is just before the replay starts. To find the cut automatically:

```python
# FILE: tools/find_replay_cut.py
"""Prints the frame where the footage changes abruptly (raw video -> app 3D replay)."""
import sys, cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); prev = None; diffs = []
while True:
    ok, fr = cap.read()
    if not ok: break
    g = cv2.cvtColor(cv2.resize(fr, (90, 160)), cv2.COLOR_BGR2GRAY).astype(float)
    diffs.append(0 if prev is None else np.abs(g - prev).mean()); prev = g
d = np.array(diffs); k = int(d.argmax())
print("largest change at frame", k, "(set frames: [0,", k, "])  diff =", round(d[k], 1))
```

```bash
python tools/find_replay_cut.py data/raw/Full_Track_video001.mp4
```

**Ground truth for this clip** (already in `config.yaml`): 52 mph, swing 0.2 sf, spin 3.6°.

---

## 4. Phase 2 — Stump detection

Quick visual sanity check with the CLI:

```bash
yolo detect predict model=models/stumpbest.pt source=data/interim/frames/0001.png conf=0.25 save=True
```

Full detection over the raw-footage frames, then aggregation. The stumps do not move, so we take the **median over all frames where a complete set of 3 is seen**, which is robust to the bowler walking in front.

```python
# FILE: src/geometry.py
import numpy as np, cv2, yaml

PITCH_L = 20.12      # stump to stump (m)
STUMP_H = 0.711      # m
STUMP_W = 0.2286     # outer edge to outer edge (m)
STUMP_D = 0.0381     # stump diameter (m)
POP = 1.22           # popping crease distance from stumps (m)
RET = 1.32           # return crease half-width (m)
PITCH_HW = 1.525     # half pitch width (m)
BALL_R = 0.0356      # m
G = 9.81
MPS_TO_MPH = 2.23694
STUMP_X = np.array([-(STUMP_W - STUMP_D) / 2, 0.0, (STUMP_W - STUMP_D) / 2])  # stump centres


def load_cfg(path="config.yaml"):
    return yaml.safe_load(open(path, encoding="utf8"))


def _R(rvec):
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def project(Xw, rvec, C, f, size):
    """World points (N,3) -> pixels (N,2).  x_cam = R (X - C); OpenCV axes (x right, y down, z fwd)."""
    Xc = (np.atleast_2d(np.asarray(Xw, float)) - np.asarray(C, float)) @ _R(rvec).T
    z = np.clip(Xc[:, 2], 1e-6, None)
    return np.stack([f * Xc[:, 0] / z + size[0] / 2, f * Xc[:, 1] / z + size[1] / 2], 1)


def backproject_to_plane(px, rvec, C, f, size, z_plane):
    """Intersect the camera ray through pixel px with the horizontal plane Z = z_plane."""
    d_cam = np.array([(px[0] - size[0] / 2) / f, (px[1] - size[1] / 2) / f, 1.0])
    d = _R(rvec).T @ d_cam
    s = (z_plane - C[2]) / d[2]
    return np.asarray(C, float) + s * d


def lookat_rvec(C, target):
    """Rotation vector (world->camera) for a camera at C looking at target, with Z up."""
    fwd = np.asarray(target, float) - np.asarray(C, float); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return cv2.Rodrigues(np.stack([right, down, fwd]))[0].ravel()


def stump_world_points(y):
    """Base and top of the 3 stumps at pitch position y -> (6,3): [b0,t0,b1,t1,b2,t2]."""
    pts = []
    for x in STUMP_X:
        pts += [[x, y, 0.0], [x, y, STUMP_H]]
    return np.array(pts)
```

```python
# FILE: src/detect_stumps.py
"""Detect stumps on the raw-footage frames and aggregate into near/far sets.
Output: data/interim/stumps.json  {near: [[cx, y_top, y_bottom] x3], far: [...]} (pixels, left->right)."""
import argparse, json
import numpy as np, cv2
from ultralytics import YOLO
from geometry import load_cfg


def detect(video, model_path, f0, f1, conf):
    model = YOLO(model_path); cap = cv2.VideoCapture(video); dets = []; i = 0
    while True:
        ok, fr = cap.read()
        if not ok or i >= f1: break
        if i >= f0:
            r = model.predict(fr, conf=conf, verbose=False)[0]
            boxes = [[*map(float, xy), float(c)] for xy, c in
                     zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            dets.append({"frame": i, "boxes": boxes})
        i += 1
    return dets


def aggregate(dets):
    hs = np.array([b[3] - b[1] for d in dets for b in d["boxes"]])
    if len(hs) == 0: raise SystemExit("No stumps detected. Lower stump_conf or check the frame range.")
    lh = np.sort(np.log(hs)); gaps = np.diff(lh); thr = None
    if len(gaps) and gaps.max() > np.log(1.8):                 # near stumps are much taller than far ones
        j = gaps.argmax(); thr = float(np.exp((lh[j] + lh[j + 1]) / 2))
    per = {"near": [], "far": []}
    for d in dets:
        bs = d["boxes"]
        groups = {"near": bs, "far": []} if thr is None else {
            "near": [b for b in bs if b[3] - b[1] >= thr], "far": [b for b in bs if b[3] - b[1] < thr]}
        for k, g in groups.items():
            if len(g) == 3: per[k].append(sorted(g, key=lambda b: b[0]))
    out = {"height_split_px": thr}
    for k, v in per.items():
        if len(v) >= 3:
            a = np.median(np.array(v), axis=0)                  # (3,5): x1,y1,x2,y2,conf
            out[k] = [[float((x1 + x2) / 2), float(y1), float(y2)] for x1, y1, x2, y2, _ in a]
            out[k + "_frames_used"] = len(v)
    return out


def save_check_image(video, stumps, out_png):
    cap = cv2.VideoCapture(video); ok, fr = cap.read()
    for k, col in (("near", (0, 255, 0)), ("far", (0, 200, 255))):
        for cx, yt, yb in stumps.get(k, []):
            cv2.circle(fr, (int(cx), int(yt)), 3, col, -1); cv2.circle(fr, (int(cx), int(yb)), 3, (0, 0, 255), -1)
            cv2.line(fr, (int(cx), int(yt)), (int(cx), int(yb)), col, 1)
    cv2.imwrite(out_png, fr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
    cfg = load_cfg(a.config)
    dets = detect(cfg["video"], cfg["model"], cfg["frames"][0], cfg["frames"][1], cfg["stump_conf"])
    st = aggregate(dets)
    json.dump(st, open("data/interim/stumps.json", "w"), indent=2)
    save_check_image(cfg["video"], st, "outputs/stump_check.png")
    print(json.dumps(st, indent=2)); print("-> data/interim/stumps.json, outputs/stump_check.png")
```

```bash
cd src && python detect_stumps.py --config ../config.yaml && cd ..
```

(If you prefer running from the project root, use `PYTHONPATH=src python src/detect_stumps.py`. All commands below assume the root with `PYTHONPATH=src`.)

```bash
export PYTHONPATH=src            # Windows PowerShell: $env:PYTHONPATH="src"
python src/detect_stumps.py
```

**Check `outputs/stump_check.png`:** green/orange lines must sit on the stumps (green = near set, orange = far set). Problems to look for:
- **Bowler occludes the stumps** → fewer than 3 boxes in most frames. Widen `frames`, or write `data/interim/stumps.json` by hand (same format).
- **Box bottoms include the black base plug** → the "bottom" point is a few cm below the real stump base. Either retrain/relabel, or shrink: in `aggregate`, replace `y2` with `y2 - 0.05*(y2-y1)`.
- **Far set missing** (hidden behind the batter) → calibration still works with the near set + priors, but depth accuracy drops. Add `extra_points` (e.g. visible crease corners) in `config.yaml`.

---

## 5. Phase 3 — Camera calibration

Least squares over 7 unknowns (rotation 3, camera position 3, focal length 1) with the FullTrack setup as soft priors. We start from many initial focal lengths and keep the best minimum, because focal length, height and tilt trade off against each other when few points are available.

```python
# FILE: src/calibrate.py
import json
import numpy as np
from scipy.optimize import least_squares
from geometry import *


def build_correspondences(stumps, extra_points=()):
    W, P = [], []
    for key, Y in (("near", 0.0), ("far", PITCH_L)):
        if key not in stumps: continue
        for i, (cx, yt, yb) in enumerate(stumps[key]):
            x = STUMP_X[i]
            W += [[x, Y, 0.0], [x, Y, STUMP_H]]; P += [[cx, yb], [cx, yt]]
    for e in extra_points or []:
        W.append(e["world"]); P.append(e["px"])
    return np.array(W, float), np.array(P, float)


def _res(p, W, P, size, mu, sg, wp):
    r = (project(W, p[:3], p[3:6], p[6], size) - P).ravel()
    return np.concatenate([r, (p[3:6] - mu) / sg * wp])


def calibrate(W, P, size, prior):
    mu = np.array(prior["mu"], float); sg = np.array(prior["sigma"], float); wp = prior.get("weight", 1.0)
    lo = [-np.inf] * 6 + [200.0]; hi = [np.inf] * 6 + [3000.0]; best = None
    for ty in (8.0, 12.0, 16.0):                              # initial aim point down the pitch
        rv0 = lookat_rvec(mu, [0, ty, 0.0])
        for f0 in np.linspace(300, 1600, 27):
            sol = least_squares(_res, np.r_[rv0, mu, f0], args=(W, P, size, mu, sg, wp),
                                bounds=(lo, hi), x_scale=[0.05] * 3 + [0.2] * 3 + [50.0])
            if best is None or sol.cost < best.cost: best = sol
    p = best.x
    rep = (project(W, p[:3], p[3:6], p[6], size) - P)
    rmse = float(np.sqrt((rep ** 2).sum(1).mean()))
    f = float(p[6])
    return {"rvec": p[:3].tolist(), "C": p[3:6].tolist(), "f": f, "size": list(size),
            "rmse_px": rmse, "n_points": int(len(W)),
            "hfov_deg": float(np.degrees(2 * np.arctan(size[0] / 2 / f))),
            "vfov_deg": float(np.degrees(2 * np.arctan(size[1] / 2 / f)))}


if __name__ == "__main__":
    cfg = load_cfg(); st = json.load(open("data/interim/stumps.json"))
    W, P = build_correspondences(st, cfg.get("extra_points"))
    cal = calibrate(W, P, cfg["size"], cfg["camera_prior"])
    json.dump(cal, open("outputs/calibration.json", "w"), indent=2)
    print(json.dumps(cal, indent=2))
    if cal["rmse_px"] > 3: print("WARNING: reprojection error > 3 px -> check stump_check.png / priors.")
    mu = np.array(cfg["camera_prior"]["mu"])
    print("camera moved from prior by (m):", np.round(np.array(cal["C"]) - mu, 2))
```

```bash
python src/calibrate.py
```

**How to judge it:**
- `rmse_px` < ~2 px is good (the clip is only 360×640). Much larger means wrong stump points or a wrong prior.
- `C` should be near `(0, −4, 1.3)`. If it drifts by more than ~0.5 m, the data is fighting the prior: re-check the stump boxes before loosening `sigma`.
- `hfov_deg` for a phone main camera is typically 55–75° (for the cropped portrait frame). Wildly different values mean calibration is not trustworthy.

---

## 6. Phase 4 — Ball data (your labelled data)

Put your labels in `data/labels/ball.csv`. Supported formats:

1. **CSV** with columns `frame,x,y` (also accepted: `cx,cy` or `u,v`). Values in pixels, or normalised 0–1 (auto-detected). Missing frames are fine, so simply leave them out.
2. **YOLO txt folder** — set `ball_csv` to a directory of files like `clip_000012.txt` (`class cx cy w h`, normalised). The frame index is the last number in the file name.

Set `ball_frame_offset: -1` in `config.yaml` if your frame numbers start at 1.

```python
# FILE: src/ball_io.py
import re, glob, os
import numpy as np, pandas as pd


def load_ball(path, size, offset=0):
    """-> frames (N,), obs (N,2) pixel coordinates sorted by frame."""
    rows = []
    if os.path.isdir(path):
        for f in glob.glob(os.path.join(path, "*.txt")):
            nums = re.findall(r"\d+", os.path.basename(f))
            line = open(f).read().strip().splitlines()
            if not nums or not line: continue
            v = line[0].split()
            rows.append((int(nums[-1]), float(v[1]), float(v[2])))
    else:
        df = pd.read_csv(path); cols = {c.lower().strip(): c for c in df.columns}
        xc = next(cols[k] for k in ("x", "cx", "u") if k in cols)
        yc = next(cols[k] for k in ("y", "cy", "v") if k in cols)
        fc = cols.get("frame", df.columns[0])
        if "visible" in cols: df = df[df[cols["visible"]] > 0]
        rows = list(zip(df[fc].astype(int), df[xc].astype(float), df[yc].astype(float)))
    a = np.array(sorted(rows), float)
    if len(a) == 0: raise SystemExit("No ball labels found in " + path)
    if a[:, 1:].max() <= 1.5:                                  # normalised -> pixels
        a[:, 1] *= size[0]; a[:, 2] *= size[1]
    return a[:, 0].astype(int) + offset, a[:, 1:3]
```

Checks before fitting: the labels must be in **the same pixel space as the video** (360×640, not the original phone resolution), and there should be ≥ 8 consecutive-ish points that include the bounce. If your labelling tool used another resolution, rescale in the CSV first.

---

## 7. Phase 5 — 3D trajectory fit (the core)

The model is simulated numerically (small time steps), projected through the calibrated camera, and compared to your pixels. For every candidate bounce frame we build a physically sensible starting guess (ball-plane intersections at release height and at the bounce) and refine; the lowest-cost fit wins.

```python
# FILE: src/trajectory.py
import numpy as np
from scipy.optimize import least_squares
from geometry import *

KD = 0.0061   # quadratic drag a = -KD*|v|*v   (Cd~0.4, r=35.6 mm, m=156 g)
NAMES = ["x0", "y0", "z0", "vx", "vy", "vz", "ax", "e", "kh", "delta"]
LO = np.array([-2.0, -3.0, 0.5, -8.0, 10.0, -15.0, -6.0, 0.25, 0.40, np.radians(-12)])
HI = np.array([2.0, 12.0, 3.4, 8.0, 50.0, 10.0, 6.0, 0.80, 1.00, np.radians(12)])
XS = np.array([0.2, 0.5, 0.2, 1.0, 2.0, 1.0, 1.0, 0.1, 0.1, 0.05])


def simulate(theta, ts, fps, dt_factor=8):
    """theta = (x0,y0,z0, vx,vy,vz, ax, e, kh, delta).  Position at first observed frame = (x0,y0,z0).
    ax: lateral (swing) acceleration before bounce; e: vertical restitution; kh: horizontal speed
    retention at bounce; delta: rotation of horizontal velocity at bounce (rad).
    Returns positions at times ts (N,3) and info with dense path + bounce."""
    x0, y0, z0, vx, vy, vz, ax, e, kh, delta = theta
    dt = 1.0 / (fps * dt_factor)
    p = np.array([x0, y0, z0], float); v = np.array([vx, vy, vz], float); t = 0.0
    T = float(np.max(ts)); T_list, P_list, V_list = [0.0], [p.copy()], [v.copy()]
    bounced = False; bounce = None
    while t < T + dt:
        a = -KD * np.linalg.norm(v) * v + np.array([0.0 if bounced else ax, 0.0, -G])
        v_n = v + a * dt; p_n = p + v_n * dt
        if (not bounced) and p_n[2] <= BALL_R and v_n[2] < 0:
            s = (p[2] - BALL_R) / (p[2] - p_n[2] + 1e-12)
            pb = p + (p_n - p) * s; vb = v + (v_n - v) * s; tb = t + dt * s
            c, sn = np.cos(delta), np.sin(delta)
            vh = np.array([c * vb[0] - sn * vb[1], sn * vb[0] + c * vb[1]]) * kh
            v_out = np.array([vh[0], vh[1], -e * vb[2]])
            bounce = {"t": float(tb), "pos": pb.copy(), "v_in": vb.copy(), "v_out": v_out.copy()}
            bounced = True; t = tb; p = pb.copy(); v = v_out
            T_list.append(t); P_list.append(p.copy()); V_list.append(v.copy()); continue
        t += dt; p = p_n; v = v_n
        T_list.append(t); P_list.append(p.copy()); V_list.append(v.copy())
    Tt = np.array(T_list); Pp = np.array(P_list)
    pos = np.stack([np.interp(ts, Tt, Pp[:, k]) for k in range(3)], 1)
    return pos, {"t": Tt, "pos": Pp, "vel": np.array(V_list), "bounce": bounce}


def _res(theta, ts, obs, cal, fps, sig):
    P, _ = simulate(theta, ts, fps, dt_factor=6)
    pix = project(P, cal["rvec"], cal["C"], cal["f"], cal["size"])
    return ((pix - obs) / sig).ravel()


def _init(k, ts, obs, cal, z0=2.2):
    P0 = backproject_to_plane(obs[0], cal["rvec"], cal["C"], cal["f"], cal["size"], z0)
    B = backproject_to_plane(obs[k], cal["rvec"], cal["C"], cal["f"], cal["size"], BALL_R)
    tb = ts[k]
    vxy = (B[:2] - P0[:2]) / tb
    vz = (BALL_R - z0 + 0.5 * G * tb ** 2) / tb
    return np.array([P0[0], P0[1], z0, vxy[0], vxy[1], vz, 0.0, 0.55, 0.80, 0.0])


def fit_trajectory(frames, obs, cal, fps, sig=2.0, verbose=True, candidates=None):
    frames = np.asarray(frames); obs = np.asarray(obs, float); ts = (frames - frames[0]) / fps
    best = None
    for k in (candidates if candidates is not None else range(2, len(frames) - 1)):
        th0 = _init(k, ts, obs, cal)
        if not (10 < th0[4] < 50): continue
        th0 = np.clip(th0, LO + 1e-6, HI - 1e-6)
        sol = least_squares(_res, th0, bounds=(LO, HI), x_scale=XS, max_nfev=60,
                            args=(ts, obs, cal, fps, sig))
        if verbose: print(f"  bounce candidate frame {frames[k]:4d}: cost={sol.cost:9.2f}")
        if best is None or sol.cost < best[0].cost: best = (sol, k)
    if best is None: raise SystemExit("No feasible start: check ball labels / calibration.")
    sol, k = best
    n = len(ts); rms = float(np.sqrt(2 * sol.cost * sig ** 2 / n / 2))  # px, rms per coordinate pair
    return {"theta": sol.x, "ts": ts, "frames": frames, "bounce_candidate": int(frames[k]),
            "cost": float(sol.cost), "rms_px": rms}
```

```python
# FILE: src/metrics.py
import numpy as np
from geometry import *
from trajectory import simulate

LENGTH_BANDS = [(0.0, 2.0, "Yorker/full toss"), (2.0, 4.0, "Full"), (4.0, 6.0, "Good length"),
                (6.0, 8.0, "Short of a length"), (8.0, 99.0, "Short")]   # metres from batter's stumps (adjustable)


def compute_metrics(theta, ts, fps):
    x0, y0, z0, vx, vy, vz, ax, e, kh, delta = theta
    _, info = simulate(theta, ts, fps, dt_factor=20)
    b = info["bounce"]; m = {}
    m["speed_mph"] = float(np.linalg.norm([vx, vy, vz]) * MPS_TO_MPH)   # at first labelled frame (~release)
    m["speed_kph"] = m["speed_mph"] * 1.60934
    if b is not None:
        tb = b["t"]
        m["bounce_xy_m"] = [float(b["pos"][0]), float(b["pos"][1])]
        m["length_m"] = float(PITCH_L - b["pos"][1])               # distance from batter's stumps
        m["length_class"] = next((n for lo, hi, n in LENGTH_BANDS if lo <= m["length_m"] < hi), "?")
        m["swing_dev_cm"] = float(100 * 0.5 * ax * tb ** 2)         # lateral deviation caused by swing (cm)
        m["swing_deg"] = float(np.degrees(np.arctan2(ax * tb, b["v_in"][1])))
        m["spin_deg"] = float(np.degrees(delta))                    # horizontal direction change at bounce
        m["speed_after_bounce_mph"] = float(np.linalg.norm(b["v_out"]) * MPS_TO_MPH)
        m["bounce_height_ratio"] = float(e)
    else:
        m["note"] = "no bounce detected (full toss)"
    P, T = info["pos"], info["t"]
    idx = np.where(P[:, 1] >= PITCH_L)[0]
    if len(idx):
        i = idx[0]; w = (PITCH_L - P[i - 1, 1]) / (P[i, 1] - P[i - 1, 1] + 1e-12)
        q = P[i - 1] + w * (P[i] - P[i - 1])
        m["line_x_at_stumps_m"] = float(q[0]); m["height_at_stumps_m"] = float(q[2])
        m["hits_stumps"] = bool(abs(q[0]) <= STUMP_W / 2 and 0 <= q[2] <= STUMP_H)
    return m, info
```

Notes on what these numbers mean (so you can compare with FullTrack later):
- `speed_mph` is the speed at the first labelled ball frame. If your labels start late, speed is slightly lower than release speed. Label from release if possible.
- `swing_dev_cm` / `swing_deg` are physical quantities; FullTrack's **"sf"** unit is proprietary. Use Phase 8 to learn a scale factor between your swing value and their "sf".
- `spin_deg` is the change in horizontal direction at the bounce (spin + seam + pitch), which is how broadcast "turn/deviation" works.

---

## 8. Phase 6 — Run the pipeline end to end

```python
# FILE: src/run_pipeline.py
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
```

```bash
export PYTHONPATH=src
python src/run_pipeline.py                 # first run (detects stumps, calibrates, fits, renders)
python src/run_pipeline.py --mc 20         # adds an uncertainty estimate (takes longer)
python src/run_pipeline.py --redo-stumps   # re-run stump detection after changing frames/conf
```

Before your real ball labels exist, Phase 8's synthetic test already exercises the whole math chain.

---

## 9. Phase 7 — FullTrack-style visuals

Two outputs: (1) `overlay.mp4`: your clip with the pitch, stumps and the reconstructed path projected back onto the video (a direct visual check of the calibration); (2) `virtual_view.png`: a virtual camera behind the bowler's end, higher up, like the FullTrack render, with Swing / Spin / Speed cards.

```python
# FILE: src/render.py
import cv2, numpy as np
from geometry import *

RED = (40, 40, 220); BLUE = (230, 130, 150); WHITE = (255, 255, 255); YEL = (60, 210, 240)


def _pts(P, rvec, C, f, size):
    return np.round(project(P, rvec, C, f, size)).astype(np.int32)


def _pitch_geometry(img, rvec, C, f, size, fill=None, bounce_y=None):
    hw = PITCH_HW
    poly = np.array([[-hw, 0, 0], [hw, 0, 0], [hw, PITCH_L, 0], [-hw, PITCH_L, 0]])
    if fill is not None:
        ov = img.copy(); cv2.fillPoly(ov, [_pts(poly, rvec, C, f, size)], fill)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    cv2.polylines(img, [_pts(poly, rvec, C, f, size)], True, WHITE, 1, cv2.LINE_AA)
    for y in (0, POP, PITCH_L - POP, PITCH_L):                       # bowling/popping creases
        cv2.polylines(img, [_pts(np.array([[-RET, y, 0], [RET, y, 0]]), rvec, C, f, size)], False, WHITE, 1, cv2.LINE_AA)
    for Y in (0.0, PITCH_L):
        for x in STUMP_X:
            cv2.line(img, *map(tuple, _pts(np.array([[x, Y, 0], [x, Y, STUMP_H]]), rvec, C, f, size)), YEL, 2, cv2.LINE_AA)
    if bounce_y is not None:                                         # blue line bounce -> stumps
        strip = np.array([[-.15, bounce_y, 0], [.15, bounce_y, 0], [.15, PITCH_L, 0], [-.15, PITCH_L, 0]])
        ov = img.copy(); cv2.fillPoly(ov, [_pts(strip, rvec, C, f, size)], BLUE)
        cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)


def _card(img, x, y, label, value, w=150, h=52):
    ov = img.copy(); cv2.rectangle(ov, (x, y), (x + w, y + h), (30, 40, 30), -1)
    cv2.addWeighted(ov, 0.7, img, 0.3, 0, img); cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(img, label, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, value, (x + 8, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)


def _cards(img, m, scale=1.0, sf_scale=None):
    sw = m.get("swing_dev_cm", 0.0) * (sf_scale if sf_scale else 1.0)
    unit = "sf" if sf_scale else "cm"
    _card(img, 12, 12, "Swing", f"{sw:.1f} {unit}")
    _card(img, 12, 72, "Spin", f"{m.get('spin_deg', 0.0):.1f} deg")
    _card(img, 12, 132, "Speed", f"{m['speed_mph']:.0f} mph")


def render_overlay_video(video, cal, fit, metrics, info, fps, out_path, sf_scale=None):
    cap = cv2.VideoCapture(video); size = tuple(cal["size"])
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    f0 = int(fit["frames"][0]); T, P = info["t"], info["pos"]; i = 0
    by = metrics["bounce_xy_m"][1] if "bounce_xy_m" in metrics else None
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i <= int(fit["frames"][-1]) + 3:
            _pitch_geometry(fr, cal["rvec"], cal["C"], cal["f"], size, bounce_y=by)
            if i >= f0:
                upto = (i - f0) / fps; sel = T <= upto
                if sel.sum() > 1:
                    cv2.polylines(fr, [_pts(P[sel], cal["rvec"], cal["C"], cal["f"], size)], False, RED, 2, cv2.LINE_AA)
            _cards(fr, metrics, sf_scale=sf_scale)
        vw.write(fr); i += 1
    vw.release()


def render_virtual_view(cal, fit, metrics, info, out_png, sf_scale=None, size=(540, 960)):
    img = np.full((size[1], size[0], 3), (70, 120, 60), np.uint8)
    C = np.array([0.0, -6.0, 5.0]); rvec = lookat_rvec(C, [0, 11.0, 0.0]); f = 850.0
    by = metrics["bounce_xy_m"][1] if "bounce_xy_m" in metrics else None
    _pitch_geometry(img, rvec, C, f, size, fill=(150, 175, 185), bounce_y=by)
    cv2.polylines(img, [_pts(info["pos"], rvec, C, f, size)], False, RED, 3, cv2.LINE_AA)
    _cards(img, metrics, sf_scale=sf_scale)
    cv2.imwrite(out_png, img)
```

Open `outputs/overlay.mp4`: the white pitch outline and yellow stump lines must lie on the real pitch and stumps, and the red path must follow the ball. Misalignment at the far end means calibration depth is off, so go back to Phase 3.

---

## 10. Phase 8 — Validation

### 8a. Synthetic test (run this first, no real data needed)

Builds a known camera and a known delivery, projects it with 1 px noise, then checks that calibration and trajectory fitting recover the truth.

```python
# FILE: tests/test_synthetic.py
import sys; sys.path.insert(0, "src")
import numpy as np
from geometry import *
from calibrate import calibrate
from trajectory import simulate, fit_trajectory
from metrics import compute_metrics

SIZE = [360, 640]; FPS = 30
TRUE_C = np.array([0.15, -4.3, 1.4]); TRUE_F = 520.0
TRUE_RV = lookat_rvec(TRUE_C, [0, 12, 0.2])
PRIOR = {"mu": [0.0, -4.0, 1.3], "sigma": [0.3, 0.3, 0.15], "weight": 1.0}


def test_calibration_recovers_camera():
    rng = np.random.default_rng(1)
    W = np.vstack([stump_world_points(0.0), stump_world_points(PITCH_L)])
    P = project(W, TRUE_RV, TRUE_C, TRUE_F, SIZE) + rng.normal(0, 0.7, (len(W), 2))
    cal = calibrate(W, P, SIZE, PRIOR)
    print("calib:", cal["C"], cal["f"], cal["rmse_px"])
    assert cal["rmse_px"] < 2.0
    assert np.linalg.norm(np.array(cal["C"]) - TRUE_C) < 0.4
    assert abs(cal["f"] / TRUE_F - 1) < 0.12


def test_trajectory_recovers_metrics():
    rng = np.random.default_rng(2)
    cal = {"rvec": TRUE_RV.tolist(), "C": TRUE_C.tolist(), "f": TRUE_F, "size": SIZE}
    true = np.array([0.10, 0.6, 2.2, 0.35, 22.6, -0.5, 0.8, 0.55, 0.85, np.radians(3.0)])
    frames = np.arange(0, 22); ts = frames / FPS
    P, _ = simulate(true, ts, FPS)
    obs = project(P, TRUE_RV, TRUE_C, TRUE_F, SIZE) + rng.normal(0, 1.0, (len(ts), 2))
    fit = fit_trajectory(frames, obs, cal, FPS, verbose=False)
    m_true, _ = compute_metrics(true, ts, FPS); m_fit, _ = compute_metrics(fit["theta"], ts, FPS)
    print("true:", {k: round(v, 2) for k, v in m_true.items() if isinstance(v, float)})
    print("fit :", {k: round(v, 2) for k, v in m_fit.items() if isinstance(v, float)})
    assert abs(m_fit["speed_mph"] / m_true["speed_mph"] - 1) < 0.06
    assert abs(m_fit["length_m"] - m_true["length_m"]) < 0.6
    assert abs(m_fit["spin_deg"] - m_true["spin_deg"]) < 3.0
```

```bash
export PYTHONPATH=src
pytest -q -s tests/test_synthetic.py
```

### 8b. The real clip against ground truth

Compare `outputs/result.json` with the app's replay values (52 mph, 0.2 sf, 3.6°):

- **Speed** should land within roughly ±5–10% if calibration and labels are good. A bigger gap usually means the camera prior is wrong (distance to the stumps scales speed directly) or labels are off by a frame.
- **Spin** (3.6°) is small, so noise matters; expect ±3° at 30 fps.
- **Swing in "sf"**: one clip gives you one point. Collect **10+ FullTrack clips** (each shows its numbers in the replay), run them all, then fit `sf = k · swing_dev_cm` by regression and put `k` into `render` via `sf_scale`.

Batch script for many clips (one folder per clip with `video.mp4`, `ball.csv`, `gt.json`):

```bash
for d in data/clips/*/; do
  sed -e "s#^video:.*#video: ${d}video.mp4#" -e "s#^ball_csv:.*#ball_csv: ${d}ball.csv#" config.yaml > /tmp/cfg.yaml
  python src/run_pipeline.py --config /tmp/cfg.yaml --redo-stumps --no-render
  cp outputs/result.json "${d}result.json"
done
```

### 8c. Quality gates (do not trust a result unless all pass)

| Check | Pass condition |
|---|---|
| Stump check image | Lines sit on stumps, near and far set |
| Calibration RMSE | < 2–3 px |
| Camera position | Within ~0.5 m of FullTrack setup (0, −4, 1.3) |
| Overlay video | Pitch/stumps line up across the whole clip |
| Ball fit RMS | < ~3 px (label noise dominates) |
| Bounce | The fitted bounce frame matches the visible kink in the ball path |
| Monte-Carlo (`--mc 20`) | Speed std < ~5%, length std < ~0.7 m |

---

## 11. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `Model not found` / load error | Re-create `models/stumpbest.pt` (Phase 0). Update ultralytics: `pip install -U ultralytics`. |
| Fewer than 3 stumps per set | Bowler occludes or `stump_conf` is too high → lower to 0.15, widen `frames`, or write `stumps.json` by hand. |
| Calibration RMSE high | Box bottoms include the stump base, or near/far sets are swapped → inspect `stump_check.png`. |
| Camera drifts far from the prior | Data and prior disagree; verify stump pixels first. If the setup really differs, change `camera_prior.mu`. |
| Speed far too high/low | The camera distance prior is wrong (speed scales with it) or the first label is not near release. |
| Fit fails / no feasible start | Labels not in 360×640 pixel space, wrong `ball_frame_offset`, or fewer than ~8 points. |
| Bounce at a nonsense location | The bounce frame was not labelled, or false ball detections exist. Remove outlier labels. |
| Trajectory looks right in 3D but not in overlay | Frame offset between labels and video. Try `ball_frame_offset: -1` or `1`. |

---

## 12. Accuracy expectations and next improvements

Realistic with this 360×640 / 30 fps source: speed ±5–10%, length ±0.5–1 m, spin/swing noisy. Improvements in order of value:

1. **Higher-quality source video** (original 1080p/60 fps instead of the screen recording) → biggest gain; ball pixels and temporal resolution both improve.
2. **Hand-click 4–6 extra pitch points** (popping/return crease corners at both ends) into `extra_points` → better focal length and depth.
3. **Label the ball from release to past the bounce** (not just near the end).
4. **Several clips** to fit the swing "sf" and spin scales against FullTrack's numbers.
5. Later: per-frame ball detection with your YOLO + Kalman pipeline to replace manual labels, temperature/ball-type drag tuning, and spin-axis estimation.

---

## 13. Quick command summary

```bash
mkdir cricket3d && cd cricket3d && python -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python numpy scipy pandas matplotlib pyyaml tqdm pytest
mkdir -p data/raw data/interim data/labels models src tools tests outputs
# copy video, stumpbest.pt, this guide, then run the extractor in Phase 0
export PYTHONPATH=src
pytest -q -s tests/test_synthetic.py            # math sanity
python src/detect_stumps.py                      # check outputs/stump_check.png
python src/calibrate.py                          # check rmse and camera position
python src/run_pipeline.py --mc 20               # metrics + overlay.mp4 + virtual_view.png
```
