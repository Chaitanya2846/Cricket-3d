# Cricket3D: Monocular 3D Cricket Delivery Tracking & Virtual Hawkeye Replay

An end-to-end computer vision and trajectory physics system for single-camera cricket delivery analysis. Given a broadcast or phone recording taken from behind the bowler's stumps, Cricket3D:
1. **Calibrates Camera 3D Extrinsics & Intrinsics** using regulation cricket wicket geometry.
2. **Estimates 3D Trajectory & Aerodynamics** (release velocity, flight arc, aerodynamic drag, pitch bounce, rebound, spin, and swing deviation) using numerical ODE physics fitting.
3. **Renders Broadcast Overlays & 3D Virtual Hawk-Eye Replay** with smooth multicamera orbits (Front view -> Parallel close-up side view -> Batter's end view).

---

## 🚀 Quick Start (Setup)

### 1. Clone & Set Up Environment

```bash
git clone <your-repo-url>
cd cricket3d

# Create and activate virtual environment (Python 3.10 or 3.11 recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on FFmpeg**: Video transcoding into WhatsApp-compatible MP4 format uses FFmpeg. Ensure `ffmpeg` is available on your PATH or in `C:\ffmpeg\bin\ffmpeg.exe`.

---

## 🎯 How to Process a Delivery (Which File to Run)

### Method A: Run on Any Custom Video via CLI (Quickest)

You can run the pipeline directly on any new video and labeled ball coordinates:

```bash
python src/run_pipeline.py --video path/to/delivery.mp4 --ball path/to/ball.csv --out-dir outputs
```

#### CLI Options:
| Flag | Description | Default |
|---|---|---|
| `--video <path>` | Path to the bowling delivery video (`.mp4`) | `config.yaml` |
| `--ball <path>` | Path to ball tracking CSV (or YOLO txt folder) | `config.yaml` |
| `--out-dir <dir>` | Directory to save rendered outputs and metrics | `outputs` |
| `--fps <float>` | Frame rate (automatically detected from video if omitted) | Auto-detected |
| `--stumps <path>` | Path to precomputed `stumps.json` (skips re-detection) | `None` |
| `--redo-stumps` | Force re-running YOLO stump detection model | `False` |
| `--no-render` | Run physics fitting & metrics only (skip MP4 rendering) | `False` |

---

### Method B: Run via Configuration File (`config.yaml`)

Edit `config.yaml` with your video and parameters:

```yaml
video: data/raw/fulltrack_clip_1.mp4
model: models/stumpbest.pt
fps: 60
size: [720, 1280]            # [width, height]
frames: [0, 140]            # frame search range for stumps
ball_csv: data/labels/ball.csv
ball_frame_offset: 0
stump_conf: 0.15
```

Then run:

```bash
python src/run_pipeline.py --config config.yaml
```

---

### Method C: Run Batch Evaluation Across Multiple Clips

To process and evaluate all deliveries in batch mode with a formatted metrics summary table:

```bash
python tools/batch_evaluate.py
```

---

## 📂 Input Data Specifications

To process a new cricket delivery, the system requires:

1. **Video File (`.mp4`)**:
   - Camera positioned behind the bowler's wickets looking down the pitch towards the batter's wickets.
   - Both near and far sets of wickets should be visible in early frames for automated 3D calibration.

2. **Ball Coordinates (`ball.csv`)**:
   - A CSV file containing 2D ball detections in the video coordinate space:
     ```csv
     frame,x,y
     66,201.7,366.3
     67,212.8,370.9
     68,221.6,375.1
     ...
     ```
   - Accepted column headers: `frame, x, y` (or `cx, cy` / `u, v`). Pixel values or normalized 0-1 values are auto-detected.
   - Alternatively, a folder of YOLO `.txt` detection files (`class cx cy w h`).

3. **Stump Detector (`models/stumpbest.pt`)**:
   - Pre-trained YOLO11 checkpoint included in `models/stumpbest.pt` that detects near and far stumps automatically.

---

## 📊 Generated Outputs

All outputs are saved to the designated `--out-dir` (default: `outputs/`):

- **`overlay.mp4`**: High-definition video combining:
  1. **Section 1**: Real delivery footage with synchronized ball arc, white ball graphic, blue ground delivery guide, and delayed stats HUD (speed, spin, swing).
  2. **Section 2**: 3D Hawk-Eye virtual replay orbiting from Front view -> Close-up parallel Side view -> Batter's end view with regulation crease markings.
- **`virtual_view.png`**: High-resolution standalone frame of the 3D Hawkeye virtual replay.
- **`result.json`**: Computed delivery physics metrics:
  - `speed_mph` & `speed_kph` (release velocity)
  - `length_m` & `length_class` (Full, Good Length, Short)
  - `spin_deg` (deviation angle off pitch)
  - `swing_dev_cm` (aerodynamic trajectory curvature)
  - `bounce_xy_m` (pitch bounce coordinates)
- **`calibration.json`**: Recovered camera 3D position ($C$), rotation vector ($rvec$), and focal length ($f$).
- **`stump_check.png`**: Visual verification overlay of detected near and far wicket boxes.

---

## 🧪 Testing

To run the synthetic physics and trajectory math test suite:

```bash
pytest -q -s tests/test_synthetic.py
```
