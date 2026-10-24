# Cricket3D: Monocular 3D Cricket Delivery Tracking & Virtual Hawkeye Replay

An end-to-end computer vision and trajectory physics system for single-camera cricket delivery analysis. Given a broadcast or phone recording taken from behind the bowler's stumps, Cricket3D:
1. **Calibrates Camera 3D Extrinsics & Intrinsics** using regulation cricket wicket geometry.
2. **Estimates 3D Trajectory & Aerodynamics** (release velocity, flight arc, aerodynamic drag, pitch bounce, rebound, spin, and swing deviation) using numerical ODE physics fitting.
3. **Renders Broadcast Overlays & 3D Virtual Hawk-Eye Replay** with smooth multicamera orbits (Front view -> Parallel close-up side view -> Batter's end view).
4. **Generates Broadcast Session Analytics Images**: Aggregates deliveries into three standalone visual analytics cards:
   - **`pitch_map.png`**: Top-down broadcast pitch map with 5 color-coded length zones, lateral zone labels, right-hand distance capsules, and both sets of wickets with official ICC crease markings.
   - **`beehive.png`**: Batter-end arrival view with 4 color-coded arrival zones, percentage distribution, wooden stumps with bails, and $1\text{M} / 2\text{M}$ height markers.
   - **`release_points.png`**: Original calibrated tripod camera view with a 3D measurement frame rising directly from the bowler's no-ball crease ($Y = 1.22\text{ m}$), yellow average release height line ($2.19\text{ m}$), exactly 2 horizontal lines, and pitch removed behind stumps.

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
| `--session-analytics` | Generate the 3 session analytics images (`pitch_map.png`, `beehive.png`, `release_points.png`) | `False` |

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

### Method C: Run Batch Evaluation Across Any Session of Videos

To process and evaluate deliveries in batch mode across any folder of video clips with automatic YOLO stump detection, grand metrics summary table, and automatic session analytics image generation:

```bash
# Process default clips in data/clips/
python tools/batch_evaluate.py

# Process custom clips directory or specific clips
python tools/batch_evaluate.py --clips-dir path/to/my_clips/ --out-dir my_outputs/
```

#### Batch CLI Options:
| Flag | Description | Default |
|---|---|---|
| `--clips-dir <dir>` | Directory containing clip subfolders (`clip_1`, `clip_2`, etc.) | `data/clips` |
| `--clips <names>` | Specific clip subfolder names to process | All subfolders |
| `--out-dir <dir>` | Directory to save `batch_summary.json` and session images | `outputs` |
| `--redo-stumps` | Force automatic YOLO stump detection across all clips | `False` |

---

### Method D: Generate Session Analytics Cards Directly

To generate the three standalone session images from any existing session summary, single delivery result, directory of clips, or glob pattern:

```bash
# From session summary JSON (default):
python src/session_analytics.py --source outputs/batch_summary.json --out-dir outputs

# From a directory containing clips/results:
python src/session_analytics.py --source data/clips --out-dir outputs

# From a single delivery result:
python src/session_analytics.py --source outputs/result.json --out-dir outputs

# Using glob pattern across multiple results:
python src/session_analytics.py --source "data/clips/*/outputs/result.json" --out-dir outputs
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
- **`pitch_map.png`**: Standalone broadcast pitch map image:
  - Gentle $1.71:1$ broadcast perspective with authentic clay turf and turf roller lines.
  - Color-coded length zones: Full Toss (grey), Yorker (gold), Full (green), Good Length (coral/red), Short (slate purple).
  - Left-flanking zone percentage labels and right-flanking distance capsules (`0.0m`, `2.0m`, `5.0m`, `8.0m`).
  - Both wicket ends with official ICC Law 7 & 8 bowling, popping, return creases, and wide guidelines.
- **`beehive.png`**: Standalone batter-end wicket arrival image:
  - Continuous pitch turf receding towards the foreground off-canvas.
  - 4 vertical arrival columns with top-edge percentage labels.
  - 3D wooden wickets with bails, realistic contact shadows, and $1\text{M} / 2\text{M}$ height markers.
- **`release_points.png`**: Standalone bowler release points image:
  - Calibrated tripod perspective view matching original video capture.
  - 3D measurement frame rising directly from the bowler's no-ball crease ($Y = 1.22\text{ m}$).
  - Yellow dashed line at average release height slicing through plotted release coordinates.
  - Exactly 2 horizontal ground lines (bowling crease at stumps $Y = 0.0\text{ m}$, popping crease at $Y = 1.22\text{ m}$).
  - Pitch turf and central blue corridor cleanly end at the stumps with turf removed behind.

---

## 🧪 Testing

To run the synthetic physics and trajectory math test suite:

```bash
pytest -q -s tests/test_synthetic.py
```
