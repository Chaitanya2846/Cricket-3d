import re, glob, os
import numpy as np, pandas as pd


def load_ball(path, size, offset=0, auto_trim_release=True):
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

    frames = a[:, 0].astype(int) + offset
    obs = a[:, 1:3]

    if auto_trim_release and len(obs) > 6:
        # Detect true release point: ball moves downwards (y increasing in image space)
        # Skip initial frames where arm is swinging upwards (dy < 0) or ball is still at apex in fingers
        start_idx = 0
        for i in range(min(6, len(obs) - 4)):
            if obs[i + 1, 1] < obs[i, 1]:  # ball moving upwards
                start_idx = i + 1
        # If candidate frame is top of delivery swing before ball separates into free flight
        if start_idx < len(obs) - 4:
            dy0 = obs[start_idx + 1, 1] - obs[start_idx, 1]
            if dy0 < 10.0:  # minimal descent, bowler still clasping ball at top of swing
                start_idx += 1
        frames = frames[start_idx:]
        obs = obs[start_idx:]

    return frames, obs

