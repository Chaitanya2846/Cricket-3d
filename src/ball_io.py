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
