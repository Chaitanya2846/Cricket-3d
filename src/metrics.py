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
