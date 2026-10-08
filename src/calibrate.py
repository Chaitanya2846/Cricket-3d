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
