import numpy as np
from scipy.optimize import least_squares
from geometry import *

KD = 0.0061   # quadratic drag a = -KD*|v|*v   (Cd~0.4, r=35.6 mm, m=156 g)
NAMES = ["x0", "y0", "z0", "vx", "vy", "vz", "ax", "e", "kh", "delta"]
# Physics bounds: legal bowling crease zone y0 in [0.20, 1.20] m, release height z0 in [1.70, 2.45] m
LO = np.array([-1.8,  0.20, 1.70, -7.0, 14.0, -15.0, -6.0, 0.25, 0.40, np.radians(-12)])
HI = np.array([ 1.8,  1.20, 2.45,  7.0, 48.0,  10.0,  6.0, 0.80, 1.00, np.radians(12)])
XS = np.array([ 0.2,  0.20, 0.15,  1.0,  2.0,   1.0,  1.0,  0.1,  0.1, np.radians(2)])



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


def _res(theta, ts, obs, cal, fps, sig, crease_prior=(0.65, 2.10), w_crease=3.5):
    P, _ = simulate(theta, ts, fps, dt_factor=6)
    pix = project(P, cal["rvec"], cal["C"], cal["f"], cal["size"])
    r_pix = ((pix - obs) / sig).ravel()
    if w_crease > 0 and crease_prior is not None:
        y0_ref, z0_ref = crease_prior
        r_prior = np.array([
            (theta[1] - y0_ref) / 0.18 * w_crease,
            (theta[2] - z0_ref) / 0.18 * w_crease
        ])
        return np.concatenate([r_pix, r_prior])
    return r_pix


def _init(k, ts, obs, cal, z0=2.10, y0_ref=0.65):
    P0 = backproject_to_plane(obs[0], cal["rvec"], cal["C"], cal["f"], cal["size"], z0)
    B = backproject_to_plane(obs[k], cal["rvec"], cal["C"], cal["f"], cal["size"], BALL_R)
    tb = ts[k]
    dy = max(4.0, B[1] - y0_ref)
    vy = dy / tb * (1.0 + 0.5 * KD * dy)
    vz = (BALL_R - z0 + 0.5 * G * tb ** 2) / tb
    vx = (B[0] - P0[0]) / tb
    return np.array([P0[0], y0_ref, z0, vx, vy, vz, 0.0, 0.65, 0.85, 0.0])


def fit_trajectory(frames, obs, cal, fps, sig=2.0, verbose=True, candidates=None,
                   crease_prior=(0.65, 2.10), w_crease=3.5):
    frames = np.asarray(frames); obs = np.asarray(obs, float); ts = (frames - frames[0]) / fps
    best = None
    for k in (candidates if candidates is not None else range(2, len(frames) - 1)):
        th0 = _init(k, ts, obs, cal, z0=crease_prior[1] if crease_prior else 2.10,
                    y0_ref=crease_prior[0] if crease_prior else 0.65)
        if not (10 < th0[4] < 50): continue
        th0 = np.clip(th0, LO + 1e-4, HI - 1e-4)
        sol = least_squares(_res, th0, bounds=(LO, HI), x_scale=XS, max_nfev=150,
                            args=(ts, obs, cal, fps, sig, crease_prior, w_crease))
        if verbose: print(f"  bounce candidate frame {frames[k]:4d}: cost={sol.cost:9.2f}")
        if best is None or sol.cost < best[0].cost: best = (sol, k)
    if best is None: raise SystemExit("No feasible start: check ball labels / calibration.")
    sol, k = best
    P_sol, _ = simulate(sol.x, ts, fps, dt_factor=6)
    pix_sol = project(P_sol, cal["rvec"], cal["C"], cal["f"], cal["size"])
    rms = float(np.sqrt(np.mean((pix_sol - obs) ** 2)))  # pure pixel reprojection RMS
    return {"theta": sol.x, "ts": ts, "frames": frames, "bounce_candidate": int(frames[k]),
            "cost": float(sol.cost), "rms_px": rms}

