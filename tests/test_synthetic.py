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
    assert abs(m_fit["spin_deg"] - m_true["spin_deg"]) < 4.0
