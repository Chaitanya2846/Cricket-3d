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
