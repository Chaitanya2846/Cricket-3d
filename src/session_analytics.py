import os
import sys
import json
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Add project root and src to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.trajectory import simulate

IMAGE_W = 720
IMAGE_H = 1050

# Pitch & Crease Regulations (Meters per ICC/MCC Law 7 & 8)
PITCH_L = 20.12     # 22 yards stump-to-stump
PITCH_W = 3.05      # 10 feet
PITCH_HW = 1.525
POP = 1.22          # Popping crease distance from stumps (4 ft)
RET = 1.32          # Return crease distance from pitch center (4 ft 4 in, 2.64m total width)
WIDE = 0.89         # Wide guidelines (35 inches from middle stump)
STUMP_H = 0.711     # 28 inches
STUMP_X = [-0.1143, 0.0, 0.1143]  # 3 stumps across 9 inches

# Authentic colors matching overlay.mp4 / render.py
BLUE = (235, 140, 120)        # Translucent pitch corridor in BGR
WOOD = (65, 185, 230)         # Timber wicket tone in BGR
DARK_WOOD = (30, 40, 50)      # Stump crown cap in BGR
PITCH_COLOR = (168, 182, 184) # Clay pitch surface in BGR
ROLLER_LINE = (158, 172, 174) # Pitch roller lines in BGR

BALL_RHB = (255, 255, 255)
BALL_LHB = (235, 55, 55)
BALL_UNKNOWN = (50, 50, 50)
COLOR_NAVY_PILL = (15, 38, 70)

def get_font(size, bold=False):
    """Loads system TTF font with fallback."""
    names = ["segoeuib.ttf" if bold else "segoeui.ttf",
             "arialbd.ttf" if bold else "arial.ttf",
             "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    return ImageFont.load_default()

def lookat_rvec(C, target, up=np.array([0.0, 0.0, 1.0])):
    """Computes OpenCV Rodrigues rotation vector for camera C looking at target."""
    fwd = target - C
    norm = np.linalg.norm(fwd)
    if norm < 1e-8:
        fwd = np.array([0.0, 1.0, 0.0])
    else:
        fwd /= norm

    right = np.cross(fwd, up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= r_norm

    real_up = np.cross(right, fwd)
    R = np.vstack([right, -real_up, fwd])
    rvec, _ = cv2.Rodrigues(R)
    return rvec

def project_point(P, rvec, C, f, size, near_z=0.15):
    """Projects 3D point P into pixel coordinates (u, v)."""
    P = np.asarray(P, float).reshape(1, 1, 3)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    tvec = -R @ np.asarray(C, float).reshape(3, 1)
    w, h = size
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], float)
    pts_cam = (R @ P.reshape(3, 1) + tvec).flatten()
    if pts_cam[2] < near_z:
        return None
    pts_2d, _ = cv2.projectPoints(P, rvec, tvec, K, None)
    return pts_2d.reshape(2)

def project_poly(pts, rvec, C, f, size, near_z=0.15):
    """Projects 3D polygon with near-plane clipping into 2D pixel coordinates."""
    pts = np.asarray(pts, float)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    Xc = (pts - np.asarray(C, float)) @ R.T
    out = []
    n = len(Xc)
    for i in range(n):
        curr_p = Xc[i]
        prev_p = Xc[(i - 1 + n) % n]
        curr_in = curr_p[2] >= near_z
        prev_in = prev_p[2] >= near_z
        if curr_in:
            if not prev_in:
                alpha = (near_z - prev_p[2]) / (curr_p[2] - prev_p[2] + 1e-12)
                out.append(prev_p + alpha * (curr_p - prev_p))
            out.append(curr_p)
        elif prev_in:
            alpha = (near_z - prev_p[2]) / (curr_p[2] - prev_p[2] + 1e-12)
            out.append(prev_p + alpha * (curr_p - prev_p))
    if len(out) < 3:
        return None
    out = np.array(out)
    u = f * out[:, 0] / out[:, 2] + size[0] / 2.0
    v = f * out[:, 1] / out[:, 2] + size[1] / 2.0
    return np.stack([u, v], 1).round().astype(np.int32)

def project_segments(pts_3d, rvec, C, f, size, near_z=0.15):
    """Projects 3D line segments with clipping (matching render.py)."""
    pts = np.asarray(pts_3d, float)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    Xc = (pts - np.asarray(C, float)) @ R.T
    segments = []
    for i in range(len(Xc) - 1):
        p0 = Xc[i].copy()
        p1 = Xc[i+1].copy()
        z0, z1 = p0[2], p1[2]
        if z0 < near_z and z1 < near_z:
            continue
        if z0 < near_z:
            alpha = (near_z - z0) / (z1 - z0 + 1e-12)
            p0 = p0 + alpha * (p1 - p0)
        elif z1 < near_z:
            alpha = (near_z - z1) / (z0 - z1 + 1e-12)
            p1 = p1 + alpha * (p0 - p1)
        u0 = f * p0[0] / p0[2] + size[0] / 2.0
        v0 = f * p0[1] / p0[2] + size[1] / 2.0
        u1 = f * p1[0] / p1[2] + size[0] / 2.0
        v1 = f * p1[1] / p1[2] + size[1] / 2.0
        segments.append([(int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1)))])
    return segments

def load_session_deliveries(source="outputs/batch_summary.json"):
    """Loads deliveries for the session."""
    data = None
    if isinstance(source, str) and os.path.exists(source):
        try:
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {source}: {e}")
            return []
    elif isinstance(source, list):
        data = source

    if not data:
        return []

    deliveries = []
    for item in data:
        cid = item.get("clip_id", "unknown")
        metrics = item.get("metrics", {})
        theta_dict = item.get("theta", {})
        if isinstance(theta_dict, dict) and "x0" in theta_dict:
            theta = [theta_dict["x0"], theta_dict["y0"], theta_dict["z0"],
                     theta_dict["vx"], theta_dict["vy"], theta_dict["vz"],
                     theta_dict["ax"], theta_dict["e"], theta_dict["kh"], theta_dict["delta"]]
        elif isinstance(theta_dict, list):
            theta = theta_dict
        else:
            continue

        release_pos = np.array([theta[0], theta[1], theta[2]], float)

        # Bounce position
        if "bounce_xy_m" in metrics:
            bx, by = metrics["bounce_xy_m"]
            bounce_pos = np.array([bx, by, 0.0356], float)
        else:
            bounce_pos = None

        # Simulate trajectory for arrival at stumps
        ts = np.linspace(0, 1.5, 300)
        pos_sim, info = simulate(theta, ts, fps=60, dt_factor=10)
        P = info["pos"]

        # Stumps arrival (Y = 20.12)
        idx_stump = np.where(P[:, 1] >= PITCH_L)[0]
        if len(idx_stump) > 0:
            i = idx_stump[0]
            frac = (PITCH_L - P[i-1, 1]) / (P[i, 1] - P[i-1, 1] + 1e-12)
            stump_arrival = P[i-1] + frac * (P[i] - P[i-1])
        else:
            stump_arrival = P[-1].copy()

        # Batter handedness: check gt.json or default to RHB
        batter_hand = item.get("batter_hand")
        if not batter_hand:
            gt_path = f"data/clips/{cid}/gt.json"
            if os.path.exists(gt_path):
                try:
                    gt_data = json.load(open(gt_path, "r", encoding="utf-8"))
                    batter_hand = gt_data.get("batter_hand", "RHB")
                except Exception:
                    batter_hand = "RHB"
            else:
                batter_hand = "RHB"

        length_m = float(PITCH_L - bounce_pos[1]) if bounce_pos is not None else 0.0
        deliveries.append({
            "clip_id": cid,
            "theta": theta,
            "release_pos": release_pos,
            "bounce_pos": bounce_pos,
            "stump_arrival": stump_arrival,
            "length_m": length_m,
            "batter_hand": batter_hand,
            "metrics": metrics
        })

    return deliveries

def draw_double_arrow(d, cx, cy, col=(40, 45, 50)):
    """Draws small up/down vertical arrows."""
    d.polygon([(cx, cy - 6), (cx - 4, cy - 1), (cx + 4, cy - 1)], fill=col)
    d.polygon([(cx, cy + 6), (cx - 4, cy + 1), (cx + 4, cy + 1)], fill=col)

def make_turf_background(size=(IMAGE_W, IMAGE_H)):
    """Matches the exact spotlight turf background from overlay.mp4 / render.py."""
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - w // 2)**2 + (Y - int(h * 0.53))**2)
    spotlight = np.exp(-0.5 * (dist / 480.0)**2)

    base_b = 82 - 15 * (Y / float(h))
    base_g = 142 - 20 * (Y / float(h))
    base_r = 72 - 15 * (Y / float(h))

    b = np.clip(base_b + 20 * spotlight, 0, 255).astype(np.uint8)
    g = np.clip(base_g + 26 * spotlight, 0, 255).astype(np.uint8)
    r = np.clip(base_r + 18 * spotlight, 0, 255).astype(np.uint8)

    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

def draw_overlay_pitch_turf(img, poly_px, rv, C, f, size, hw=PITCH_HW, y_start=0.0, y_end=PITCH_L, with_corridor=True):
    """Draws the exact pitch turf and roller lines from overlay.mp4 / render.py."""
    # 1. Base pitch turf
    cv2.fillPoly(img, [poly_px], PITCH_COLOR)
    cv2.polylines(img, [poly_px], True, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. Subtle turf roller lines across pitch
    for y_strip in np.arange(y_start + 0.5, y_end, 1.5):
        seg = project_segments(np.array([[-hw + 0.05, y_strip, 0.001], [hw - 0.05, y_strip, 0.001]]), rv, C, f, size)
        for p0, p1 in seg:
            cv2.line(img, p0, p1, ROLLER_LINE, 1, cv2.LINE_AA)

    # 3. Translucent central delivery guide corridor
    if with_corridor:
        strip = np.array([[-0.18, y_start, 0.002], [0.18, y_start, 0.002], [0.18, y_end, 0.002], [-0.18, y_end, 0.002]])
        s_px = project_poly(strip, rv, C, f, size)
        if s_px is not None:
            ov = img.copy()
            cv2.fillPoly(ov, [s_px], BLUE)
            cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)

def draw_professional_creases(img, rv, C, f, size, ends=('bowler', 'batter'), color=(255, 255, 255), thickness=2, bowler_behind=0.0):
    """
    Draws authentic ICC/MCC professional cricket crease markings:
    1. Bowling Creases (2.64m wide, through the stumps)
    2. Popping Creases (3.66m wide, 1.22m in front of stumps)
    3. Return Creases (perpendicular at X=±1.32m)
    4. Wide Guidelines (at X=±0.89m between popping and bowling creases)
    """
    if 'bowler' in ends:
        # Bowling crease: through stumps at Y=0.0 (X in [-1.32, 1.32])
        for p0, p1 in project_segments(np.array([[-RET, 0.0, 0.003], [RET, 0.0, 0.003]]), rv, C, f, size):
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
        # Popping crease: 1.22m in front at Y=POP (X in [-1.83, 1.83])
        for p0, p1 in project_segments(np.array([[-1.83, POP, 0.003], [1.83, POP, 0.003]]), rv, C, f, size):
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
        # Return creases: perpendicular at X=±RET, from Y=-bowler_behind to Y=POP
        for sign in [-1, 1]:
            for p0, p1 in project_segments(np.array([[sign * RET, -bowler_behind, 0.003], [sign * RET, POP, 0.003]]), rv, C, f, size):
                cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
            # Wide guidelines: between bowling and popping creases
            for p0, p1 in project_segments(np.array([[sign * WIDE, 0.0, 0.003], [sign * WIDE, POP, 0.003]]), rv, C, f, size):
                cv2.line(img, p0, p1, color, 1, cv2.LINE_AA)

    if 'batter' in ends:
        # Bowling crease: through stumps at Y=PITCH_L (X in [-1.32, 1.32])
        for p0, p1 in project_segments(np.array([[-RET, PITCH_L, 0.003], [RET, PITCH_L, 0.003]]), rv, C, f, size):
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
        # Popping crease: 1.22m in front at Y=PITCH_L - POP (X in [-1.83, 1.83])
        for p0, p1 in project_segments(np.array([[-1.83, PITCH_L - POP, 0.003], [1.83, PITCH_L - POP, 0.003]]), rv, C, f, size):
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
        # Return creases: perpendicular at X=±RET, from Y=PITCH_L - POP to Y=PITCH_L + 1.22
        for sign in [-1, 1]:
            for p0, p1 in project_segments(np.array([[sign * RET, PITCH_L - POP, 0.003], [sign * RET, PITCH_L + 1.22, 0.003]]), rv, C, f, size):
                cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
            # Wide guidelines: between popping and bowling creases
            for p0, p1 in project_segments(np.array([[sign * WIDE, PITCH_L - POP, 0.003], [sign * WIDE, PITCH_L, 0.003]]), rv, C, f, size):
                cv2.line(img, p0, p1, color, 1, cv2.LINE_AA)

def draw_3d_stumps(card, y_s, rv, C, f, size, stump_h=STUMP_H, col_bgr=WOOD, bail=True, thickness=3):
    """Draws realistic wooden cricket stumps matching overlay.mp4 / render.py."""
    # Directional drop shadows
    for xs in STUMP_X:
        p_base = np.array([xs, y_s, 0.001])
        p_sh = np.array([xs - 0.20, y_s + 0.35, 0.001])
        sh_seg = project_segments(np.array([p_base, p_sh]), rv, C, f, size)
        for p0, p1 in sh_seg:
            cv2.line(card, p0, p1, (45, 55, 45), thickness + 1, cv2.LINE_AA)

    # Stumps
    for xs in STUMP_X:
        seg = project_segments(np.array([[xs, y_s, 0.0], [xs, y_s, stump_h]]), rv, C, f, size)
        for p0, p1 in seg:
            cv2.line(card, p0, p1, col_bgr, thickness, cv2.LINE_AA)
            cv2.circle(card, p1, max(1, thickness // 2 + 1), DARK_WOOD, -1, cv2.LINE_AA)

    if bail:
        seg_b = project_segments(np.array([[STUMP_X[0] - 0.02, y_s, stump_h], [STUMP_X[-1] + 0.02, y_s, stump_h]]), rv, C, f, size)
        for p0, p1 in seg_b:
            cv2.line(card, p0, p1, col_bgr, max(1, thickness - 1), cv2.LINE_AA)

def draw_3d_ball(card, u, v, hand="RHB", radius=6):
    """Draws a 3D cricket ball marker matching overlay.mp4 styling."""
    col = BALL_LHB if hand == "LHB" else (BALL_RHB if hand == "RHB" else BALL_UNKNOWN)
    bgr = col[::-1]

    # Ground shadow
    cv2.ellipse(card, (u + 1, v + 2), (radius + 1, max(2, int(radius * 0.45))), 0, 0, 360, (30, 35, 30), -1, cv2.LINE_AA)
    # Ball core
    cv2.circle(card, (u, v), radius, bgr, -1, cv2.LINE_AA)
    # Subtle dark rim
    cv2.circle(card, (u, v), radius, (50, 50, 50), 1, cv2.LINE_AA)

def save_image_safely(img_pil, out_path):
    """Saves PIL image with retry logic to avoid transient Windows / cloud sync file locks."""
    abs_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    import time
    for attempt in range(5):
        try:
            img_pil.save(abs_path, format="PNG")
            print(f"Generated standalone: {out_path}")
            return abs_path
        except OSError:
            time.sleep(0.15)
    img_pil.save(abs_path, format="PNG")
    print(f"Generated standalone: {out_path}")
    return abs_path

def draw_batter_hand_legend(draw, x=25, y=960, w=245, h=60):
    """Draws the floating BATTER HAND card inside the scene with subtle shadow."""
    draw.rounded_rectangle([x+2, y+2, x+w+2, y+h+2], radius=12, fill=(0, 0, 0, 45))
    draw.rounded_rectangle([x, y, x+w, y+h], radius=12, fill=(255, 255, 255, 245), outline=(225, 230, 235), width=1)

    draw.text((x + 14, y + 10), "BATTER HAND", fill=(110, 115, 120), font=get_font(11, bold=True))

    items = [
        ("LHB", BALL_LHB),
        ("RHB", BALL_RHB),
        ("Unknown", BALL_UNKNOWN)
    ]
    cur_x = x + 14
    dot_y = y + 38
    for name, col in items:
        r = 5
        draw.ellipse([cur_x, dot_y - r, cur_x + 2*r, dot_y + r], fill=col, outline=(90, 95, 100), width=1)
        draw.text((cur_x + 2*r + 6, dot_y), name, fill=(35, 40, 45), font=get_font(13, bold=True), anchor="lm")
        cur_x += 2*r + 6 + int(draw.textlength(name, font=get_font(13, bold=True))) + 12

def create_neutral_stadium_header(w=IMAGE_W, h=160):
    """Creates a clean, realistic blurred stadium crowd header with NO commercial branding."""
    im = Image.new("RGB", (w, h), (40, 45, 55))
    d = ImageDraw.Draw(im)

    # Roof canopy at top
    d.rectangle([0, 0, w, 28], fill=(65, 80, 95))
    d.rectangle([0, 28, w, 38], fill=(45, 55, 68))
    for x in range(30, w, 80):
        d.line([(x, 0), (x, 38)], fill=(85, 100, 115), width=3)

    # Spectator tiers
    tiers = [
        (38, 65, (65, 75, 90)),
        (65, 92, (75, 85, 100)),
        (92, 118, (80, 90, 105)),
        (118, 138, (60, 70, 85))
    ]
    for y0, y1, t_col in tiers:
        d.rectangle([0, y0, w, y1], fill=t_col)
        d.line([(0, y1), (w, y1)], fill=(38, 42, 50), width=2)
        np.random.seed(y0)
        for _ in range(120):
            sx = np.random.randint(5, w - 5)
            sy = np.random.randint(y0 + 2, y1 - 2)
            scol = [(180, 70, 60), (230, 220, 210), (60, 100, 160), (40, 40, 45), (210, 180, 80)][np.random.randint(0, 5)]
            d.ellipse([sx - 2, sy - 2, sx + 2, sy + 2], fill=scol)

    cv_img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    cv_blur = cv2.GaussianBlur(cv_img, (11, 11), 0)

    # Clean dark perimeter barrier ribbon at bottom (y = 138 to 160)
    cv_blur[138:160, :] = (42, 48, 55)
    cv2.line(cv_blur, (0, 138), (w, 138), (80, 90, 100), 1, cv2.LINE_AA)
    cv2.line(cv_blur, (0, 159), (w, 159), (30, 35, 40), 1, cv2.LINE_AA)
    return cv_blur

def get_tripod_camera(cal_path="outputs/calibration.json", size=(IMAGE_W, IMAGE_H)):
    """Retrieves the exact calibrated tripod camera position from overlay.mp4."""
    w, h = size
    if os.path.exists(cal_path):
        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
            C = np.array(cal["C"], float)
            rvec = np.array(cal["rvec"], float)
            f_cal = float(cal["f"]) * (h / 1280.0)
            return C, rvec, f_cal
        except Exception as e:
            print(f"Warning: could not load {cal_path}: {e}")

    # Default calibrated tripod camera from overlay.mp4
    C = np.array([0.02, -6.73, 1.52], float)
    rvec = np.array([1.6312, 0.0043, -0.0043], float)
    f_cal = 1920.0 * (h / 1280.0)
    return C, rvec, f_cal

# --- Image 1: Release Points (Standalone Scene) ---

def render_release_points_image(deliveries, out_path="outputs/release_points.png"):
    """
    Renders the standalone Release Points cricket analytics image:
    Uses the exact calibrated tripod placement and angle from overlay.mp4,
    matching pitch and ground texture, 3D measurement frame starting DIRECTLY from the bowling crease,
    foreground wooden stumps with bails, average release height, and ball clusters.
    """
    w, h = IMAGE_W, IMAGE_H
    card = make_turf_background((w, h))

    # Stadium crowd header at top (neutral broadcast)
    stadium_h = 160
    card[0:stadium_h, :] = create_neutral_stadium_header(w, stadium_h)

    # EXACT TRIPOD CAMERA ANGLE from overlay.mp4
    C, rv, f = get_tripod_camera("outputs/calibration.json", (w, h))
    size = (w, h)

    # Pitch surface starts at the bowler stumps (Y=0.0) and extends to batter end (Y=PITCH_L)
    # The pitch and blue delivery corridor do NOT continue behind the bowler stumps
    pitch_poly = np.array([
        [-PITCH_HW, 0.0, 0.0],
        [ PITCH_HW, 0.0, 0.0],
        [ PITCH_HW, PITCH_L + 1.2, 0.0],
        [-PITCH_HW, PITCH_L + 1.2, 0.0]
    ])
    p_px = project_poly(pitch_poly, rv, C, f, size)
    if p_px is not None:
        draw_overlay_pitch_turf(card, p_px, rv, C, f, size, y_start=0.0, y_end=PITCH_L, with_corridor=True)

    # Professional Cricket Creases: exactly 2 horizontal lines at bowler end (bowling crease & popping crease)
    draw_professional_creases(card, rv, C, f, size, ends=('bowler', 'batter'), bowler_behind=0.0)

    # Stumps: Bowler stumps in foreground (at Y=0.0), Batter stumps in distance (at Y=20.12)
    draw_3d_stumps(card, 0.0, rv, C, f, size, stump_h=STUMP_H, col_bgr=WOOD, bail=True, thickness=4)
    draw_3d_stumps(card, PITCH_L, rv, C, f, size, stump_h=STUMP_H, col_bgr=WOOD, bail=True, thickness=2)

    # Average Release Height
    valid_releases = [d["release_pos"] for d in deliveries if d.get("release_pos") is not None]
    if valid_releases:
        avg_z = float(np.mean([rp[2] for rp in valid_releases]))
    else:
        avg_z = 2.19

    # White 3D Measurement Frame: Starts DIRECTLY AT THE BOWLER'S NO-BALL CREASE (Y = 1.22m)
    # The release point is captured at the popping crease / no-ball crease
    frame_y = POP  # 1.22m (Bowler's no-ball crease)
    frame_z_max = 2.45
    frame_x = 1.35  # Frames release corridor matching return crease width

    p_l0 = project_point([-frame_x, frame_y, 0.0], rv, C, f, size)
    p_l1 = project_point([-frame_x, frame_y, frame_z_max], rv, C, f, size)
    p_r0 = project_point([frame_x, frame_y, 0.0], rv, C, f, size)
    p_r1 = project_point([frame_x, frame_y, frame_z_max], rv, C, f, size)

    if p_l0 is not None and p_l1 is not None:
        cv2.line(card, tuple(p_l0.round().astype(int)), tuple(p_l1.round().astype(int)), (255, 255, 255), 3, cv2.LINE_AA)
        p_ltop = project_point([-frame_x + 0.16, frame_y, frame_z_max], rv, C, f, size)
        if p_ltop is not None:
            cv2.line(card, tuple(p_l1.round().astype(int)), tuple(p_ltop.round().astype(int)), (255, 255, 255), 3, cv2.LINE_AA)
        p_lbot = project_point([-frame_x + 0.16, frame_y, 0.0], rv, C, f, size)
        if p_lbot is not None:
            cv2.line(card, tuple(p_l0.round().astype(int)), tuple(p_lbot.round().astype(int)), (255, 255, 255), 3, cv2.LINE_AA)

    if p_r0 is not None and p_r1 is not None:
        cv2.line(card, tuple(p_r0.round().astype(int)), tuple(p_r1.round().astype(int)), (255, 255, 255), 3, cv2.LINE_AA)
        p_rtop = project_point([frame_x - 0.16, frame_y, frame_z_max], rv, C, f, size)
        if p_rtop is not None:
            cv2.line(card, tuple(p_r1.round().astype(int)), tuple(p_rtop.round().astype(int)), (255, 255, 255), 3, cv2.LINE_AA)
        p_rbot = project_point([frame_x - 0.16, frame_y, 0.0], rv, C, f, size)
        if p_rbot is not None:
            cv2.line(card, tuple(p_r0.round().astype(int)), tuple(p_rbot.round().astype(int)), (255, 255, 255), 3, cv2.LINE_AA)

    # Yellow dashed line at Z = avg_z across the release frame
    p_yl = project_point([-frame_x, frame_y, avg_z], rv, C, f, size)
    p_yr = project_point([frame_x, frame_y, avg_z], rv, C, f, size)
    if p_yl is not None and p_yr is not None:
        x_pts = np.linspace(p_yl[0], p_yr[0], 28)
        y_pts = np.linspace(p_yl[1], p_yr[1], 28)
        for i in range(0, len(x_pts) - 1, 2):
            cv2.line(card, (int(round(x_pts[i])), int(round(y_pts[i]))),
                           (int(round(x_pts[i+1])), int(round(y_pts[i+1]))), (0, 215, 245), 2, cv2.LINE_AA)

    # Plotted Release Points projected on the measurement frame matching overlay.mp4 ball styling
    for d in deliveries:
        rp = d.get("release_pos")
        if rp is not None:
            pt = project_point([rp[0], frame_y, rp[2]], rv, C, f, size)
            if pt is not None:
                u, v = int(round(pt[0])), int(round(pt[1]))
                draw_3d_ball(card, u, v, hand=d.get("batter_hand", "RHB"), radius=7)

    img_pil = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Frame Ticks: 2M and 1M on left and right directly measuring from the bowler's no-ball crease
    for z_tick, z_label in [(2.0, "2M"), (1.0, "1M")]:
        pt_l = project_point([-frame_x, frame_y, z_tick], rv, C, f, size)
        if pt_l is not None:
            lx, ly = int(pt_l[0]), int(pt_l[1])
            draw.line([(lx, ly), (lx + 16, ly)], fill=(255, 255, 255), width=2)
            draw.text((lx - 8, ly), z_label, fill=(255, 255, 255), font=get_font(18, bold=True), anchor="rm")
        pt_r = project_point([frame_x, frame_y, z_tick], rv, C, f, size)
        if pt_r is not None:
            rx, ry = int(pt_r[0]), int(pt_r[1])
            draw.line([(rx - 16, ry), (rx, ry)], fill=(255, 255, 255), width=2)
            draw.text((rx + 8, ry), z_label, fill=(255, 255, 255), font=get_font(18, bold=True), anchor="lm")
            lx, ly = int(pt_l[0]), int(pt_l[1])
            draw.line([(lx, ly), (lx + 16, ly)], fill=(255, 255, 255), width=2)
            draw.text((lx - 8, ly), z_label, fill=(255, 255, 255), font=get_font(18, bold=True), anchor="rm")
        pt_r = project_point([frame_x, frame_y, z_tick], rv, C, f, size)
        if pt_r is not None:
            rx, ry = int(pt_r[0]), int(pt_r[1])
            draw.line([(rx - 16, ry), (rx, ry)], fill=(255, 255, 255), width=2)
            draw.text((rx + 8, ry), z_label, fill=(255, 255, 255), font=get_font(18, bold=True), anchor="lm")

    # Average Release Height pill placed cleanly below wickets/crease
    pill_text = f"Average Release Height: {avg_z:.2f}m"
    pw, ph = 460, 56
    px = (w - pw) // 2
    py = 830
    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=16, fill=COLOR_NAVY_PILL)
    draw.text((w // 2, py + ph // 2), pill_text, fill=(255, 255, 255), font=get_font(22, bold=True), anchor="mm")

    # Floating BATTER HAND legend
    draw_batter_hand_legend(draw, x=25, y=960, w=245, h=60)

    return save_image_safely(img_pil, out_path)

# --- Image 2: Beehive (Standalone Scene) ---

def render_beehive_image(deliveries, out_path="outputs/beehive.png"):
    """
    Renders the standalone Beehive cricket analytics image:
    Matches turf and pitch texture from overlay.mp4, continuous pitch extending towards
    the foreground off the bottom of the canvas, official ICC batter-end crease markings,
    4 color-coded arrival zones, wooden stumps with bails, top percentage labels, and height ticks.
    """
    w, h = IMAGE_W, IMAGE_H
    card = make_turf_background((w, h))

    # Stadium crowd header at top (neutral broadcast)
    stadium_h = 160
    card[0:stadium_h, :] = create_neutral_stadium_header(w, stadium_h)

    # Elevated camera looking at batter-end wickets
    C = np.array([0.0, 7.5, 2.30], float)
    target = np.array([0.0, 20.12, 1.05], float)
    f = 1750.0
    rv = lookat_rvec(C, target)
    size = (w, h)

    # Continuous pitch strip extending from Y=13.0 (off bottom of canvas) to Y=21.5 (behind wickets)
    pitch_poly = np.array([
        [-PITCH_HW, 13.0, 0.0],
        [ PITCH_HW, 13.0, 0.0],
        [ PITCH_HW, 21.5, 0.0],
        [-PITCH_HW, 21.5, 0.0]
    ])
    p_px = project_poly(pitch_poly, rv, C, f, size)
    if p_px is not None:
        draw_overlay_pitch_turf(card, p_px, rv, C, f, size, y_start=13.0, y_end=21.5, with_corridor=False)

    # Professional Cricket Creases at batter end
    draw_professional_creases(card, rv, C, f, size, ends=('batter',))

    # 4-Zone Beehive Grid standing vertically at Y = 20.12
    grid_y = 20.12
    grid_z_top = 2.10
    zones_def = [
        ("Z1", -1.20, -0.35, (56, 185, 160), 0.55, (176, 200, 54)),   # Olive / Yellow-green
        ("Z2", -0.35, 0.00,  (80, 175, 76),  0.60, (50, 205, 50)),     # Green
        ("Z3", 0.00,  0.35,  (64, 96, 230),  0.60, (240, 90, 60)),     # Coral / Orange-Red
        ("Z4", 0.35,  1.20,  (200, 120, 59), 0.55, (59, 120, 230))     # Blue
    ]

    z_counts = [0, 0, 0, 0]
    total_valid = 0
    for d in deliveries:
        arr = d.get("stump_arrival")
        if arr is not None:
            x_arr = arr[0]
            total_valid += 1
            if x_arr < -0.35:
                z_counts[0] += 1
            elif x_arr < 0.00:
                z_counts[1] += 1
            elif x_arr < 0.35:
                z_counts[2] += 1
            else:
                z_counts[3] += 1

    z_pcts = [int(round(c / total_valid * 100)) if total_valid > 0 else 0 for c in z_counts]

    ov = card.copy()
    for (zid, x0, x1, col_bgr, alpha, _), pct in zip(zones_def, z_pcts):
        poly_3d = np.array([
            [x0, grid_y, 0.0],
            [x1, grid_y, 0.0],
            [x1, grid_y, grid_z_top],
            [x0, grid_y, grid_z_top]
        ])
        z_px = project_poly(poly_3d, rv, C, f, size)
        if z_px is not None:
            cv2.fillPoly(ov, [z_px], col_bgr)
            cv2.polylines(ov, [z_px], True, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.55, card, 0.45, 0, card)

    # Grid outer frame and divider lines
    grid_outer = np.array([
        [-1.20, grid_y, 0.0],
        [ 1.20, grid_y, 0.0],
        [ 1.20, grid_y, grid_z_top],
        [-1.20, grid_y, grid_z_top]
    ])
    g_px = project_poly(grid_outer, rv, C, f, size)
    if g_px is not None:
        cv2.polylines(card, [g_px], True, (255, 255, 255), 2, cv2.LINE_AA)

    for div_x in [-0.35, 0.00, 0.35]:
        p_bot = project_point([div_x, grid_y, 0.0], rv, C, f, size)
        p_top = project_point([div_x, grid_y, grid_z_top], rv, C, f, size)
        if p_bot is not None and p_top is not None:
            cv2.line(card, tuple(p_bot.round().astype(int)), tuple(p_top.round().astype(int)), (255, 255, 255), 2, cv2.LINE_AA)

    # Batter Stumps with realistic wood tone and bails
    draw_3d_stumps(card, PITCH_L, rv, C, f, size, stump_h=STUMP_H, col_bgr=WOOD, bail=True, thickness=3)

    # Plotted Balls matching overlay.mp4 styling
    for d in deliveries:
        arr = d.get("stump_arrival")
        if arr is not None:
            pt = project_point([arr[0], grid_y, arr[2]], rv, C, f, size)
            if pt is not None:
                u, v = int(round(pt[0])), int(round(pt[1]))
                draw_3d_ball(card, u, v, hand=d.get("batter_hand", "RHB"), radius=7)

    img_pil = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Top Percentages placed cleanly above the grid line
    for (zid, x0, x1, _, _, col_rgb), pct in zip(zones_def, z_pcts):
        x_mid = (x0 + x1) / 2.0
        pt_top = project_point([x_mid, grid_y, grid_z_top], rv, C, f, size)
        if pt_top is not None:
            tx, ty = int(pt_top[0]), int(pt_top[1]) - 16
            draw.text((tx + 1, ty + 1), f"{pct}%", fill=(20, 30, 20), font=get_font(22, bold=True), anchor="mm")
            draw.text((tx, ty), f"{pct}%", fill=col_rgb, font=get_font(22, bold=True), anchor="mm")

    # Height ticks: — 2M and — 1M
    right_x = 1.20
    for h_val, h_label in [(2.0, "— 2M"), (1.0, "— 1M")]:
        pt_h = project_point([right_x, grid_y, h_val], rv, C, f, size)
        if pt_h is not None:
            draw.text((int(pt_h[0]) + 6, int(pt_h[1])), h_label, fill=(255, 255, 255), font=get_font(18, bold=True), anchor="lm")

    # Floating BATTER HAND legend
    draw_batter_hand_legend(draw, x=25, y=960, w=245, h=60)

    return save_image_safely(img_pil, out_path)

# --- Image 3: Pitch Map (Standalone Scene) ---

def render_pitch_map_image(deliveries, out_path="outputs/pitch_map.png"):
    """
    Renders the standalone Pitch Map cricket analytics image:
    Wide, broad pitch perspective (not narrow/tapered), textures matching overlay.mp4,
    left labels placed directly by the side of their respective zones,
    right measurement capsules aligned on the right, both bowler and batter stumps visible,
    official ICC crease markings, and delivery bounce markers.
    """
    w, h = IMAGE_W, IMAGE_H
    card = make_turf_background((w, h))

    # Elevated top-down broadcast camera: gentle taper, wide and broad view
    C = np.array([0.0, -2.5, 18.0], float)
    target = np.array([0.0, 8.0, 0.0], float)
    f = 930.0
    rv = lookat_rvec(C, target)
    size = (w, h)

    # Wide pitch strip modeling broadcast pitch map cards
    pitch_hw = 3.6
    pitch_poly = np.array([
        [-pitch_hw, -1.4, 0.0],
        [ pitch_hw, -1.4, 0.0],
        [ pitch_hw, PITCH_L + 1.3, 0.0],
        [-pitch_hw, PITCH_L + 1.3, 0.0]
    ])
    p_px = project_poly(pitch_poly, rv, C, f, size)
    if p_px is not None:
        draw_overlay_pitch_turf(card, p_px, rv, C, f, size, hw=pitch_hw, y_start=-1.4, y_end=PITCH_L + 1.3, with_corridor=True)

    # Length Bands: authentic cricket pitch zones
    # Full Toss: from bowling crease (20.12m) to beyond popping crease (21.4m)
    # Yorker: 0.0m to 2.0m from batter stumps (Y = 18.12 to 20.12)
    # Full: 2.0m to 5.0m from batter stumps (Y = 15.12 to 18.12)
    # Good: 5.0m to 8.0m from batter stumps (Y = 12.12 to 15.12)
    # Short: 8.0m to 12.5m from batter stumps (Y = 7.62 to 12.12)
    # From Y = -1.4 to 7.62: Clean pitch turf with roller lines & blue delivery corridor
    bands = [
        ("Short", 7.62, 12.12, (194, 115, 124), 0.55),      # Slate Purple/Violet
        ("Good", 12.12, 15.12, (83, 83, 229), 0.65),        # Coral/Red
        ("Full", 15.12, 18.12, (80, 175, 76), 0.65),        # Green
        ("Yorker", 18.12, 20.12, (42, 177, 212), 0.65),     # Yellow/Gold
        ("Full Toss", 20.12, 21.4, (180, 190, 195), 0.50)   # Tan/Grey
    ]

    counts = {"Full Toss": 0, "Yorker": 0, "Full": 0, "Good": 0, "Short": 0}
    total_valid = 0
    for d in deliveries:
        if d.get("bounce_pos") is not None:
            length_m = d["length_m"]
            total_valid += 1
            if length_m < 0.0:
                counts["Full Toss"] += 1
            elif length_m < 2.0:
                counts["Yorker"] += 1
            elif length_m < 5.0:
                counts["Full"] += 1
            elif length_m < 8.0:
                counts["Good"] += 1
            else:
                counts["Short"] += 1

    pcts = {}
    for k, v in counts.items():
        pcts[k] = int(round(v / total_valid * 100)) if total_valid > 0 else 0

    for name, y0, y1, col_bgr, alpha in bands:
        b_poly = np.array([
            [-pitch_hw, y0, 0.002],
            [ pitch_hw, y0, 0.002],
            [ pitch_hw, y1, 0.002],
            [-pitch_hw, y1, 0.002]
        ])
        bp_px = project_poly(b_poly, rv, C, f, size)
        if bp_px is not None:
            ov = card.copy()
            cv2.fillPoly(ov, [bp_px], col_bgr)
            cv2.addWeighted(ov, alpha, card, 1.0 - alpha, 0, card)

    # Official Professional Cricket Creases at both ends
    draw_professional_creases(card, rv, C, f, size, ends=('bowler', 'batter'), bowler_behind=1.22)

    # Prominent wooden stumps at BOTH ends
    draw_3d_stumps(card, 0.0, rv, C, f, size, stump_h=STUMP_H, col_bgr=WOOD, bail=True, thickness=3)
    draw_3d_stumps(card, PITCH_L, rv, C, f, size, stump_h=STUMP_H, col_bgr=WOOD, bail=True, thickness=2)

    # Plotted Bounce Dots matching overlay.mp4 styling
    for d in deliveries:
        b_pos = d.get("bounce_pos")
        if b_pos is not None:
            pt = project_point(b_pos, rv, C, f, size)
            if pt is not None:
                u, v = int(round(pt[0])), int(round(pt[1]))
                draw_3d_ball(card, u, v, hand=d.get("batter_hand", "RHB"), radius=7)

    img_pil = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Pitch specification pill at top right
    draw.rounded_rectangle([485, 25, 685, 78], radius=14, fill=(255, 255, 255, 240), outline=(230, 235, 230), width=1)
    draw.text((585, 42), "22 yds / 20.12 m", fill=(20, 35, 60), font=get_font(16, bold=True), anchor="mm")
    draw.text((585, 62), "Stump to Stump", fill=(120, 130, 135), font=get_font(12), anchor="mm")

    # Left Category Labels: Placed DIRECTLY BY THE SIDE of each respective zone
    label_info = [
        (f"Full Toss ({pcts['Full Toss']}%)", 20.76, (25, 25, 25)),
        (f"Yorker ({pcts['Yorker']}%)", 19.12, (235, 225, 45)),
        (f"Full ({pcts['Full']}%)", 16.62, (45, 225, 65)),
        (f"Good ({pcts['Good']}%)", 13.62, (240, 50, 50)),
        (f"Short ({pcts['Short']}%)", 9.87, (65, 110, 240))
    ]
    for text, y_world, col in label_info:
        pt = project_point([-pitch_hw, y_world, 0.0], rv, C, f, size)
        if pt is not None:
            lx, ly = int(pt[0]) - 16, int(pt[1])
            draw.text((lx + 1, ly + 1), text, fill=(30, 40, 30), font=get_font(18, bold=True), anchor="rm")
            draw.text((lx, ly), text, fill=col, font=get_font(18, bold=True), anchor="rm")

    # Right pills: Placed DIRECTLY BY THE RIGHT SIDE of each horizontal boundary
    pills = [
        ("0.0m", 20.12),
        ("2.0m", 18.12),
        ("5.0m", 15.12),
        ("8.0m", 12.12)
    ]
    for dist_label, y_world in pills:
        pt = project_point([pitch_hw, y_world, 0.0], rv, C, f, size)
        if pt is not None:
            py = int(pt[1])
            px = int(pt[0]) + 14
            cap_w, cap_h = 38, 24
            draw.rounded_rectangle([px, py - cap_h // 2, px + cap_w, py + cap_h // 2], radius=12, fill=(255, 255, 255), outline=(40, 50, 40), width=1)
            draw_double_arrow(draw, px + cap_w // 2, py)
            draw.text((px + cap_w + 10, py), dist_label, fill=(245, 250, 245), font=get_font(18, bold=True), anchor="lm")

    # Floating BATTER HAND legend
    draw_batter_hand_legend(draw, x=25, y=960, w=245, h=60)

    return save_image_safely(img_pil, out_path)

def generate_session_analytics(deliveries_source="outputs/batch_summary.json", out_dir="outputs"):
    """
    Independent generator of three standalone session-level cricket analytics images:
    1. pitch_map.png
    2. beehive.png
    3. release_points.png
    """
    deliveries = load_session_deliveries(deliveries_source)
    if not deliveries:
        print("Warning: No valid deliveries found to generate session analytics.")
        return {}

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    # 1. Pitch Map
    pm_path = os.path.join(out_dir, "pitch_map.png")
    render_pitch_map_image(deliveries, out_path=pm_path)
    results["pitch_map"] = pm_path

    # 2. Beehive
    bh_path = os.path.join(out_dir, "beehive.png")
    render_beehive_image(deliveries, out_path=bh_path)
    results["beehive"] = bh_path

    # 3. Release Points
    rp_path = os.path.join(out_dir, "release_points.png")
    render_release_points_image(deliveries, out_path=rp_path)
    results["release_points"] = rp_path

    return results

if __name__ == "__main__":
    generate_session_analytics()
