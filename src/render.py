import subprocess, shutil, os
import cv2, numpy as np
from geometry import *

RED = (30, 30, 230)
BLUE = (235, 140, 120)       # Translucent cyan/periwinkle pitch corridor
WHITE = (255, 255, 255)
WOOD = (65, 185, 230)        # Timber wicket tone
DARK_WOOD = (30, 40, 50)


def _R(rvec):
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def _pts(P, rvec, C, f, size):
    return np.round(project(P, rvec, C, f, size)).astype(np.int32)


def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=8):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    radius = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)
    if thickness == -1:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def project_point(pt_3d, rvec, C, f, size, near_z=0.2):
    R = _R(rvec)
    Xc = (np.asarray(pt_3d, float) - np.asarray(C, float)) @ R.T
    if Xc[2] < near_z:
        return None, Xc[2]
    u = f * Xc[0] / Xc[2] + size[0] / 2
    v = f * Xc[1] / Xc[2] + size[1] / 2
    return np.array([u, v]), Xc[2]


def project_segments(pts_3d, rvec, C, f, size, near_z=0.2):
    pts = np.asarray(pts_3d, float)
    R = _R(rvec)
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
        u0 = f * p0[0] / p0[2] + size[0] / 2
        v0 = f * p0[1] / p0[2] + size[1] / 2
        u1 = f * p1[0] / p1[2] + size[0] / 2
        v1 = f * p1[1] / p1[2] + size[1] / 2
        segments.append([(int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1)))])
    return segments


def project_polygon(poly_3d, rvec, C, f, size, near_z=0.2):
    pts = np.asarray(poly_3d, float)
    R = _R(rvec)
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
                inter = prev_p + alpha * (curr_p - prev_p)
                out.append(inter)
            out.append(curr_p)
        elif prev_in:
            alpha = (near_z - prev_p[2]) / (curr_p[2] - prev_p[2] + 1e-12)
            inter = prev_p + alpha * (curr_p - prev_p)
            out.append(inter)
    if len(out) < 3:
        return None
    out = np.array(out)
    u = f * out[:, 0] / out[:, 2] + size[0] / 2
    v = f * out[:, 1] / out[:, 2] + size[1] / 2
    return np.stack([u, v], 1).round().astype(np.int32)


def get_virtual_view_trajectory(pos, times, vel=None, target_y=PITCH_L):
    """Extrapolates past batter if necessary and clamps precisely at the stump-line plane."""
    P = np.array(pos, float).copy()
    T = np.array(times, float).copy()

    if P[-1, 1] < target_y:
        if len(P) >= 2:
            v = (P[-1] - P[-2]) / (T[-1] - T[-2] + 1e-9)
        elif vel is not None and len(vel) > 0:
            v = vel[-1].copy()
        else:
            v = np.array([0.0, 20.0, 0.0])
        if v[1] <= 1.0:
            v[1] = 20.0

        dt = 0.005
        p_curr = P[-1].copy()
        t_curr = T[-1]
        P_ext, T_ext = [], []
        while p_curr[1] < target_y + 0.1:
            t_curr += dt
            p_curr = p_curr + v * dt
            p_curr[2] -= 0.5 * 9.81 * dt**2
            v[2] -= 9.81 * dt
            if p_curr[2] < BALL_R:
                p_curr[2] = BALL_R
            P_ext.append(p_curr.copy())
            T_ext.append(t_curr)
        if P_ext:
            P = np.vstack([P, np.array(P_ext)])
            T = np.concatenate([T, np.array(T_ext)])

    # Clamp precisely at target_y (stump line)
    idx = np.where(P[:, 1] >= target_y)[0]
    if len(idx) > 0:
        first_idx = idx[0]
        if first_idx > 0:
            p0, p1 = P[first_idx-1], P[first_idx]
            t0, t1 = T[first_idx-1], T[first_idx]
            frac = (target_y - p0[1]) / (p1[1] - p0[1] + 1e-9)
            p_end = p0 + frac * (p1 - p0)
            t_end = t0 + frac * (t1 - t0)
            P = np.vstack([P[:first_idx], p_end.reshape(1, 3)])
            T = np.concatenate([T[:first_idx], [t_end]])
        else:
            P = P[:1]
            T = T[:1]
    return P, T


def draw_ball(img, pos, rvec, C, f, size):
    """Draws a clean white ball with neat dark contour, matching original video ball size and appearance."""
    pt_2d, z_c = project_point(pos, rvec, C, f, size, near_z=0.2)
    if pt_2d is None:
        return
    u, v = int(round(pt_2d[0])), int(round(pt_2d[1]))
    # Ball radius matches original video (5 to 8 pixels)
    r_ball = int(np.clip(round(f * 0.05 / z_c), 5, 8))

    # Ground shadow on turf when near pitch
    if pos[2] < 0.8:
        pt_ground, _ = project_point([pos[0], pos[1], 0.001], rvec, C, f, size)
        if pt_ground is not None:
            ug, vg = int(round(pt_ground[0])), int(round(pt_ground[1]))
            cv2.ellipse(img, (ug, vg), (r_ball + 1, max(2, int(r_ball * 0.45))), 0, 0, 360, (35, 40, 35), -1, cv2.LINE_AA)

    # Crisp solid white ball with subtle dark rim
    cv2.circle(img, (u, v), r_ball, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (u, v), r_ball, (50, 50, 50), 1, cv2.LINE_AA)


def draw_metrics_cards(img, metrics, sf_scale=None, start_x=28, start_y=50, w=160, h=50, alpha=0.82):
    sw = metrics.get("swing_dev_cm", 0.0) * (sf_scale if sf_scale else 1.0)
    sw_unit = "sf" if sf_scale else "cm"
    cards = [
        ("Swing", f"{sw:.1f} {sw_unit}", True),
        ("Spin", f"{metrics.get('spin_deg', 0.0):.1f} deg", True),
        ("Speed", f"{metrics['speed_mph']:.0f} mph", False)
    ]
    ov = img.copy()
    for i, (title, val, _) in enumerate(cards):
        cy = start_y + i * (h + 10)
        draw_rounded_rect(ov, (start_x, cy), (start_x + w, cy + h), (18, 22, 20), -1, radius=8)
    cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0, img)

    for i, (title, val, has_info) in enumerate(cards):
        cy = start_y + i * (h + 10)
        draw_rounded_rect(img, (start_x, cy), (start_x + w, cy + h), (90, 110, 95), 1, radius=8)
        cv2.putText(img, title, (start_x + 12, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 215, 205), 1, cv2.LINE_AA)
        cv2.putText(img, val, (start_x + 12, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        if has_info:
            ix = start_x + w - 20
            iy = cy + 34
            cv2.circle(img, (ix, iy), 8, (210, 185, 60), 1, cv2.LINE_AA)
            cv2.putText(img, "i", (ix - 2, iy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 185, 60), 1, cv2.LINE_AA)


def draw_cricket_creases(img, rv, C, f, size, color=WHITE, thickness=2):
    """Draws authentic ICC/MCC cricket crease markings at bowler and batter ends."""
    WIDE = 0.89

    # 1. Bowling Creases (between return creases: 2.64m total width)
    for y_b in [0.0, PITCH_L]:
        seg = project_segments(np.array([[-RET, y_b, 0.003], [RET, y_b, 0.003]]), rv, C, f, size)
        for p0, p1 in seg:
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)

    # 2. Popping Creases (between return creases across pitch)
    for y_p in [POP, PITCH_L - POP]:
        seg = project_segments(np.array([[-RET, y_p, 0.003], [RET, y_p, 0.003]]), rv, C, f, size)
        for p0, p1 in seg:
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)

    # 3. Return Creases (connecting popping crease to 1m behind bowling crease)
    for sign in [-1, 1]:
        seg_ret0 = project_segments(np.array([[sign * RET, -1.0, 0.003], [sign * RET, POP, 0.003]]), rv, C, f, size)
        for p0, p1 in seg_ret0:
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
        seg_ret1 = project_segments(np.array([[sign * RET, PITCH_L + 1.0, 0.003], [sign * RET, PITCH_L - POP, 0.003]]), rv, C, f, size)
        for p0, p1 in seg_ret1:
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)

    # 4. Wide Guidelines (between popping crease and bowling crease)
    for sign in [-1, 1]:
        w_seg0 = project_segments(np.array([[sign * WIDE, 0.0, 0.003], [sign * WIDE, POP, 0.003]]), rv, C, f, size)
        w_seg1 = project_segments(np.array([[sign * WIDE, PITCH_L, 0.003], [sign * WIDE, PITCH_L - POP, 0.003]]), rv, C, f, size)
        for p0, p1 in w_seg0 + w_seg1:
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


def _pitch_geometry(img, rvec, C, f, size, fill=None, bounce_y=None):
    """Draws pitch overlay on original video. NO white crease lines drawn here per user specification.
    Blue strip starts from stump itself (Y=0.0) without overlapping or washing out the stumps."""
    if fill is not None:
        hw = PITCH_HW
        poly = np.array([[-hw, -1.2, 0], [hw, -1.2, 0], [hw, PITCH_L + 1.2, 0], [-hw, PITCH_L + 1.2, 0]])
        ov = img.copy()
        cv2.fillPoly(ov, [_pts(poly, rvec, C, f, size)], fill)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    if bounce_y is not None:
        orig = img.copy()
        strip = np.array([[-.15, 0.0, 0.002], [.15, 0.0, 0.002], [.15, PITCH_L, 0.002], [-.15, PITCH_L, 0.002]])
        ov = img.copy()
        cv2.fillPoly(ov, [_pts(strip, rvec, C, f, size)], BLUE)
        cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)

        # Preserve the near stumps on the original video so the blue corridor does not overlap them
        w, h = size
        stump_mask = np.zeros((h, w), dtype=np.uint8)
        zc_list = []
        bots = []
        for xs in STUMP_X:
            p_bot, zc = project_point([xs, 0.0, 0.0], rvec, C, f, size)
            p_top, _ = project_point([xs, 0.0, STUMP_H], rvec, C, f, size)
            if p_bot is not None and p_top is not None:
                zc_list.append(zc)
                bots.append(p_bot)
                thick = int(np.clip(round(f * 0.038 / zc), 6, 16))
                cv2.line(stump_mask, (int(round(p_top[0])), int(round(p_top[1]))),
                                     (int(round(p_bot[0])), int(round(p_bot[1]))), 255, thick)
        if bots:
            zc_m = np.mean(zc_list)
            b_h = int(np.clip(round(f * 0.045 / zc_m), 8, 24))
            b_w_pad = int(np.clip(round(f * 0.04 / zc_m), 8, 20))
            min_x = int(min(b[0] for b in bots)) - b_w_pad
            max_x = int(max(b[0] for b in bots)) + b_w_pad
            bot_y = int(np.mean([b[1] for b in bots]))
            cv2.rectangle(stump_mask, (min_x, bot_y - b_h), (max_x, bot_y + int(b_h * 0.3)), 255, -1)

        np.copyto(img, orig, where=(stump_mask[:, :, None] > 0))


def get_orbit_camera(phi_deg, cal=None, size=(720, 1280)):
    """Computes camera parameters for Front -> Side -> Batter views.
    Anchors Front view (phi=0) to calibrated original video pitch scale and angle.
    Side view (phi=90) is parallel with pitch and closeup.
    Stops at Batter view (phi=180) exactly 4m behind batter stumps (X=0, Y=PITCH_L+4m),
    maintaining identical pitch ratio, scale, and perspective angle as the start view."""
    w, h = size
    if cal is not None:
        f_cal = float(cal['f'])
        c_cal = np.array(cal['C'], float)
    else:
        f_cal = 1920.0 * (w / 720.0)
        c_cal = np.array([0.0, -6.7, 1.52], float)

    d_start = max(5.0, abs(c_cal[1]))
    s_scale = 4.0 / d_start

    # State 0: Front View (phi = 0.0)
    C_front = np.array([0.0, c_cal[1], c_cal[2]], float)
    t_front = np.array([0.0, 12.6, 0.355], float)
    f_front = f_cal

    # State 1: Side View (phi = 90.0) - parallel with pitch and closeup
    C_side = np.array([6.2, 17.5, 1.35], float)
    t_side = np.array([0.0, 17.5, 0.30], float)
    f_side = f_cal * 0.40

    # State 2: Batter End View (phi = 180.0)
    # Exactly behind the batter stumps (X=0), 4m behind stumps (Y=PITCH_L + 4.0)
    # Pitch ratio, scale, and angle matched to start view
    C_batter = np.array([0.0, PITCH_L + 4.0, c_cal[2] * s_scale], float)
    t_batter = np.array([0.0, (PITCH_L + 4.0) - (12.6 - c_cal[1]) * s_scale, 0.355 * s_scale], float)
    f_batter = f_cal * s_scale

    phi = float(np.clip(phi_deg, 0.0, 180.0))
    if phi <= 90.0:
        w = phi / 90.0
        s = w * w * (3.0 - 2.0 * w)
        t = (1.0 - s) * t_front + s * t_side
        C = np.zeros(3, float)
        C[0] = np.sin(0.5 * np.pi * w) * C_side[0]
        C[1] = (1.0 - s) * C_front[1] + s * C_side[1]
        C[2] = (1.0 - s) * C_front[2] + s * C_side[2]
        f = (1.0 - s) * f_front + s * f_side
    else:
        w = (phi - 90.0) / 90.0
        s = w * w * (3.0 - 2.0 * w)
        t = (1.0 - s) * t_side + s * t_batter
        C = np.zeros(3, float)
        C[0] = np.cos(0.5 * np.pi * w) * C_side[0]
        C[1] = (1.0 - s) * C_side[1] + s * C_batter[1]
        C[2] = (1.0 - s) * C_side[2] + s * C_batter[2]
        f = (1.0 - s) * f_side + s * f_batter

    rv = lookat_rvec(C, t)
    return C, rv, f, t


def make_turf_background(size=(720, 1280)):
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - w // 2)**2 + (Y - int(h * 0.53))**2)
    spotlight = np.exp(-0.5 * (dist / 480.0)**2)

    base_b = 82 - 15 * (Y / h)
    base_g = 142 - 20 * (Y / h)
    base_r = 72 - 15 * (Y / h)

    b = np.clip(base_b + 20 * spotlight, 0, 255).astype(np.uint8)
    g = np.clip(base_g + 26 * spotlight, 0, 255).astype(np.uint8)
    r = np.clip(base_r + 18 * spotlight, 0, 255).astype(np.uint8)

    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def render_virtual_3d_frame(phi_deg, metrics, info_or_pos, ball_pos=None, visible_idx=None,
                            bounce_pos=None, size=(720, 1280), sf_scale=None,
                            cal=None, bg_template=None, delivery_label=None):
    w, h = size
    C, rv, f, _ = get_orbit_camera(phi_deg, cal=cal, size=size)

    if isinstance(info_or_pos, dict):
        P_full, _ = get_virtual_view_trajectory(info_or_pos["pos"], info_or_pos["t"], info_or_pos.get("vel"), target_y=PITCH_L - 0.08)
        if bounce_pos is None and info_or_pos.get("bounce"):
            bounce_pos = info_or_pos["bounce"]["pos"]
        if ball_pos is None:
            ball_pos = P_full[-1]
        if visible_idx is None:
            visible_idx = len(P_full)
    else:
        P_full = np.asarray(info_or_pos, float)

    # 1. Base stadium outfield
    if bg_template is not None and bg_template.shape[:2] == (h, w):
        img = bg_template.copy()
    else:
        img = make_turf_background(size)

    # 2. Pitch turf
    hw = 1.525
    pitch_poly = np.array([[-hw, -1.2, 0], [hw, -1.2, 0], [hw, PITCH_L + 1.2, 0], [-hw, PITCH_L + 1.2, 0]])
    p_px = project_polygon(pitch_poly, rv, C, f, size)
    if p_px is not None:
        cv2.fillPoly(img, [p_px], (168, 182, 184))
        cv2.polylines(img, [p_px], True, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. Subtle turf roller lines across pitch
    for y_strip in np.arange(0.5, PITCH_L, 1.5):
        seg = project_segments(np.array([[-hw + 0.05, y_strip, 0.001], [hw - 0.05, y_strip, 0.001]]), rv, C, f, size)
        for p0, p1 in seg:
            cv2.line(img, p0, p1, (158, 172, 174), 1, cv2.LINE_AA)

    # 4. Translucent central delivery guide corridor (from bowler stumps Y=0.0 to batter stumps Y=PITCH_L)
    strip = np.array([[-0.18, 0.0, 0.002], [0.18, 0.0, 0.002], [0.18, PITCH_L, 0.002], [-0.18, PITCH_L, 0.002]])
    s_px = project_polygon(strip, rv, C, f, size)
    if s_px is not None:
        ov = img.copy()
        cv2.fillPoly(ov, [s_px], BLUE)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)

    # 5. Creases (Authentic ICC/MCC markings in virtual view)
    draw_cricket_creases(img, rv, C, f, size, WHITE, 2)

    # 6. Bounce marker on turf
    if bounce_pos is not None:
        pt_b, _ = project_point([bounce_pos[0], bounce_pos[1], 0.003], rv, C, f, size)
        if pt_b is not None:
            ub, vb = int(round(pt_b[0])), int(round(pt_b[1]))
            cv2.circle(img, (ub, vb), 5, (30, 30, 230), -1, cv2.LINE_AA)
            cv2.circle(img, (ub, vb), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # 7. Trajectory Line
    if visible_idx is not None and visible_idx > 1:
        P_sub = P_full[:visible_idx]
        segs = project_segments(P_sub, rv, C, f, size)
        for p0, p1 in segs:
            cv2.line(img, p0, p1, (30, 30, 230), 4, cv2.LINE_AA)

    # 8. Clean White Ball
    if ball_pos is not None:
        draw_ball(img, ball_pos, rv, C, f, size)

    # 9. Stumps, bails, and directional drop shadows ON TOP (ensures 100% full visibility with zero trajectory overlap)
    for Y_s in [0.0, PITCH_L]:
        dist_s = abs(C[1] - Y_s)
        thick = 4 if dist_s < 10.0 else 2
        for xs in STUMP_X:
            p_base = np.array([xs, Y_s, 0.001])
            p_sh = np.array([xs - 0.25, Y_s + 0.45, 0.001])
            sh_seg = project_segments(np.array([p_base, p_sh]), rv, C, f, size)
            for p0, p1 in sh_seg:
                cv2.line(img, p0, p1, (50, 60, 50), 3, cv2.LINE_AA)

        for xs in STUMP_X:
            seg = project_segments(np.array([[xs, Y_s, 0.0], [xs, Y_s, STUMP_H]]), rv, C, f, size)
            for p0, p1 in seg:
                cv2.line(img, p0, p1, WOOD, thick, cv2.LINE_AA)
                cv2.circle(img, p1, 2, DARK_WOOD, -1)
        seg_b = project_segments(np.array([[STUMP_X[0], Y_s, STUMP_H], [STUMP_X[-1], Y_s, STUMP_H]]), rv, C, f, size)
        for p0, p1 in seg_b:
            cv2.line(img, p0, p1, WOOD, max(1, thick - 1), cv2.LINE_AA)

    # 10. Sleek metrics cards (top-left, no dashboard buttons or header)
    draw_metrics_cards(img, metrics, sf_scale=sf_scale)
    return img


def render_overlay_video(video, cal, fit, metrics, info, fps, out_path, sf_scale=None,
                         include_virtual_view=True, delivery_label=None):
    cap = cv2.VideoCapture(video); size = tuple(cal["size"])
    tmp_path = out_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    f0 = int(fit["frames"][0]); T, P = info["t"], info["pos"]
    by = metrics["bounce_xy_m"][1] if "bounce_xy_m" in metrics else None

    # Delay metrics appearance until the bowl is bowled (bounce candidate frame)
    bounce_candidate = fit.get("bounce_candidate", fit["frames"][-1])
    show_metrics_frame = int(bounce_candidate)

    # 1. Section 1: Real bowling clip with live ball tracking & delayed metrics
    i = 0
    last_clip_frame = int(fit["frames"][-1]) + 8
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i <= last_clip_frame:
            _pitch_geometry(fr, cal["rvec"], cal["C"], cal["f"], size, bounce_y=by)
            if i >= f0:
                upto = (i - f0) / fps
                sel = T <= upto
                if sel.sum() > 1:
                    cv2.polylines(fr, [_pts(P[sel], cal["rvec"], cal["C"], cal["f"], size)], False, RED, 3, cv2.LINE_AA)
                    ball_pt = tuple(_pts(P[sel][-1:], cal["rvec"], cal["C"], cal["f"], size)[0])
                    cv2.circle(fr, ball_pt, 6, (255, 255, 255), -1)
                    cv2.circle(fr, ball_pt, 6, (50, 50, 50), 1, cv2.LINE_AA)

            if i >= show_metrics_frame:
                draw_metrics_cards(fr, metrics, sf_scale=sf_scale)

            vw.write(fr)
        i += 1
    cap.release()

    # 2. Section 2: Virtual View Replay
    # Phase 1: Front View hold while trajectory completes at natural delivery speed + brief hold
    # Phase 2: Pan to Side View & hold
    # Phase 3: Pan to Batter End & hold & end (no return)
    if include_virtual_view:
        v_duration_sec = 5.2
        total_v_frames = int(v_duration_sec * fps)
        bg_template = make_turf_background(size)

        P_full, _ = get_virtual_view_trajectory(info["pos"], info["t"], info.get("vel"), target_y=PITCH_L - 0.08)
        b_pos = info["bounce"]["pos"] if info.get("bounce") else None
        if b_pos is not None:
            b_idx = int(np.argmin(np.linalg.norm(P_full - b_pos, axis=1)))
        else:
            b_idx = len(P_full) // 2

        M = len(P_full)

        def get_replay_phi(tau):
            if tau <= 0.30:
                return 0.0
            elif tau <= 0.52:
                w = (tau - 0.30) / 0.22
                return 90.0 * 0.5 * (1.0 - np.cos(np.pi * w))
            elif tau <= 0.65:
                return 90.0
            elif tau <= 0.86:
                w = (tau - 0.65) / 0.21
                return 90.0 + 90.0 * 0.5 * (1.0 - np.cos(np.pi * w))
            else:
                return 180.0

        for fi in range(total_v_frames):
            tau = fi / float(total_v_frames - 1)
            phi = get_replay_phi(tau)

            # Natural delivery speed: flight finishes by tau = 0.16 (~0.85s), then ball holds at stumps!
            if tau <= 0.16:
                u = tau / 0.16
                idx = int(round(u * (M - 1)))
            else:
                idx = M - 1

            idx = np.clip(idx, 0, M - 1)
            ball_p = P_full[idx]
            bnc_marker = b_pos if idx >= b_idx else None

            v_fr = render_virtual_3d_frame(phi, metrics, P_full, ball_pos=ball_p,
                                           visible_idx=idx + 1, bounce_pos=bnc_marker,
                                           size=size, sf_scale=sf_scale, cal=cal,
                                           bg_template=bg_template)
            vw.write(v_fr)

    vw.release()

    # 3. Transcode to WhatsApp-compliant H.264 (yuv420p) + AAC audio + +faststart
    ffmpeg_bin = shutil.which("ffmpeg") or "C:/ffmpeg/bin/ffmpeg.exe"
    if os.path.exists(ffmpeg_bin) or shutil.which("ffmpeg"):
        cmd = [
            ffmpeg_bin, "-y",
            "-i", tmp_path,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-shortest",
            out_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and os.path.exists(out_path):
            try: os.remove(tmp_path)
            except Exception: pass
            return
    if os.path.exists(tmp_path):
        if os.path.exists(out_path):
            try: os.remove(out_path)
            except Exception: pass
        os.rename(tmp_path, out_path)


def render_virtual_view(cal, fit, metrics, info, out_png, sf_scale=None, size=(720, 1280),
                        phi_deg=0.0, delivery_label=None):
    img = render_virtual_3d_frame(phi_deg, metrics, info, size=size, sf_scale=sf_scale, cal=cal)
    cv2.imwrite(out_png, img)
