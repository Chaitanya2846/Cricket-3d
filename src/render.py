import subprocess, shutil, os
import cv2, numpy as np
from geometry import *

RED = (40, 40, 220); BLUE = (230, 130, 150); WHITE = (255, 255, 255); YEL = (60, 210, 240)


def _pts(P, rvec, C, f, size):
    return np.round(project(P, rvec, C, f, size)).astype(np.int32)


def _pitch_geometry(img, rvec, C, f, size, fill=None, bounce_y=None):
    hw = PITCH_HW
    poly = np.array([[-hw, 0, 0], [hw, 0, 0], [hw, PITCH_L, 0], [-hw, PITCH_L, 0]])
    if fill is not None:
        ov = img.copy(); cv2.fillPoly(ov, [_pts(poly, rvec, C, f, size)], fill)
        cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    cv2.polylines(img, [_pts(poly, rvec, C, f, size)], True, WHITE, 1, cv2.LINE_AA)
    for y in (0, POP, PITCH_L - POP, PITCH_L):                       # bowling/popping creases
        cv2.polylines(img, [_pts(np.array([[-RET, y, 0], [RET, y, 0]]), rvec, C, f, size)], False, WHITE, 1, cv2.LINE_AA)
    for Y in (0.0, PITCH_L):
        for x in STUMP_X:
            cv2.line(img, *map(tuple, _pts(np.array([[x, Y, 0], [x, Y, STUMP_H]]), rvec, C, f, size)), YEL, 2, cv2.LINE_AA)
    if bounce_y is not None:                                         # blue line bounce -> stumps
        strip = np.array([[-.15, bounce_y, 0], [.15, bounce_y, 0], [.15, PITCH_L, 0], [-.15, PITCH_L, 0]])
        ov = img.copy(); cv2.fillPoly(ov, [_pts(strip, rvec, C, f, size)], BLUE)
        cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)


def _card(img, x, y, label, value, w=150, h=52):
    ov = img.copy(); cv2.rectangle(ov, (x, y), (x + w, y + h), (30, 40, 30), -1)
    cv2.addWeighted(ov, 0.7, img, 0.3, 0, img); cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(img, label, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, value, (x + 8, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)


def _cards(img, m, scale=1.0, sf_scale=None):
    sw = m.get("swing_dev_cm", 0.0) * (sf_scale if sf_scale else 1.0)
    unit = "sf" if sf_scale else "cm"
    _card(img, 12, 12, "Swing", f"{sw:.1f} {unit}")
    _card(img, 12, 72, "Spin", f"{m.get('spin_deg', 0.0):.1f} deg")
    _card(img, 12, 132, "Speed", f"{m['speed_mph']:.0f} mph")


def render_overlay_video(video, cal, fit, metrics, info, fps, out_path, sf_scale=None):
    cap = cv2.VideoCapture(video); size = tuple(cal["size"])
    tmp_path = out_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    f0 = int(fit["frames"][0]); T, P = info["t"], info["pos"]; i = 0
    by = metrics["bounce_xy_m"][1] if "bounce_xy_m" in metrics else None
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i <= int(fit["frames"][-1]) + 3:
            _pitch_geometry(fr, cal["rvec"], cal["C"], cal["f"], size, bounce_y=by)
            if i >= f0:
                upto = (i - f0) / fps; sel = T <= upto
                if sel.sum() > 1:
                    cv2.polylines(fr, [_pts(P[sel], cal["rvec"], cal["C"], cal["f"], size)], False, RED, 2, cv2.LINE_AA)
            _cards(fr, metrics, sf_scale=sf_scale)
        vw.write(fr); i += 1
    vw.release()

    # Convert to WhatsApp-compliant H.264 (yuv420p) + AAC audio + faststart
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
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



def render_virtual_view(cal, fit, metrics, info, out_png, sf_scale=None, size=(540, 960)):
    img = np.full((size[1], size[0], 3), (70, 120, 60), np.uint8)
    C = np.array([0.0, -6.0, 5.0]); rvec = lookat_rvec(C, [0, 11.0, 0.0]); f = 850.0
    by = metrics["bounce_xy_m"][1] if "bounce_xy_m" in metrics else None
    _pitch_geometry(img, rvec, C, f, size, fill=(150, 175, 185), bounce_y=by)
    cv2.polylines(img, [_pts(info["pos"], rvec, C, f, size)], False, RED, 3, cv2.LINE_AA)
    _cards(img, metrics, sf_scale=sf_scale)
    cv2.imwrite(out_png, img)
