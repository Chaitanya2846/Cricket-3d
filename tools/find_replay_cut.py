"""Prints the frame where the footage changes abruptly (raw video -> app 3D replay)."""
import sys, cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); prev = None; diffs = []
while True:
    ok, fr = cap.read()
    if not ok: break
    g = cv2.cvtColor(cv2.resize(fr, (90, 160)), cv2.COLOR_BGR2GRAY).astype(float)
    diffs.append(0 if prev is None else np.abs(g - prev).mean()); prev = g
d = np.array(diffs); k = int(d.argmax())
print("largest change at frame", k, "(set frames: [0,", k, "])  diff =", round(d[k], 1))
