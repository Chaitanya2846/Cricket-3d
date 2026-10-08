"""Detect stumps on the raw-footage frames and aggregate into near/far sets.
Output: data/interim/stumps.json  {near: [[cx, y_top, y_bottom] x3], far: [...]} (pixels, left->right)."""
import argparse, json
import numpy as np, cv2
from ultralytics import YOLO
from geometry import load_cfg


def detect(video, model_path, f0, f1, conf):
    model = YOLO(model_path); cap = cv2.VideoCapture(video); dets = []; i = 0
    while True:
        ok, fr = cap.read()
        if not ok or i >= f1: break
        if i >= f0:
            r = model.predict(fr, conf=conf, verbose=False)[0]
            boxes = [[*map(float, xy), float(c)] for xy, c in
                     zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            dets.append({"frame": i, "boxes": boxes})
        i += 1
    return dets


def aggregate(dets):
    hs = np.array([b[3] - b[1] for d in dets for b in d["boxes"]])
    if len(hs) == 0: raise SystemExit("No stumps detected. Lower stump_conf or check the frame range.")
    lh = np.sort(np.log(hs)); gaps = np.diff(lh); thr = None
    if len(gaps) and gaps.max() > np.log(1.8):                 # near stumps are much taller than far ones
        j = gaps.argmax(); thr = float(np.exp((lh[j] + lh[j + 1]) / 2))
    per = {"near": [], "far": []}
    for d in dets:
        bs = d["boxes"]
        groups = {"near": bs, "far": []} if thr is None else {
            "near": [b for b in bs if b[3] - b[1] >= thr], "far": [b for b in bs if b[3] - b[1] < thr]}
        for k, g in groups.items():
            if len(g) == 3:
                # 3 individual stump boxes
                st_list = []
                for b in sorted(g, key=lambda x: x[0]):
                    cx = float((b[0] + b[2]) / 2.0)
                    yt = float(b[1])
                    yb = float(b[3] - 0.05 * (b[3] - b[1]))
                    st_list.append([cx, yt, yb])
                per[k].append(st_list)
            elif len(g) == 1:
                # 1 box covering all 3 stumps of the wicket
                b = g[0]
                w = b[2] - b[0]
                yt = float(b[1])
                yb = float(b[3] - 0.05 * (b[3] - b[1]))
                cx0 = float(b[0] + 0.16 * w)
                cx1 = float((b[0] + b[2]) / 2.0)
                cx2 = float(b[2] - 0.16 * w)
                per[k].append([[cx0, yt, yb], [cx1, yt, yb], [cx2, yt, yb]])
    out = {"height_split_px": thr}
    for k, v in per.items():
        if len(v) >= 1:
            a = np.median(np.array(v), axis=0)                  # (3,3): cx, yt, yb
            out[k] = [[float(cx), float(yt), float(yb)] for cx, yt, yb in a]
            out[k + "_frames_used"] = len(v)
    return out



def save_check_image(video, stumps, out_png):
    cap = cv2.VideoCapture(video); ok, fr = cap.read()
    for k, col in (("near", (0, 255, 0)), ("far", (0, 200, 255))):
        for cx, yt, yb in stumps.get(k, []):
            cv2.circle(fr, (int(cx), int(yt)), 3, col, -1); cv2.circle(fr, (int(cx), int(yb)), 3, (0, 0, 255), -1)
            cv2.line(fr, (int(cx), int(yt)), (int(cx), int(yb)), col, 1)
    cv2.imwrite(out_png, fr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
    cfg = load_cfg(a.config)
    dets = detect(cfg["video"], cfg["model"], cfg["frames"][0], cfg["frames"][1], cfg["stump_conf"])
    st = aggregate(dets)
    json.dump(st, open("data/interim/stumps.json", "w"), indent=2)
    save_check_image(cfg["video"], st, "outputs/stump_check.png")
    print(json.dumps(st, indent=2)); print("-> data/interim/stumps.json, outputs/stump_check.png")
