#!/usr/bin/env python3
"""Hero-leaf per-frame matte sequences — SAM2 video segmentation + measured
centroids, exported per COHERENT RUN.

SAM2's identity drifts across the busy late-video field (red leaves merge
with the girl / with each other), so raw tracks are segmented into runs
(gap > 2f or centroid jump > 90px breaks a run) and only verified runs are
exported.  Every exported frame is hard-checked against the girl zone:
overlap > 50px rejects the frame, small fringe pixels are zeroed.

Output: public/seq/hero/<name>/f***.png, src/hero_seq.json, work/hero_meta.json
"""
import cv2
import numpy as np
import json
import os

H, W = 1280, 720
FR = [0, 20, 40, 60, 80, 100, 121]
TX = [0, -50, -105, -159, -212, -272, -336]
tx_of = np.interp(np.arange(122), FR, TX)

MASK_DIR = "work/sam2_masks"
OUT_ROOT = "public/seq/hero"
PAD = 12
MIN_COMPONENT = 8
GIRL_PLATE = (480, 1000, 660, 1200)  # x0, y0, x1, y1 (plate coords)
GIRL_MAX_OVERLAP = 50

def girl_view_mask(f):
    """girl keep-out zone in VIEW coords for frame f (the girl is static in
    plate coords; the camera pans, so the view-coords zone shifts by tx)"""
    g = np.zeros((H, W), np.uint8)
    x_off = int(round(tx_of[f]))
    gx0 = max(0, GIRL_PLATE[0] + x_off)
    gx1 = min(W, GIRL_PLATE[2] + x_off)
    g[GIRL_PLATE[1]:GIRL_PLATE[3], gx0:gx1] = 255
    return g

# (sam2 obj id, run start, run end, scene name) — runs from the coherence
# audit; junk/teleport segments and the girl-contaminated obj3 are excluded
RUNS = [
    (1, 26, 109, "heroA"),
    (2, 77, 95, "heroD"),
    (4, 75, 110, "heroE"),
    (5, 40, 62, "heroF"),
    (5, 99, 121, "heroG"),
]

meta_out = {}
os.makedirs(OUT_ROOT, exist_ok=True)
kept_names = set()
for obj, rf0, rf1, name in RUNS:
    outdir = f"{OUT_ROOT}/{name}"
    os.makedirs(outdir, exist_ok=True)
    frames = {}
    rejected = 0
    for f in range(rf0, rf1 + 1):
        p = f"{MASK_DIR}/obj{obj}_f{f:03d}.png"
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None or (m > 0).sum() == 0:
            continue
        # component cleanup: drop specks and satellites flung far from the
        # biggest component (SAM2 occasionally grabs an unrelated leaf)
        nn, ll, ss, cent = cv2.connectedComponentsWithStats(
            (m > 0).astype(np.uint8), connectivity=8)
        comps = [(ss[i, 0], ss[i, 1], ss[i, 2], ss[i, 3], ss[i, 4], i)
                 for i in range(1, nn) if ss[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT]
        if not comps:
            continue
        big = max(comps, key=lambda c: c[4])
        bcx, bcy = big[0] + big[2] / 2, big[1] + big[3] / 2
        clean = np.zeros_like(m)
        for c in comps:
            if np.hypot(c[0] + c[2] / 2 - bcx, c[1] + c[3] / 2 - bcy) < 110:
                clean[ll == c[5]] = 255
        m = clean
        # girl guard: reject heavily contaminated frames, zero the fringe
        gv = girl_view_mask(f)
        ov = int(((m > 0) & (gv > 0)).sum())
        if ov > GIRL_MAX_OVERLAP:
            rejected += 1
            continue
        m[(gv > 0)] = 0
        if (m > 0).sum() == 0:
            rejected += 1
            continue
        ys, xs = np.where(m > 0)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        rw = min(W, x1 - x0 + 1 + 2 * PAD)
        rh = min(H, y1 - y0 + 1 + 2 * PAD)
        ccx, ccy = (x0 + x1) / 2, (y0 + y1) / 2
        rx0 = int(np.clip(round(ccx - rw / 2), 0, W - rw))
        ry0 = int(np.clip(round(ccy - rh / 2), 0, H - rh))
        src = cv2.imread(f"work/frames/f{f+1:03d}.png")
        rgba = cv2.cvtColor(src[ry0:ry0 + rh, rx0:rx0 + rw], cv2.COLOR_BGR2BGRA)
        a = m[ry0:ry0 + rh, rx0:rx0 + rw].astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), 0.8)
        rgba[:, :, 3] = np.clip(a * 255, 0, 255).astype(np.uint8)
        if not cv2.imwrite(f"{OUT_ROOT}/{name}/f{f:03d}.png", rgba):
            raise RuntimeError(f"imwrite failed: {name}/f{f:03d}.png")
        frames[f] = {
            "cx": float(rx0 + rw / 2 - tx_of[f]),  # plate coords
            "cy": float(ry0 + rh / 2),
            "w": int(rw), "h": int(rh),
        }
    life = sorted(frames)
    if len(life) < 5:
        print(f"{name}: only {len(life)} frames survived the girl guard — skipped")
        continue
    kept_names.add(name)
    meta_out[name] = {"f0": life[0], "f1": life[-1], "pos": frames,
                      "girl_rejected_frames": rejected}
    print(f"{name}: f{life[0]}-f{life[-1]}, {len(life)} frames "
          f"({rejected} girl-rejected)")

with open("work/hero_meta.json", "w") as fj:
    json.dump(meta_out, fj)
scene = json.loads(json.dumps(meta_out, default=int))
with open("src/hero_seq.json", "w") as fj:
    json.dump(scene, fj)

# prune sequence dirs that are no longer exported
for d in os.listdir(OUT_ROOT):
    if d not in kept_names:
        import shutil
        shutil.rmtree(f"{OUT_ROOT}/{d}")
        print("pruned stale sequence dir:", d)
print("hero seq + src/hero_seq.json written")
