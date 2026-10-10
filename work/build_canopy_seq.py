#!/usr/bin/env python3
"""Canopy erosion sequence v2 — measured, not modeled.

For each source frame: warp back to plate coords, take the leaf-color mask
restricted to the erodible region, use it as the canopy layer's alpha.
Spatial erosion pattern is then exactly the source video's.
Also exports per-frame erosion-edge points for particle spawning.
"""
import cv2
import numpy as np
import json
import os

rng = np.random.default_rng(7)
H, W = 1280, 720
os.makedirs("public/seq/canopy", exist_ok=True)

canopy = cv2.imread("public/canopy.png", cv2.IMREAD_UNCHANGED)  # BGRA
with open("work/manifest.json") as fj:
    manifest = json.load(fj)
bx = manifest["canopy_bbox"]["x"]
by = manifest["canopy_bbox"]["y"]
ch, cw = canopy.shape[:2]

erodible = cv2.imread("work/erodible.png", cv2.IMREAD_GRAYSCALE)
erod_bin = (erodible > 128)
erod_roi = erod_bin[by:by + ch, bx:bx + cw]
prot = cv2.imread("work/protected_roi.png", cv2.IMREAD_GRAYSCALE)
prot_roi = (prot[by:by + ch, bx:bx + cw] > 128)

FR = [0, 20, 40, 60, 80, 100, 121]
TX = [0, -50, -105, -159, -212, -272, -336]
tx_of = np.interp(np.arange(122), FR, TX)

canopy_alpha0 = canopy[:, :, 3].astype(np.float32)
spawn_points = {}
remain = []
kernel_o = np.ones((3, 3), np.uint8)

for f in range(122):
    src = cv2.imread(f"work/frames/f{f+1:03d}.png")
    M = np.float32([[1, 0, -tx_of[f]], [0, 1, 0]])
    warped = cv2.warpAffine(src, M, (W, H))
    roi = warped[by:by + ch, bx:bx + cw]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # very permissive leaf test: dark maroon canopy leaves (val 30-90) must
    # count as leaves, or whole dark clusters get eroded into plate holes
    warm = ((hsv[:, :, 0] < 33) | (hsv[:, :, 0] > 167))
    perm = (warm & (hsv[:, :, 1] > 42) & (hsv[:, :, 2] > 38)).astype(np.uint8)
    lm = (perm * erod_roi.astype(np.uint8))
    # clean speckle noise from video compression: drop dark val noise and
    # small components, keep the leaf-scale structure
    lm = cv2.morphologyEx(lm, cv2.MORPH_OPEN, kernel_o)
    lm = cv2.morphologyEx(lm, cv2.MORPH_CLOSE, kernel_o)
    nn, ll, ss, _ = cv2.connectedComponentsWithStats(lm, connectivity=8)
    for i in range(1, nn):
        if ss[i, cv2.CC_STAT_AREA] < 60:
            lm[ll == i] = 0
    a_raw = cv2.GaussianBlur(lm.astype(np.float32), (0, 0), 1.1)
    # treat the video mask as RELATIVE erosion: dilate it first so
    # threshold-noise pinholes (which would expose the quilted plate as
    # pale rings) are absorbed; only clearly-lost regions erode
    lm_safe = cv2.dilate(lm, np.ones((9, 9), np.uint8))
    erosion = (erod_roi.astype(np.uint8) & (lm_safe == 0)).astype(np.float32)
    erosion = cv2.GaussianBlur(erosion, (0, 0), 0.6)
    a = np.maximum(a_raw, 1.0 - erosion)
    # protected canopy (over brick / trunk corridor) stays static: video
    # thresholds can miss dark shadowed leaves there
    a[prot_roi] = np.maximum(a[prot_roi], 1.0)
    out = canopy.copy()
    out[:, :, 3] = np.clip(canopy_alpha0 * a, 0, 255).astype(np.uint8)
    cv2.imwrite(f"public/seq/canopy/f{f:03d}.png", out)
    remain.append(float((lm > 0).sum()) / max(float(erod_roi.sum()), 1))
    # spawn edge points (plate coords): outer boundary of kept region
    grad = cv2.morphologyEx(lm, cv2.MORPH_GRADIENT, kernel_o)
    grad = grad * (lm > 0)
    eys, exs = np.where(grad > 0)
    pts = []
    if len(eys) > 0:
        idx = rng.choice(len(eys), size=min(46, len(eys)), replace=False)
        for i in idx:
            pts.append([int(exs[i] + bx), int(eys[i] + by)])
    spawn_points[int(f)] = pts
    if f % 20 == 0:
        print(f"f{f:3d}: erodible remain {remain[-1]:.3f} edge_pts={len(pts)}")

with open("work/canopy_spawn.json", "w") as fj:
    json.dump(spawn_points, fj)
# ---------- tree-top sway: measured, applied AFTER metrics ----------
# Consecutive-frame phase correlation on the canopy top strip (neighbor
# dissolve deltas are tiny), pan-compensated; clip per-step jumps, median-
# smooth, mean-removed cumsum -> per-frame offset vs f0.  Runs after
# spawn/remain are computed so those outputs stay byte-identical.
# Pan-compensation sign: warp prev by +dtx (dtx median -2.75px).  The brief's
# original -dtx ran against the actual pan and doubled it (raw |dx| p90 5.69
# ~= 2x pan); +dtx converges to sub-pixel (p90 1.12), matching the visible
# ~1-1.5px sway.
def measure_sway():
    y0, y1, x0, x1 = 70, 320, 20, 700
    prev = cv2.cvtColor(cv2.imread("work/frames/f001.png")[y0:y1, x0:x1],
                        cv2.COLOR_BGR2GRAY).astype(np.float32)
    steps = [(0.0, 0.0)]
    for f in range(1, 122):
        dtx = float(tx_of[f] - tx_of[f - 1])
        cur = cv2.imread(f"work/frames/f{f+1:03d}.png")[y0:y1, x0:x1]
        cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)
        a = cv2.warpAffine(prev, np.float32([[1, 0, +dtx], [0, 1, 0]]),
                           (x1 - x0, y1 - y0))
        (dx, dy), _ = cv2.phaseCorrelate(a, cur)
        # real sway steps are well under a pixel; dissolve/leaf events clip
        dx = float(np.clip(dx, -0.8, 0.8))
        dy = float(np.clip(dy, -0.8, 0.8))
        steps.append((dx, dy))
        prev = cur
    sm = np.array(steps, np.float32)
    k = 9
    pad = np.pad(sm, ((k // 2, k // 2), (0, 0)), mode="edge")
    sm = np.median(np.stack([pad[i:i + len(sm)] for i in range(k)]), axis=0)
    dx = np.cumsum(sm[:, 0]); dy = np.cumsum(sm[:, 1])
    dx -= dx.mean(); dy -= dy.mean()
    return dx.astype(np.float32), dy.astype(np.float32)

sway_dx, sway_dy = measure_sway()
span = float(max(abs(sway_dx).max(), abs(sway_dy).max()))
print(f"measured sway span: {span:.2f}px")
if span > 3.0:
    # measurement polluted (>3px gate) -> spec fallback: low-freq sinusoid,
    # 2 cycles over the 122-frame (5s) clip, 1.2px x-only amplitude
    print("measured curve polluted (>3px) -> sinusoidal fallback")
    sway_dx = (1.2 * np.sin(np.arange(122) / 122 * 2 * np.pi * 2)
               ).astype(np.float32)
    sway_dy = np.zeros(122, np.float32)
    sway_method = (f"sinusoidal fallback 1.2px, 2 cycles/5s "
                   f"(measured phase-corr polluted: span {span:.2f}px > 3px)")
else:
    sway_method = "consecutive phase-corr (+dtx), clip±0.8/step, " \
                  "median-9, mean-removed"
print(f"sway range: dx [{sway_dx.min():.2f},{sway_dx.max():.2f}] "
      f"dy [{sway_dy.min():.2f},{sway_dy.max():.2f}]")
for f in range(122):
    p = f"public/seq/canopy/f{f:03d}.png"
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    M = np.float32([[1, 0, sway_dx[f]], [0, 1, sway_dy[f]]])
    cv2.imwrite(p, cv2.warpAffine(im, M, (im.shape[1], im.shape[0]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101))

with open("work/canopy_meta.json", "w") as fj:
    json.dump({"remain": remain,
               "sway_dx": [round(float(v), 3) for v in sway_dx],
               "sway_dy": [round(float(v), 3) for v in sway_dy],
               "sway_method": sway_method}, fj)
print("done")
