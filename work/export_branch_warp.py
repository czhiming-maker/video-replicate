#!/usr/bin/env python3
"""Branch sway — measured rigid per-band displacement, applied as a
feathered warp over the branch pixels only.

Method: per frame, camera-compensate the source to plate coords, Farneback
flow vs f0, then take the MEDIAN flow over eroded branch pixels per height
band (flying leaves crossing the region are outliers the median rejects;
the static wall is excluded by the branch mask).  Result: 3 bands x (dx,dy)
per frame — a cantilever sway profile (amplitude grows away from the base).
The exported warp displaces only feathered branch pixels; the wall never
moves.  Amplitudes clamped (dissolve/corruption guard) and smoothed over
time.  Deterministic.

Output: public/seq/branch/warp_f***.png (RGBA crop, plate-aligned) +
work/branch_meta.json.  Scene: place above plate, below canopy seq.
"""
import cv2
import numpy as np
import json
import os

H, W = 1280, 720
FR = [0, 20, 40, 60, 80, 100, 121]
TX = [0, -50, -105, -159, -212, -272, -336]
tx_of = np.interp(np.arange(122), FR, TX)

X0, Y0, X1, Y1 = 0, 380, 470, 1280
CW, CH = X1 - X0, Y1 - Y0
BANDS = [(380, 620), (620, 940), (940, 1120)]  # 上/中/下（y>1120 是地面落叶堆，静止）
AMP_CLAMP = 2.5     # px per band — guards against measurement spikes
MIN_PIX = 400       # need enough branch pixels in a band to trust the median

os.makedirs("public/seq/branch", exist_ok=True)

def plate_aligned(f):
    src = cv2.imread(f"work/frames/f{f+1:03d}.png")
    return cv2.warpAffine(src, np.float32([[1, 0, -tx_of[f]], [0, 1, 0]]), (W, H))

base_gray = cv2.cvtColor(plate_aligned(0), cv2.COLOR_BGR2GRAY)
base_roi = base_gray[Y0:Y1, X0:X1]
branch0 = (base_roi < 120).astype(np.uint8)
branch0 = cv2.morphologyEx(branch0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
# eroded core: pixels safely ON the branch (not its soft edge)
core = cv2.erode(branch0, np.ones((5, 5), np.uint8))
# feather weight: branch mask dilated + blurred — the warp's spatial support
weight = cv2.GaussianBlur(cv2.dilate(branch0, np.ones((7, 7), np.uint8)).astype(np.float32), (9, 9), 3)
weight = np.clip(weight, 0, 1)

# static canopy exclusion (dissolving leaves corrupt the flow)
canopy_a = cv2.imread("public/canopy.png", cv2.IMREAD_UNCHANGED)[:, :, 3]
canopy_free = np.zeros((H, W), np.uint8)
# canopy.png is a plate-coords layer at (x=0, y=63), 688x900 — place it on
# the full-frame canvas before slicing the branch region
canopy_free[63:63 + canopy_a.shape[0], 0:0 + canopy_a.shape[1]] = (
    (cv2.dilate((canopy_a > 30).astype(np.uint8),
                np.ones((9, 9), np.uint8)) == 0).astype(np.uint8))
canopy_free_roi = canopy_free[Y0:Y1, X0:X1]

band_masks = []
for by0, by1 in BANDS:
    bm = np.zeros((CH, CW), np.uint8)
    bm[max(0, by0 - Y0):min(CH, by1 - Y0), :] = 255
    band_masks.append(bm)

flows = {}
for f in range(122):
    g = cv2.cvtColor(plate_aligned(f), cv2.COLOR_BGR2GRAY)
    g_roi = np.float32(g[Y0:Y1, X0:X1])
    flow = cv2.calcOpticalFlowFarneback(
        base_roi, g_roi, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    fx = np.clip(flow[..., 0], -15, 15)
    fy = np.clip(flow[..., 1], -15, 15)
    vis = np.zeros((CH, CW), np.uint8)
    vis[:, int(round(abs(tx_of[f]))):] = 255  # black-band guard
    band_disp = []
    for bi, bm in enumerate(band_masks):
        sel = (core > 0) & (bm > 0) & (canopy_free_roi > 0) & (vis > 0)
        if sel.sum() < MIN_PIX:
            band_disp.append(None)  # band not measurable this frame
            continue
        mdx = float(np.median(fx[sel]))
        mdy = float(np.median(fy[sel]))
        band_disp.append((float(np.clip(mdx, -AMP_CLAMP, AMP_CLAMP)),
                          float(np.clip(mdy, -AMP_CLAMP, AMP_CLAMP))))
    flows[f] = band_disp

# temporal smoothing (5-frame median) per band; unmeasurable frames carry
# the previous smoothed value (the sway is slow — hold is safe)
sm = [[(0.0, 0.0)] * 3 for _ in range(122)]
for bi in range(3):
    last = (0.0, 0.0)
    for f in range(122):
        vals = [flows[ff][bi] for ff in range(max(0, f - 2), min(122, f + 3))
                if flows[ff][bi] is not None]
        cur = (float(np.median([v[0] for v in vals])),
               float(np.median([v[1] for v in vals]))) if vals else last
        sm[f][bi] = cur
        last = cur

# amplitude sanity report
for bi, (by0, by1) in enumerate(BANDS):
    xs = [sm[f][bi][0] for f in range(122)]
    ys = [sm[f][bi][1] for f in range(122)]
    print(f"band{bi} (y{by0}-{by1}): dx [{min(xs):+.2f},{max(xs):+.2f}] "
          f"dy [{min(ys):+.2f},{max(ys):+.2f}]")

# export warped RGBA crops
meta = {"x": X0, "y": Y0, "w": CW, "h": CH,
        "bands": BANDS, "method": "median-flow-per-band, clamped +/-2.5px"}
for f in range(122):
    fx_full = np.zeros((CH, CW), np.float32)
    fy_full = np.zeros((CH, CW), np.float32)
    # piecewise-constant bands, blended over 80px at the seams
    wsum = np.zeros((CH, CW), np.float32)
    for bi, (by0, by1) in enumerate(BANDS):
        w = np.zeros((CH, CW), np.float32)
        y_lo = max(0, by0 - Y0 - 40)
        y_hi = min(CH, by1 - Y0 + 40)
        w[y_lo:y_hi, :] = 1.0
        w = cv2.GaussianBlur(w, (0, 0), 20)
        dx, dy = sm[f][bi]
        fx_full += dx * w
        fy_full += dy * w
        wsum += w
    fx_full /= np.maximum(wsum, 1e-6)
    fy_full /= np.maximum(wsum, 1e-6)
    disp_x = fx_full * weight
    disp_y = fy_full * weight
    ym, xm = np.mgrid[0:CH, 0:CW]
    map_x = (xm + disp_x).astype(np.float32)
    map_y = (ym + disp_y).astype(np.float32)
    warped = cv2.remap(plate_aligned(f)[Y0:Y1, X0:X1], map_x, map_y,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    alpha = (weight * 255).astype(np.uint8)
    # black-band guard: hide the plate part frame f's camera can't see
    vis = np.zeros((CH, CW), np.uint8)
    vis[:, int(round(abs(tx_of[f]))):] = 255
    alpha[vis == 0] = 0
    rgba = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    if not cv2.imwrite(f"public/seq/branch/warp_f{f:03d}.png", rgba):
        raise RuntimeError(f"imwrite failed f{f}")
with open("work/branch_meta.json", "w") as fj:
    json.dump(meta, fj)
print("branch warp sequence written:", CW, "x", CH, "x 122")
