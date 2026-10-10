#!/usr/bin/env python3
"""Calibrate camera path: map static image -> each sampled video frame.

Method: pick small high-texture patches from the static image (on stable
structures: storefront masonry, window frames, ground line), template-match
each patch into every video frame, then least-squares fit x' = s*x + tx,
y' = s*y + ty (uniform scale + translation, no rotation).
"""
import cv2
import numpy as np
import json
import sys

STATIC = "work/source_static.jpg"
FRAMES = {1: 0, 2: 20, 3: 40, 4: 60, 5: 80, 6: 100, 7: 121}  # file -> frame idx

img = cv2.imread(STATIC)
H, W = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Patches on structures that do not move within the scene (avoid canopy,
# leaves, girl). Coordinates chosen from the static image: brick wall edges,
# storefront cornice, door frames, ground pavement lines.
PATCHES = [
    (95, 700, 60),    # trunk base / ground left
    (250, 640, 60),   # brick wall right edge
    (210, 880, 60),   # cornice left
    (330, 900, 55),   # cornice right
    (250, 1050, 60),  # window/door left
    (390, 1100, 55),  # door right stile
    (150, 1230, 60),  # ground pavement left
    (520, 1235, 55),  # ground pavement right
    (300, 760, 55),   # brick mid
    (170, 820, 50),   # brick left
]

def match_frame(frame_gray):
    corr = []
    for (cx, cy, r) in PATCHES:
        tpl = gray[cy - r:cy + r, cx - r:cx + r]
        if tpl.shape[0] < 20 or tpl.shape[1] < 20:
            continue
        # search in a generous window around the expected position
        exp_x, exp_y = int(cx), int(cy)
        x0 = max(0, exp_x - r - 60); x1 = min(W, exp_x + r + 60)
        y0 = max(0, exp_y - r - 60); y1 = min(H, exp_y + r + 60)
        region = frame_gray[y0:y1, x0:x1]
        res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv < 0.45:
            continue
        px = x0 + maxloc[0] + r  # matched center
        py = y0 + maxloc[1] + r
        corr.append(((cx, cy), (px, py), maxv))
    return corr

def fit(corr):
    # keep best 6+ matches, drop worst score if enough
    corr = sorted(corr, key=lambda c: -c[2])
    if len(corr) > 7:
        corr = corr[:7]
    src = np.array([c[0] for c in corr], dtype=np.float64)
    dst = np.array([c[1] for c in corr], dtype=np.float64)
    # solve [x y 1 0; 0 x y 1] * [s tx s' ty'] = [x'; y'] with shared s
    A = np.zeros((2 * len(corr), 3))
    b = np.zeros(2 * len(corr))
    A[0::2, 0] = src[:, 0]; A[0::2, 1] = 1.0
    A[1::2, 0] = src[:, 1]; A[1::2, 2] = 1.0
    b[0::2] = dst[:, 0]; b[1::2] = dst[:, 1]
    # unknowns: s, tx, ty  (x' = s*x + tx ; y' = s*y + ty)
    A2 = np.zeros((2 * len(corr), 3))
    A2[0::2, 0] = src[:, 0]; A2[0::2, 1] = 1.0
    A2[1::2, 0] = src[:, 1]; A2[1::2, 2] = 1.0
    sol, res, rank, _ = np.linalg.lstsq(A2, b, rcond=None)
    s, tx, ty = sol
    pred = np.column_stack([s * src[:, 0] + tx, s * src[:, 1] + ty])
    err = np.sqrt(((pred - dst) ** 2).sum(axis=1))
    return float(s), float(tx), float(ty), float(err.mean()), len(corr)

out = {}
for fidx, fnum in sorted(FRAMES.items()):
    fr = cv2.imread(f"work/vf_{fidx}.png")
    if fr is None:
        print(f"missing vf_{fidx}.png"); sys.exit(1)
    fgray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    corr = match_frame(fgray)
    if len(corr) < 4:
        print(f"frame {fnum}: only {len(corr)} matches, FAIL")
        out[fnum] = None
        continue
    s, tx, ty, err, n = fit(corr)
    out[fnum] = {"scale": s, "tx": tx, "ty": ty, "reproj_err": round(err, 2), "matches": n}
    print(f"frame {fnum:3d}: scale={s:.4f} tx={tx:7.2f} ty={ty:7.2f} err={err:.2f}px n={n}")

with open("work/camera_path.json", "w") as f:
    json.dump(out, f, indent=1)

# Fit linear model scale(t), tx(t), ty(t) over valid frames
valid = sorted([(k, v) for k, v in out.items() if v])
t = np.array([k for k, _ in valid], dtype=np.float64)
for key in ("scale", "tx", "ty"):
    y = np.array([v[key] for _, v in valid])
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = np.abs(A @ coef - y).max()
    print(f"linear {key}(t) = {coef[0]:.6f}*t + {coef[1]:.4f}   (max resid {resid:.2f})")
