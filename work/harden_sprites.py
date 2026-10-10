#!/usr/bin/env python3
"""Harden sprite alphas + fill transparent-zone RGB with the leaf's own color.

Partial-alpha pixels carry paper RGB from the source crop; when sprites are
scaled up and blurred (bokeh / near layer) that paper RGB turns into gray
halos.  Hardening alpha and repainting low-alpha RGB with the leaf median
color makes blurred composites stay orange.
"""
import cv2
import numpy as np
import glob

for path in sorted(glob.glob("public/sprites/*.png")):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None or im.shape[2] != 4:
        continue
    a = im[:, :, 3].astype(np.float32) / 255.0
    a2 = np.clip((a - 0.5) / (0.85 - 0.5), 0, 1)
    a2 = a2 * a2 * (3 - 2 * a2)  # smoothstep
    # strict leaf-color median for RGB fill; if the sprite has no strictly
    # leaf-colored pixels (muted/dark leaves), fall back to its own opaque
    # pixels — never skip, or paper-gap RGB survives as pale rings
    hsv = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2HSV)
    strict = (((hsv[:, :, 0] >= 4) & (hsv[:, :, 0] <= 30)) &
              (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 70))
    if strict.sum() >= 10:
        fill = np.median(im[strict][:, :3].astype(np.float32), axis=0)
    else:
        opaque = a > 0.9
        if opaque.sum() < 5:
            continue
        fill = np.median(im[opaque][:, :3].astype(np.float32), axis=0)
    rgb = im[:, :, :3].astype(np.float32)
    # any pixel that is not essentially opaque carries paper RGB from the
    # source crop (leaf gaps, feathered rims) — repaint it with the leaf
    # median so scaled/blurred composites never show pale rings
    # partial-alpha pixels: fill color AND clamp lightness to the fill's
    # level so feathered rims never read as pale halos on dark backgrounds
    fill_lum = (0.299 * fill[2] + 0.587 * fill[1] + 0.114 * fill[0])
    soft = (a2 > 0.02) & (a2 < 0.98)
    solid = (a2 >= 0.98)[..., None]
    rgb = rgb * solid + fill[None, None, :] * (1 - solid)
    if soft.any():
        pl = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        cap = fill_lum + 10
        over = soft & (pl > cap)
        g = np.where(pl > 1e-3, cap / np.maximum(pl, 1e-3), 1.0)
        g = np.where(over, g, 1.0)[..., None]
        rgb = rgb * g
    out = im.copy()
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(a2 * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)
print("hardened all sprites")
