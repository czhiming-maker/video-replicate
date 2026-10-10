#!/usr/bin/env python3
"""Rebuild the plate's right extension (x >= 700) from the source video
itself: warp every frame back into plate coords and take the temporal
median per pixel.  The AI video already hallucinated the off-image
continuation — using it makes the extension perceptually seamless with
the source by construction.  Flying leaves are transients and median out.
"""
import cv2
import numpy as np

H, W, EXT = 1280, 720, 380
X0 = 700  # overwrite from here (24px feather into the quilted fallback)

FR = [0, 20, 40, 60, 80, 100, 121]
TX = [0, -50, -105, -159, -212, -272, -336]
tx_of = np.interp(np.arange(122), FR, TX)

plate = cv2.imread("public/plate.png")

n_ext = W + EXT - X0
counts = np.zeros(n_ext, np.int32)
acc = np.zeros((1280, n_ext, 122, 3), np.uint8)  # up to 122 samples
for f in range(0, 122):
    fr = cv2.imread(f"work/frames/f{f+1:03d}.png")
    tx = float(tx_of[f])
    M = np.float32([[1, 0, -tx], [0, 1, 0]])
    warped = cv2.warpAffine(fr, M, (W + EXT, H))
    xa = max(X0, int(np.ceil(-tx)))
    xb = min(W + EXT, int(720 - tx))
    if xb <= xa:
        continue
    strip = warped[:, xa:xb]
    k = counts[xa - X0:xb - X0]
    for j in range(xb - xa):
        acc[:, xa - X0 + j, k[j]] = strip[:, j]
        k[j] += 1

med = np.zeros((1280, n_ext, 3), np.float32)
for i in range(n_ext):
    kk = counts[i]
    if kk == 0:
        med[:, i] = plate[:, X0 + i].astype(np.float32)
        continue
    samples = acc[:, i, :kk].astype(np.float32)
    if kk >= 12:
        # paper is brighter than leaves: rank samples by luminance and take
        # the 70th-percentile SAMPLE (whole pixel, no per-channel fringing)
        lum = samples[:, :, 0] * 0.299 + samples[:, :, 1] * 0.587 + samples[:, :, 2] * 0.114
        order = np.argsort(lum, axis=1)
        pick = np.clip((0.70 * (kk - 1)).astype(np.int32), 0, kk - 1)
        med[:, i] = samples[np.arange(1280)[:, None], order, :][
            np.arange(1280), pick]
    else:
        med[:, i] = np.median(samples, axis=1)
med = med.astype(np.uint8)
# persisted for build_assets.py's girl-region pave bank: the mirror-synth ext
# this script overwrites carries mirror-fold seams, so the girl repair quilts
# from this clean video-median texture instead (deterministic from frames)
cv2.imwrite("work/ext_video_median.png", med)

alpha = np.ones((H, n_ext), np.float32)
ramp = np.linspace(0.0, 1.0, 24, dtype=np.float32)
alpha[:, :24] = ramp[None, :]
a3 = alpha[..., None]
region = plate[:, X0:].astype(np.float32)
plate[:, X0:] = np.clip(region * (1 - a3) + med * a3, 0, 255).astype(np.uint8)
cv2.imwrite("public/plate.png", plate)
print(f"extension rebuilt from video; coverage min={counts.min()} max={counts.max()}")
