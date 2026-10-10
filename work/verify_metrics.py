#!/usr/bin/env python3
"""Verify perceptual metrics: canopy loss curve + storm density + camera
path, rendered vs source."""
import cv2
import numpy as np
import json

H, W = 1280, 720
FR = [0, 20, 40, 60, 80, 100, 121]
TX = [0, -50, -105, -159, -212, -272, -336]
tx_of = np.interp(np.arange(122), FR, TX)

erodible = cv2.imread("work/erodible.png", cv2.IMREAD_GRAYSCALE)
erod = erodible > 128
ero_area0 = int(erod.sum())
storm = np.zeros((H, W), np.uint8)
storm[40:1100, 350:] = 255
storm &= 255 - (erod * 255).astype(np.uint8)

def measure(frames_pattern, name):
    loss, dens = [], []
    for f in range(0, 122, 10):
        fr = cv2.imread(frames_pattern.format(f + 1))
        M = np.float32([[1, 0, -tx_of[f]], [0, 1, 0]])
        warped = cv2.warpAffine(fr, M, (W, H))
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lm = (((hsv[:, :, 0] < 32) | (hsv[:, :, 0] > 168)) &
              (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 40)).astype(np.uint8)
        loss.append(int((lm & erod).sum()) / ero_area0)
        dens.append(int((lm & storm).sum()))
    return loss, dens

src_loss, src_dens = measure("work/frames/f{:03d}.png", "source")
mine_loss, mine_dens = measure("work/mine/m{:03d}.png", "mine")

print("frame | canopy_src canopy_mine | dens_src dens_mine")
for i, f in enumerate(range(0, 122, 10)):
    print(f"{f:5d} | {src_loss[i]:.3f}      {mine_loss[i]:.3f}       | "
          f"{int(src_dens[i]):7d} {int(mine_dens[i]):7d}")

src_loss = np.array(src_loss); mine_loss = np.array(mine_loss)
src_dens = np.array(src_dens); mine_dens = np.array(mine_dens)
print(f"\ncanopy loss curve: max abs err {np.abs(src_loss-mine_loss).max():.3f}, "
      f"mean abs err {np.abs(src_loss-mine_loss).mean():.3f}")
rel_d = np.abs(src_dens - mine_dens) / np.maximum(src_dens, 1)
print(f"storm density: mean rel err {rel_d.mean()*100:.1f}%, max {rel_d.max()*100:.1f}%")

# camera path: track the door corner in my render
img = cv2.imread("work/source_static.jpg")
g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
tpl = g0[850:930, 392:472]
txs = []
for f in [0, 20, 40, 60, 80, 100, 120]:
    fr = cv2.imread(f"work/mine/f{f+1:03d}.png") if False else cv2.imread(f"work/mine/m{f+1:03d}.png")
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
    _, mv, _, ml = cv2.minMaxLoc(res)
    txs.append(ml[0] - 432)
print("\ncamera tx (mine vs target):")
for f, t, target in zip([0,20,40,60,80,100,120], txs, [0,-50,-105,-159,-212,-272,-336]):
    print(f"  f{f:3d}: mine {t:6.0f}  target {target:6d}")
