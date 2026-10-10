#!/usr/bin/env python3
"""Export initial airborne-leaf positions from the static image (f0 field)."""
import cv2
import numpy as np
import json

img = cv2.imread("work/source_static.jpg")
H, W = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
leaf = (((hsv[:, :, 0] < 32) | (hsv[:, :, 0] > 168)) &
        (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255

n, lbl, stats, cent = cv2.connectedComponentsWithStats(leaf, connectivity=8)
# canopy: large upper components (same rule as build_assets)
canopy_ids = [i for i in range(1, n)
              if stats[i, cv2.CC_STAT_AREA] >= 3000 and cent[i][1] < 760]
pile = {i for i in range(1, n) if cent[i][1] > 1080 and stats[i, cv2.CC_STAT_AREA] > 800}
# girl box
gx, gy, gw, gh = 505, 1020, 120, 175

leaves = []
for i in range(1, n):
    if i in canopy_ids or i in pile:
        continue
    a = stats[i, cv2.CC_STAT_AREA]
    if a < 12 or cent[i][1] > 1120:
        continue
    cx, cy = cent[i]
    if gx <= cx <= gx + gw and gy <= cy <= gy + gh:
        continue
    leaves.append({
        "x": round(float(cx), 1),
        "y": round(float(cy), 1),
        "w": int(stats[i, cv2.CC_STAT_WIDTH]),
        "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
        "a": int(a),
    })
leaves.sort(key=lambda l: -l["a"])
print(f"initial leaves: {len(leaves)}")
with open("src/initial_leaves.json", "w") as f:
    json.dump(leaves, f)
