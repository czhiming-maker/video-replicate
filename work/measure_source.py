#!/usr/bin/env python3
"""Measure source video: canopy loss curve, airborne leaf density curve,
optical-flow wind stats.  Camera = pure horizontal pan, measured earlier:
tx(f) via piecewise linear over [0,20,40,60,80,100,121] -> [0,-50,-105,-159,-212,-272,-336].
"""
import cv2
import numpy as np
import json

H, W = 1280, 720
FR = [0, 20, 40, 60, 80, 100, 121]
TX = [0, -50, -105, -159, -212, -272, -336]
tx_of = np.interp(np.arange(122), FR, TX)

img = cv2.imread("work/source_static.jpg")
hsv0 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
leaf0 = (((hsv0[:, :, 0] < 32) | (hsv0[:, :, 0] > 168)) &
         (hsv0[:, :, 1] > 80) & (hsv0[:, :, 2] > 40)).astype(np.uint8)

erodible = cv2.imread("work/erodible.png", cv2.IMREAD_GRAYSCALE)
erod = erodible > 128
ero_area0 = int(erod.sum())

# ---- canopy loss curve: orange px remaining inside erodible region ----
loss = []
for f in range(0, 122, 10):
    fr = cv2.imread(f"work/frames/f{f+1:03d}.png")
    M = np.float32([[1, 0, -tx_of[f]], [0, 1, 0]])
    warped = cv2.warpAffine(fr, M, (W, H))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lm = (((hsv[:, :, 0] < 32) | (hsv[:, :, 0] > 168)) &
          (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 40)).astype(np.uint8)
    remain = int((lm & erod).sum())
    loss.append({"frame": f, "remain_px": remain,
                 "frac": remain / ero_area0})
    print(f"f{f:3d}: canopy remain {remain/ero_area0:.3f}")

# ---- airborne leaf density curve (storm zone = right of canopy, above ground) ----
# static canopy bbox right edge ~ x 690; storm zone: x 350..719, y 40..1100,
# excluding erodible canopy region
storm = np.zeros((H, W), np.uint8)
storm[40:1100, 350:] = 255
storm &= 255 - (erod * 255).astype(np.uint8)
dens = []
for f in range(0, 122, 10):
    fr = cv2.imread(f"work/frames/f{f+1:03d}.png")
    M = np.float32([[1, 0, -tx_of[f]], [0, 1, 0]])
    warped = cv2.warpAffine(fr, M, (W, H))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lm = (((hsv[:, :, 0] < 32) | (hsv[:, :, 0] > 168)) &
          (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 40)).astype(np.uint8)
    px = int((lm & storm).sum())
    dens.append({"frame": f, "px": px})
    print(f"f{f:3d}: storm leaf px {px}")

# ---- optical flow between consecutive frames (storm zone, half res) ----
def storm_roi(frame):
    x0 = 330 + int(tx_of[frame])  # zone moves with camera in frame coords
    x1 = min(W, 720 + int(tx_of[f]))
    roi = frame_img[40:1100, max(0, x0):x1]
    return cv2.resize(roi, (0, 0), fx=0.5, fy=0.5)

flows = []
prev_gray = None
prev_f = None
for f in range(0, 121):
    frame_img = cv2.imread(f"work/frames/f{f+1:03d}.png")
    x0 = 330 + int(tx_of[f])
    x1 = min(W, 720 + int(tx_of[f]))
    roi = frame_img[40:1100, max(0, x0):x1]
    small = cv2.resize(roi, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None and gray.shape == prev_gray.shape:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 21, 3, 5, 1.2, 0)
        fx = flow[..., 0] * 2.0  # back to full-res px/frame
        fy = flow[..., 1] * 2.0
        mag = np.sqrt(fx ** 2 + fy ** 2)
        # robust: median (leaves are sparse among static paper -> median ~ paper)
        flows.append({
            "frame": f,
            "vx_med": float(np.median(fx)),
            "vy_med": float(np.median(fy)),
            "vx_p85": float(np.percentile(fx, 85)),
            "vy_p85": float(np.percentile(fy, 85)),
            "speed_p90": float(np.percentile(mag, 90)),
            "speed_p99": float(np.percentile(mag, 99)),
        })
    prev_gray = gray

vx85 = np.array([q["vx_p85"] for q in flows])
vy85 = np.array([q["vy_p85"] for q in flows])
sp90 = np.array([q["speed_p90"] for q in flows])
print(f"\nwind (p85 over time): vx mean {vx85.mean():.2f} px/f, "
      f"vy mean {vy85.mean():.2f} px/f")
print(f"leaf speed p90 over time: mean {sp90.mean():.2f}, max {sp90.max():.2f}")

out = {
    "tx_per_frame": tx_of.tolist(),
    "canopy_loss": loss,
    "storm_density": dens,
    "flow": flows,
    "ero_area0": ero_area0,
}
with open("work/motion.json", "w") as fjson:
    json.dump(out, fjson, indent=1)
print("motion.json written")
