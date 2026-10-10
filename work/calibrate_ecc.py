#!/usr/bin/env python3
"""Camera path v2: ECC affine alignment static->frame with leaf-reject mask."""
import cv2
import numpy as np
import json

STATIC = "work/source_static.jpg"
FRAMES = {1: 0, 2: 20, 3: 40, 4: 60, 5: 80, 6: 100, 7: 121}

img = cv2.imread(STATIC)
H, W = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# leaf-colored pixels: orange/red hues, saturated
leaf_mask = ((hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 90)
leaf_mask = cv2.dilate(leaf_mask.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))

mask = np.zeros((H, W), np.uint8)
y0 = int(0.52 * H)
mask[y0:, int(0.03 * W):int(0.97 * W)] = 255
mask &= 255 - leaf_mask  # reject flying leaves inside stable zone
# keep girl region anyway (she is small; ECC dominated by rest)
print(f"mask coverage: {(mask > 0).mean() * 100:.1f}%")

out = {}
for fidx, fnum in sorted(FRAMES.items()):
    fr = cv2.imread(f"work/vf_{fidx}.png")
    fgray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(fgray, gray, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                         300, 1e-7), inputMask=mask, gaussFiltSize=9)
    except cv2.error as e:
        print(f"frame {fnum}: ECC failed {e}")
        continue
    # warp maps frame coords -> static coords.  static_pt = A @ frame_pt
    A = warp
    s = np.sqrt(A[0, 0] * A[0, 0] + A[1, 0] * A[1, 0])  # scale along x axis
    rot = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    # static_pt = warp @ [x,y,1].  so x_s = a00*x + a01*y + a02
    # frame_pt = A^-1 @ static_pt  ->  express static->frame:
    Ainv = cv2.invertAffineTransform(A)
    s2 = np.sqrt(Ainv[0, 0] ** 2 + Ainv[1, 0] ** 2)
    rot2 = np.degrees(np.arctan2(Ainv[1, 0], Ainv[0, 0]))
    out[fnum] = {"static_to_frame": Ainv.tolist(), "cc": float(cc)}
    print(f"frame {fnum:3d}: cc={cc:.4f} | static->frame scale={s2:.4f} "
          f"rot={rot2:.3f}deg tx={Ainv[0,2]:.2f} ty={Ainv[1,2]:.2f}")

with open("work/camera_path_ecc.json", "w") as f:
    json.dump(out, f, indent=1)

# verify: warp each frame back onto static, save a blend for 3 frames
for fidx in (1, 4, 7):
    fnum = FRAMES[fidx]
    if fnum not in out:
        continue
    fr = cv2.imread(f"work/vf_{fidx}.png")
    A = np.array(out[fnum]["static_to_frame"], dtype=np.float32)
    Ainv = cv2.invertAffineTransform(A)
    warped = cv2.warpAffine(fr, Ainv, (W, H), flags=cv2.INTER_LINEAR)
    blend = (img.astype(np.float32) * 0.5 + warped.astype(np.float32) * 0.5).astype(np.uint8)
    cv2.imwrite(f"work/ecc_check_{fnum}.jpg", blend, [cv2.IMWRITE_JPEG_QUALITY, 80])
print("check blends written")
