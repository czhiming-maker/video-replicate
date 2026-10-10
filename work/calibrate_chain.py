#!/usr/bin/env python3
"""Camera path v3: chained consecutive-frame alignment.

ORB+RANSAC similarity init -> ECC affine refine, per consecutive pair,
accumulated to a static->frame_N path. Frame 0 == static image exactly.
"""
import cv2
import numpy as np
import json

FRAMES = [0, 20, 40, 60, 80, 100, 121]
STATIC = "work/source_static.jpg"

img = cv2.imread(STATIC)
H, W = img.shape[:2]
gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
leaf = (((hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 90)).astype(np.uint8) * 255
leaf = cv2.dilate(leaf, np.ones((9, 9), np.uint8))

def stable_mask(frame_bgr):
    """bottom 48% minus leaf-colored pixels of that frame"""
    m = np.zeros((H, W), np.uint8)
    m[int(0.52 * H):, int(0.03 * W):int(0.97 * W)] = 255
    hsvf = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lf = (((hsvf[:, :, 0] < 30) | (hsvf[:, :, 0] > 170)) & (hsvf[:, :, 1] > 90)).astype(np.uint8) * 255
    lf = cv2.dilate(lf, np.ones((11, 11), np.uint8))
    return m & (255 - lf)

def orb_align(g_from, g_to, mask_to):
    orb = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = orb.detectAndCompute(g_from, None)
    k2, d2 = orb.detectAndCompute(g_to, mask_to)
    if d1 is None or d2 is None:
        return None, 0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=2.0)
    if M is None:
        return None, len(good)
    return M, int(inliers.sum())

def ecc_refine(g_from, g_to, M, mask):
    warp = M.astype(np.float32)
    try:
        cc, warp = cv2.findTransformECC(g_from, g_to, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                         200, 1e-6), inputMask=mask, gaussFiltSize=7)
        return warp, float(cc)
    except cv2.error:
        return warp, -1.0

# load frames
frames = {}
for i, fnum in enumerate(FRAMES):
    frames[fnum] = cv2.imread(f"work/vf_{i+1}.png")

# accumulated transform: static -> frame_N, as 2x3 affine (maps static coords to frame N coords)
# frame0 == static
acc = np.eye(2, 3, dtype=np.float64)
path = {0: {"M": acc.tolist(), "via": "identity"}}

prev_fnum = 0
for fnum in FRAMES[1:]:
    g_prev = cv2.cvtColor(frames[prev_fnum], cv2.COLOR_BGR2GRAY)
    g_cur = cv2.cvtColor(frames[fnum], cv2.COLOR_BGR2GRAY)
    mask_cur = stable_mask(frames[fnum])
    M_orb, ninl = orb_align(g_prev, g_cur, mask_cur)
    if M_orb is None:
        print(f"{prev_fnum}->{fnum}: ORB failed ({ninl} matches)")
        continue
    s = np.sqrt(M_orb[0, 0] ** 2 + M_orb[1, 0] ** 2)
    rot = np.degrees(np.arctan2(M_orb[1, 0], M_orb[0, 0]))
    print(f"{prev_fnum:3d}->{fnum:3d}: ORB inl={ninl} scale={s:.4f} rot={rot:.3f} "
          f"tx={M_orb[0,2]:.2f} ty={M_orb[1,2]:.2f}", end=" ")
    M_ecc, cc = ecc_refine(g_prev, g_cur, M_orb, mask_cur)
    s2 = np.sqrt(M_ecc[0, 0] ** 2 + M_ecc[1, 0] ** 2)
    rot2 = np.degrees(np.arctan2(M_ecc[1, 0], M_ecc[0, 0]))
    print(f"| ECC cc={cc:.4f} scale={s2:.4f} rot={rot2:.3f} tx={M_ecc[0,2]:.2f} ty={M_ecc[1,2]:.2f}")
    use = M_ecc if cc > 0.9 else M_orb
    # accumulate: prev = acc applied first, then use.  total = use ∘ acc
    A_acc = np.vstack([acc, [0, 0, 1]])
    A_use = np.vstack([use, [0, 0, 1]])
    total = A_use @ A_acc
    acc = total[:2, :]
    path[fnum] = {"M": acc.tolist(), "via": "ecc" if (cc > 0.9 and cc > 0) else "orb"}
    prev_fnum = fnum

with open("work/camera_chain.json", "w") as f:
    json.dump(path, f, indent=1)

print("\nAccumulated static->frame:")
for fnum in FRAMES:
    if fnum not in path:
        continue
    M = np.array(path[fnum]["M"])
    sc = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    print(f"  f{fnum:3d} [{path[fnum]['via']:8s}]: scale={sc:.4f} rot={rot:+.3f}deg "
          f"tx={M[0,2]:8.2f} ty={M[1,2]:8.2f}")

# verification blend for 3 frames
for fnum in FRAMES:
    if fnum not in path or fnum == 0:
        continue
    M = np.array(path[fnum]["M"], dtype=np.float32)
    Minv = cv2.invertAffineTransform(M)
    warped = cv2.warpAffine(frames[fnum], Minv, (W, H))
    blend = (img.astype(np.float32) * 0.5 + warped.astype(np.float32) * 0.5).astype(np.uint8)
    cv2.imwrite(f"work/chain_check_{fnum}.jpg", blend, [cv2.IMWRITE_JPEG_QUALITY, 80])
print("check blends written")
