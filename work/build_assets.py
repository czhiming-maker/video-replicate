#!/usr/bin/env python3
"""Asset pipeline: masks, clean plate (paper quilting), leaf sprite atlas,
canopy RGBA layer.  Outputs under public/ + manifests under work/.
"""
import cv2
import numpy as np
import json
import os

rng = np.random.default_rng(42)

STATIC = "work/source_static.jpg"
OUT = "public"
os.makedirs(f"{OUT}/sprites", exist_ok=True)
os.makedirs("work", exist_ok=True)

img = cv2.imread(STATIC)
H, W = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ---------- leaf color mask ----------
# saturated warm hues (orange/red/yellow-orange)
leaf_hue = ((hsv[:, :, 0] < 32) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 40)
leaf = (leaf_hue.astype(np.uint8)) * 255
leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

ncc, lbl, stats, cent = cv2.connectedComponentsWithStats(leaf, connectivity=8)

# canopy: large leaf components confined to the UPPER zone (building wood
# lives lower and must not leak into the canopy via warm door/window colors).
# Warm wood connects canopy+building+pile into ONE component, so cut it
# spatially: keep component pixels above y=742, plus the trunk corridor.
canopy0 = np.zeros((H, W), np.uint8)
canopy_ids = []
for i in range(1, ncc):
    if stats[i, cv2.CC_STAT_AREA] >= 3000 and cent[i][1] < 760:
        canopy_ids.append(i)
for i in canopy_ids:
    canopy0 |= (lbl == i).astype(np.uint8) * 255
canopy_zone = np.zeros((H, W), np.uint8)
canopy_zone[:742, :] = 255
canopy_zone[742:960, 15:235] = 255  # trunk corridor
canopy0 &= canopy_zone
# kill thin warm false-positives (brick mortar lines, cornice edges) that
# would otherwise flash as dark rectangles in the canopy layer
canopy0 = cv2.morphologyEx(canopy0, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
nn0, ll0, ss0, _ = cv2.connectedComponentsWithStats(canopy0, connectivity=8)
canopy_clean = np.zeros((H, W), np.uint8)
for i in range(1, nn0):
    if ss0[i, cv2.CC_STAT_AREA] >= 30:
        canopy_clean |= (ll0 == i).astype(np.uint8) * 255
canopy0 = canopy_clean

# ground pile: big components near ground
pile = np.zeros((H, W), np.uint8)
for i in range(1, ncc):
    if i in canopy_ids:
        continue
    if cent[i][1] > 1080 and stats[i, cv2.CC_STAT_AREA] > 800:
        pile |= (lbl == i).astype(np.uint8) * 255

canopy_union = cv2.dilate(canopy0 | pile, np.ones((9, 9), np.uint8))

# ---------- structure protect zones ----------
protect = np.zeros((H, W), np.uint8)
# wall-top shadow zone starts ~y=495: canopy hanging over the brick must be
# protected, or its alpha dips expose bright quilted paper as rings
protect[495:912, 185:458] = 255     # brick wall + cornice zone
protect[470:780, 20:225] = 255      # trunk corridor
protect = cv2.dilate(protect, np.ones((13, 13), np.uint8))

# girl keep-out box (her coat/scarf are leaf-colored)
GIRL = (505, 1020, 120, 175)  # x, y, w, h
gx, gy, gw, gh = GIRL
girl = np.zeros((H, W), np.uint8)
girl[gy:gy + gh, gx:gx + gw] = 255

# ground zone (keep everything there)
ground = np.zeros((H, W), np.uint8)
ground[1120:, :] = 255

# ---------- airborne leaves over paper (removable) ----------
canopy_lbl = (canopy0 > 0)
pile_lbl = (pile > 0)
airborne_ids = []
for i in range(1, ncc):
    a = stats[i, cv2.CC_STAT_AREA]
    if a < 5 or a > 4000 or cent[i][1] > 1120:
        continue
    comp = (lbl == i)
    if (comp & canopy_lbl).sum() > 0 or (comp & pile_lbl).sum() > 0:
        continue
    airborne_ids.append(i)
airborne_paper = np.zeros((H, W), np.uint8)   # to remove + sprite
airborne_static = np.zeros((H, W), np.uint8)  # over structure -> keep static
for i in airborne_ids:
    m = (lbl == i).astype(np.uint8) * 255
    over_struct = (m & protect).sum() > 0.25 * m.sum()
    over_girl = (m & girl).sum() > 0
    if over_struct or over_girl:
        airborne_static |= m
    else:
        airborne_paper |= m

# erodible canopy = canopy part over paper only
erodible = canopy0 & 255 - protect
erodible = cv2.morphologyEx(erodible, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
# drop tiny slivers
ner, lble, statse, _ = cv2.connectedComponentsWithStats(erodible, connectivity=8)
erodible = np.zeros((H, W), np.uint8)
for i in range(1, ner):
    if statse[i, cv2.CC_STAT_AREA] >= 60:
        erodible |= (lble == i).astype(np.uint8) * 255

protected_canopy = canopy0 & ~erodible

print(f"canopy px: {int((canopy0>0).sum())}, erodible: {int((erodible>0).sum())}, "
      f"protected: {int((protected_canopy>0).sum())}")
print(f"airborne-over-paper px: {int((airborne_paper>0).sum())}, "
      f"airborne-static px: {int((airborne_static>0).sum())}")

# ---------- paper texture quilting ----------
def find_clean_windows(bank_zone_mask, win=72, count=40):
    """scan windows fully inside zone, no leaf pixels, return patch list"""
    patches = []
    for _ in range(4000):
        y = rng.integers(0, H - win)
        x = rng.integers(0, W - win)
        if (leaf[y:y + win, x:x + win] > 0).any():
            continue
        if (pile[y:y + win, x:x + win] > 0).any():
            continue
        if (airborne_paper[y:y + win, x:x + win] > 0).any():
            continue
        patches.append({"img": img[y:y + win, x:x + win].copy(), "y": int(y)})
        if len(patches) >= count:
            break
    return patches

paper_zone = np.full((H, W), 255, np.uint8)
paper_zone[1100:, :] = 0            # no pavement in paper bank
paper_zone[850:, 180:460] = 0       # no storefront
paper_zone[460:800, 15:230] = 0     # no trunk
paper_bank = find_clean_windows(paper_zone, win=72, count=48)
print(f"paper bank patches: {len(paper_bank)}")

def normalize_bank(bank):
    """per-piece mean normalization so quilted blocks don't patchwork"""
    ref = float(np.mean([b["img"].mean() for b in bank]))
    out = []
    for b in bank:
        piece = b["img"].astype(np.float32)
        piece = piece * (ref / max(float(piece.mean()), 1.0))
        out.append({"img": piece, "y": b["y"]})
    return out

def shadow_field(valid):
    """scalar luminance-extrapolation ratio (keeps canopy shadow gradients
    without per-channel tint). 1.0 = open paper brightness."""
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    num = cv2.blur(lum * valid, (121, 121))
    den = cv2.blur(valid, (121, 121))
    L = num / np.maximum(den, 1e-4)
    ref = float(np.median(L[valid > 0]))
    return np.clip(L / max(ref, 1.0), 0.86, 1.06).astype(np.float32)

def quilt(shape_hw, region_mask, bank, block=36, feather=14, row_lock=False, rng_=None):
    """return a fully-filled image; blocks feather-blended, bank pieces
    mean-normalized.  row_lock keeps source rows aligned (for pavement).
    rng_ overrides the module-global rng (girl-region repair must not shift
    the global sequence or downstream sprites/canopy would drift)."""
    Ht, Wt = shape_hw
    r = rng if rng_ is None else rng_
    bh, bw = bank[0]["img"].shape[:2]
    out = np.zeros((Ht, Wt, 3), np.float32)
    weight = np.zeros((Ht, Wt), np.float32)
    ys, xs = np.where(region_mask > 0)
    if len(ys) == 0:
        return out
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    f = cv2.getGaussianKernel(feather * 2 + 1, feather / 2.4)
    kernel2d = (f @ f.T)
    kernel2d /= kernel2d.max()
    half = block // 2 + feather // 2
    kc = feather // 2 + block // 2
    for by in range(y0, y1 + 1, block):
        for bx in range(x0, x1 + 1, block):
            yc, xc = by + block // 2, bx + block // 2
            yA, yB = max(0, yc - half), min(Ht, yc + half)
            xA, xB = max(0, xc - half), min(Wt, xc + half)
            if yB <= yA or xB <= xA:
                continue
            wy, wx = yB - yA, xB - xA
            if bh < wy or bw < wx:
                continue
            patch = bank[r.integers(0, len(bank))]
            if row_lock:
                py = int(np.clip(yc - patch["y"], 0, bh - wy))
            else:
                py = r.integers(0, bh - wy + 1)
            px = r.integers(0, bw - wx + 1)
            piece = patch["img"][py:py + wy, px:px + wx]
            ky0 = kc - (yc - yA)
            kx0 = kc - (xc - xA)
            k = kernel2d[ky0:ky0 + wy, kx0:kx0 + wx]
            if k.shape[:2] != (wy, wx):
                k = np.ones((wy, wx), np.float32)
            out[yA:yB, xA:xB] += piece * k[..., None]
            weight[yA:yB, xA:xB] += k
    filled = out / np.maximum(weight, 1e-6)[..., None]
    hole = (region_mask > 0) & (weight < 1e-5)
    if hole.any():
        inp = cv2.inpaint(filled.astype(np.uint8), (hole * 255).astype(np.uint8),
                          5, cv2.INPAINT_TELEA)
        filled[hole] = inp[hole].astype(np.float32)
    return filled

def quilt_fill(target, region_mask, bank, block=36, feather=14):
    """fill region_mask area of target with feathered random blocks from bank"""
    Ht, Wt = target.shape[:2]
    bh, bw = bank[0].shape[:2]
    out = target.astype(np.float32).copy()
    weight = np.zeros((Ht, Wt), np.float32)
    ys, xs = np.where(region_mask > 0)
    if len(ys) == 0:
        return target
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    f = cv2.getGaussianKernel(feather * 2 + 1, feather / 2.4)
    kernel2d = (f @ f.T)
    kernel2d /= kernel2d.max()
    half = block // 2 + feather // 2
    kc = feather // 2 + block // 2  # kernel center index
    for by in range(y0, y1 + 1, block):
        for bx in range(x0, x1 + 1, block):
            yc, xc = by + block // 2, bx + block // 2
            yA, yB = max(0, yc - half), min(Ht, yc + half)
            xA, xB = max(0, xc - half), min(Wt, xc + half)
            if yB <= yA or xB <= xA:
                continue
            wy, wx = yB - yA, xB - xA
            if bh < wy or bw < wx:
                continue
            patch = bank[rng.integers(0, len(bank))]
            py = rng.integers(0, bh - wy + 1)
            px = rng.integers(0, bw - wx + 1)
            piece = patch[py:py + wy, px:px + wx].astype(np.float32)
            ky0 = kc - (yc - yA)
            kx0 = kc - (xc - xA)
            k = kernel2d[ky0:ky0 + wy, kx0:kx0 + wx]
            if k.shape[:2] != (wy, wx):
                k = np.ones((wy, wx), np.float32)
            out[yA:yB, xA:xB] += piece * k[..., None]
            weight[yA:yB, xA:xB] += k
    # fill un-covered pixels inside region from nearest covered (fallback)
    w = weight[..., None]
    filled = out / np.maximum(w, 1e-6)
    uncovered = (region_mask > 0) & (weight < 1e-5)
    if uncovered.any():
        inp = cv2.inpaint(filled.astype(np.uint8), (uncovered * 255).astype(np.uint8),
                          5, cv2.INPAINT_TELEA)
        filled[uncovered] = inp[uncovered]
    # blend: only inside region, feathered boundary
    m = cv2.GaussianBlur(region_mask.astype(np.float32) / 255.0, (9, 9), 3)[..., None]
    result = target.astype(np.float32) * (1 - m) + filled * m
    return np.clip(result, 0, 255).astype(np.uint8)

# ---------- clean plate ----------
EXT = 380  # right extension
plate = np.zeros((H, W + EXT, 3), np.uint8)
plate[:, :W] = img

# regions to remove from plate: airborne-over-paper + erodible canopy.
# airborne leaves cast soft drop-shadows reaching ~20px, so dilate enough
# to quilt them away too — but never eat structures/girl/pile.
remove_mask = cv2.dilate(airborne_paper, np.ones((21, 21), np.uint8)) | \
    cv2.dilate(erodible, np.ones((3, 3), np.uint8))
keep_zone = cv2.dilate(protect, np.ones((5, 5), np.uint8)) | girl | \
    cv2.dilate(pile, np.ones((7, 7), np.uint8))
remove_mask &= 255 - keep_zone
print("quilting plate removals...")
bank_n = normalize_bank(paper_bank)
filled = quilt((H, W), remove_mask, bank_n, block=26, feather=9)
valid = (~(remove_mask > 0)).astype(np.float32)
S = shadow_field(valid)
# shading only in a narrow band around the hole boundary; deep interior
# stays at full paper brightness (matches the video's exposed paper)
interior = cv2.erode((remove_mask > 0).astype(np.uint8) * 255,
                     np.ones((55, 55), np.uint8))
band = cv2.dilate((remove_mask > 0).astype(np.uint8) * 255,
                  np.ones((9, 9), np.uint8)) - interior
band_w = np.clip(cv2.GaussianBlur(band.astype(np.float32) / 255.0, (0, 0), 5) * 2.5,
                 0, 1)
S_eff = 1.0 + (S - 1.0) * band_w
m_soft = cv2.GaussianBlur((remove_mask > 0).astype(np.float32), (0, 0), 6)[..., None]
fill_shaded = np.clip(filled * S_eff[..., None], 0, 255)
plate_f = plate.astype(np.float32)
plate_f[:, :W] = img.astype(np.float32) * (1 - m_soft) + fill_shaded * m_soft
# canopy shadow: paper under/around the canopy mass sits slightly darker
# than open paper (the video's exposed paper confirms this)
sh_zone = cv2.GaussianBlur(((erodible > 128).astype(np.float32)), (0, 0), 24)
sh_zone = np.clip(sh_zone * 1.4, 0, 1)[..., None]
plate_f[:, :W] *= 1.0 - sh_zone * 0.025
plate = np.clip(plate_f, 0, 255).astype(np.uint8)

# flatten bright ghost speckles inside the quilted region: pull pixels
# brighter than the local paper median down toward it (removed leaves leave
# pale halos that read as white fog against clean paper)
lum = cv2.cvtColor(plate[:, :W], cv2.COLOR_BGR2GRAY).astype(np.float32)
vint = (valid * 255).astype(np.uint8)
lum8 = np.clip(lum, 0, 255).astype(np.uint8)
cov = cv2.medianBlur(vint, 31).astype(np.float32) / 255.0
loc_med = cv2.medianBlur(lum8, 31).astype(np.float32) / np.maximum(cov, 1e-3)
over = (remove_mask > 0) & (lum > loc_med + 5)
scale = np.where(over, (loc_med + 5) / np.maximum(lum, 1.0), 1.0).astype(np.float32)
plate_f2 = plate.astype(np.float32)
plate_f2[:, :W] *= scale[..., None]
plate = np.clip(plate_f2, 0, 255).astype(np.uint8)

# right extension: row-run mirror synthesis.  For each row take the longest
# leaf-free run of the ORIGINAL row and mirror-tile it rightward — the
# extension then carries the original paper/pavement grain exactly (quilted
# blocks read as flat/gray next to real grain).
invalid = ((leaf > 0) | (pile > 0) | (girl > 0) |
           (cv2.dilate(airborne_paper, np.ones((5, 5), np.uint8)) > 0))
# never sample structures: whole building + trunk corridor
invalid[440:1150, 180:465] = 255
invalid[430:1060, 15:240] = 255
ext = np.zeros((H, EXT, 3), np.float32)
prev_seg = None
for y in range(H):
    row_ok = ~invalid[y, :W].astype(bool)
    best_len, best_end, cur_len, cur_end = 0, 0, 0, 0
    for x in range(470, W):  # sampling zone: paper/pavement right of structures
        if row_ok[x]:
            cur_len += 1
            cur_end = x
            if cur_len > best_len:
                best_len, best_end = cur_len, cur_end
        else:
            cur_len = 0
    if best_len >= 48:
        start = max(0, best_end - 90)
        prev_seg = img[y, start:best_end + 1].astype(np.float32)
    seg = prev_seg
    if seg is None:
        seg = np.tile(img[y, 700:719].astype(np.float32), (20, 1))
    tile = np.zeros((EXT, 3), np.float32)
    pos, flip = 0, False
    L = seg.shape[0]
    while pos < EXT:
        take = min(L, EXT - pos)
        piece = seg[-take:] if not flip else seg[:take][::-1]
        tile[pos:pos + take] = piece
        pos += take
        flip = not flip
    ext[y] = tile
plate[:, W:] = np.clip(ext, 0, 255).astype(np.uint8)
# 28px cross-fade across the seam (paper-on-paper only)
ramp = np.linspace(0.0, 1.0, 28).astype(np.float32)[None, :, None]
join = (plate[:, W - 28:W].astype(np.float32) * (1 - ramp) +
        plate[:, W:W + 28].astype(np.float32) * ramp)
plate[:, W:W + 28] = np.clip(join, 0, 255).astype(np.uint8)

# leaf specks that slipped into the extension: inpaint (small spots only)
hsv_ext = cv2.cvtColor(plate[:, W:], cv2.COLOR_BGR2HSV)
leaf_ext = (((hsv_ext[:, :, 0] < 32) | (hsv_ext[:, :, 0] > 168)) &
            (hsv_ext[:, :, 1] > 80) & (hsv_ext[:, :, 2] > 40)).astype(np.uint8) * 255
leaf_ext = cv2.morphologyEx(leaf_ext, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
nn, ll, ss, _ = cv2.connectedComponentsWithStats(leaf_ext, connectivity=8)
specks = np.zeros_like(leaf_ext)
for i in range(1, nn):
    if ss[i, cv2.CC_STAT_AREA] <= 260:
        specks |= (ll == i).astype(np.uint8) * 255
if specks.any():
    plate[:, W:] = cv2.inpaint(plate[:, W:], specks, 4, cv2.INPAINT_TELEA)
print(f"extension specks inpainted: {int((specks > 0).sum())} px")

# ---------- girl region repair ----------
# The girl becomes a per-frame sprite (work/export_girl_seq.py); the baked
# copy must leave the plate or her old pose ghosts out behind the animation.
# Wall part quilts from a paper bank, pavement part from a row-locked
# pavement bank.  Dedicated rng: must NOT shift the global rng sequence or
# every downstream asset (sprites/canopy) would drift.
GR = (495, 1005, 145, 200)   # static-pose repair zone; NOT the sprite rect
# (GSEQ in export_girl_seq.py is [490,1005,165,232]: the per-frame sprite
# extends past GR for the outstretched hand x<=654, the contact shadow and
# the slab front-face band y<=1231.  GR only needs to cover the girl baked
# into source_static.jpg — keeping it minimal means the plate retains its
# original slab lines / band outside the static pose, and every pixel the
# sprite restores below GR composites over an identical original backdrop.
# Do not shrink GR below the static content extent.)
# regression gate margin: task gate treats x 477-658 / y 987-1223 (GR + ~18px) as the only region allowed to change
grx, gry, grw, grh = GR
gr_zone = np.zeros((H, W), np.uint8)
gr_zone[gry:gry + grh, grx:grx + grw] = 255
gr_zone = cv2.dilate(gr_zone, np.ones((9, 9), np.uint8))
SPLIT = 1120                                  # wall/pavement horizon (ground[])
gr_wall = gr_zone.copy(); gr_wall[SPLIT:, :] = 0
gr_pave = gr_zone.copy(); gr_pave[:SPLIT, :] = 0
g_rng = np.random.default_rng(4242)

def _bank(zone_rows, exclude_extra):
    """clean 72px windows for quilting; zone_rows = (y0, y1) row range"""
    excl = cv2.dilate((leaf | pile | airborne_paper | gr_zone | exclude_extra),
                      np.ones((9, 9), np.uint8))
    y0, y1 = zone_rows
    patches = []
    for _ in range(4000):
        y = int(g_rng.integers(y0, max(y0 + 1, y1 - 72)))
        x = int(g_rng.integers(0, W - 72))
        if (excl[y:y + 72, x:x + 72] > 0).any():
            continue
        patches.append({"img": img[y:y + 72, x:x + 72].copy(), "y": int(y)})
        if len(patches) >= 24:
            break
    return patches

paper_g = normalize_bank(_bank((0, SPLIT - 72), paper_zone == 0))

def _bank_pave():
    """pavement windows for row-locked quilting, same rows as the girl zone.
    Preferred source: work/ext_video_median.png — the temporal-median
    extension build_extension_from_video.py computes (clean real pavement,
    no leaves, no mirror-fold seams; med column i = plate x 700+i).  Direct
    sampling on rows >=1120 is impossible (leaves cover ~65%: 0% clean 56px
    windows), and the in-build mirror-synth ext carries fold seams that
    quilt out as vertical streaks.  Warm-pixel windows are rejected anyway
    (belt and braces).  Falls back to mirror-synth ext if the median file
    is absent (fresh clone: run build_extension_from_video.py once)."""
    ext_src = cv2.imread("work/ext_video_median.png")
    if ext_src is None:
        print("girl repair: ext_video_median.png missing — falling back to "
              "mirror-synth ext (streak-prone)")
        ext_src = plate[:, W:]
    hsv_e = cv2.cvtColor(ext_src, cv2.COLOR_BGR2HSV)
    # NB: the video pavement itself is warm tan (sat p99 ~81), so the leaf
    # mask's sat>80 would reject EVERY window — use leaf-core threshold
    ext_warm = (((hsv_e[:, :, 0] < 32) | (hsv_e[:, :, 0] > 168)) &
                (hsv_e[:, :, 1] > 110) & (hsv_e[:, :, 2] > 40)).astype(np.uint8) * 255
    ext_warm = cv2.dilate(ext_warm, np.ones((5, 5), np.uint8)) > 0
    ew, eh = ext_warm.shape[1], ext_warm.shape[0]
    patches = []
    for _ in range(4000):
        y = int(g_rng.integers(SPLIT, max(SPLIT + 1, eh - 72)))
        x = int(g_rng.integers(60, ew - 72))   # local col; skip feather zone
        if ext_warm[y:y + 72, x:x + 72].any():
            continue
        patches.append({"img": ext_src[y:y + 72, x:x + 72].copy(), "y": int(y)})
        if len(patches) >= 24:
            break
    return patches

pave_g = normalize_bank(_bank_pave())
print(f"girl repair banks: paper={len(paper_g)} pave={len(pave_g)}")
assert len(paper_g) > 5 and len(pave_g) > 5, "empty quilt bank — widen zone"

def _repair_part(zone_m, bank, row_lock):
    fill = quilt((H, W), zone_m, bank, block=26, feather=14, row_lock=row_lock,
                 rng_=g_rng)
    # low-frequency gain: pull the fill's blurred luminance toward the
    # surrounding plate's (masked blur = extrapolation inward), so residual
    # shading gradients (girl shadow remnant, vignette) don't read as
    # bright patches.  Per-channel, tightly clipped.
    soft0 = zone_m.astype(np.float32) / 255.0
    inv = (1.0 - soft0)[..., None]
    wgt = soft0[..., None]
    k = (61, 61)
    plate_f32 = plate[:, :W].astype(np.float32)
    # note: cv2.blur squeezes a trailing 1-channel axis (cv2 5.x) — re-add it
    pl_lf = (cv2.blur(plate_f32 * inv, k) /
             np.maximum(cv2.blur(inv, k), 1e-3)[..., None])
    fl_lf = (cv2.blur(fill * wgt, k) /
             np.maximum(cv2.blur(wgt, k), 1e-3)[..., None])
    gain = np.clip(pl_lf / np.maximum(fl_lf, 1e-3), 0.92, 1.08)
    # blur the gain field: the clip above saturates into flat plateaus whose
    # boundaries would otherwise read as faint tone rectangles
    gain = cv2.GaussianBlur(gain, (0, 0), 12)
    fill = fill * gain
    soft = cv2.GaussianBlur(soft0, (0, 0), 3)
    pf = plate.astype(np.float32)
    pf[:, :W] = pf[:, :W] * (1 - soft[..., None]) + fill * soft[..., None]
    return pf

# wall first, then pavement blends on top of the wall-repaired plate
# (each part soft-fades ~9px around its zone; SPLIT seam gets both fades)
plate = np.clip(_repair_part(gr_wall, paper_g, False), 0, 255).astype(np.uint8)
plate = np.clip(_repair_part(gr_pave, pave_g, True), 0, 255).astype(np.uint8)
cv2.imwrite("work/girl_plate_check.png",
            plate[gry - 24:gry + grh + 24, grx - 24:grx + grw + 24])
print("girl region repaired")

cv2.imwrite(f"{OUT}/plate.png", plate)
print("plate written", plate.shape)

# ---------- sprite atlas v2 ----------
# Problems with v1: components fused touching leaves (they fly as one blob)
# and included the leaf's cast shadow (flew as a baked shadow patch).
# v2: split components by erosion + regrow, then cut alpha from a strict-ish
# LEAF-PIXEL mask (sat>85/val>45) so shadows and paper never enter a sprite;
# interior leaf shading (opaque pixels) keeps its original RGB.
leafpix = (((hsv[:, :, 0] < 33) | (hsv[:, :, 0] > 167)) &
           (hsv[:, :, 1] > 85) & (hsv[:, :, 2] > 45))

def cut_leaf_sprite(lmask, k, prefix):
    """lmask: uint8 single-leaf mask (full-image coords). Returns manifest
    entry or None if too small / degenerate."""
    ys, xs = np.where(lmask > 0)
    if len(ys) == 0:
        return None
    x0, x1 = max(0, xs.min() - 2), min(W, xs.max() + 3)
    y0, y1 = max(0, ys.min() - 2), min(H, ys.max() + 3)
    # JPEG 4:2:0 chroma subsampling paints a saturated halo into the paper
    # around every leaf (leafpix alone cannot exclude it).  Keep only pixels
    # within 2.2px of a strongly-saturated leaf core.
    hsv_r = hsv[y0:y1, x0:x1]
    strong = ((hsv_r[:, :, 0] >= 3) & (hsv_r[:, :, 0] <= 31) &
              (hsv_r[:, :, 1] > 130) & (hsv_r[:, :, 2] > 55))
    dist = cv2.distanceTransform((~strong).astype(np.uint8),
                                 cv2.DIST_L2, 3)
    keep = dist <= 2.2
    cand = ((lmask[y0:y1, x0:x1] > 0) & keep).astype(np.uint8)
    # seal paper slits that connect to the leaf outline (edge "bites") so
    # they count as interior and get filled like enclosed holes — the source
    # leaves are solid, the slits are a cutting artifact.  Adopt only SMALL
    # closed-in regions (<=32px each): big inter-leaf paper areas of dense
    # clusters must stay transparent or the sprite turns into a solid blob.
    closed = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    add = ((closed > 0) & (cand == 0)).astype(np.uint8)
    nn, ll, ss, _ = cv2.connectedComponentsWithStats(add, connectivity=8)
    adopted = np.zeros_like(cand)
    for i in range(1, nn):
        if ss[i, cv2.CC_STAT_AREA] <= 32:
            cand[ll == i] = 1
            adopted[ll == i] = 1
    # Dark leaf interiors sit far from any strong pixel and would read as
    # "far from core" — but they are leaf body, not halo.  Enclosed
    # background (holes in cand) is interior: fill it back at full alpha.
    # Pad a background ring before flooding so the seed is always background
    # (a leaf touching the crop corner would kill a (0,0)-seeded flood and
    # mark the whole crop as "enclosed" -> solid-rect sprite).
    padded = np.pad(cand, 1, constant_values=0)
    reached = (padded == 0).astype(np.uint8)
    cv2.floodFill(reached, np.zeros((padded.shape[0] + 2, padded.shape[1] + 2),
                                    np.uint8), (0, 0), 2)
    holes = reached[1:-1, 1:-1] == 1  # enclosed background
    filled = ((cand > 0) | holes).astype(np.uint8)
    # Leaf proper = within 1.2px of the strong core (leaf + its 1px
    # anti-alias edge) or hole-filled interior -> alpha 1.  The outer 2px
    # band (the JPEG chroma halo) gets <=0.4 semi-alpha — below the
    # original-RGB threshold in harden_sprites, so the tan blend is
    # repainted leaf-color instead of rendering as an opaque ring.
    # solid = leaf proper (near the strong core / enclosed interiors) plus
    # sealed paper slits — the source leaves are solid, so all of it is
    # alpha 1; only the outer chroma-halo band keeps a 0.4 feather, and both
    # non-leaf-proper color sources (paper slits, halo blend) get repainted
    solid = (((dist <= 1.2) | holes | (adopted > 0)) & (filled > 0))
    dist_in = cv2.distanceTransform(solid.astype(np.uint8), cv2.DIST_L2, 3)
    band_a = (np.clip(1.0 - dist_in / 2.0, 0, 1) * 0.4) * (filled > 0)
    core = np.maximum(solid.astype(np.float32), band_a)
    if core.sum() < 18:
        return None
    alpha = cv2.GaussianBlur(core, (0, 0), 0.7)
    alpha = np.clip((alpha - 0.25) / 0.6, 0, 1)  # gentle edge tighten
    rgb_crop = img[y0:y1, x0:x1].copy()
    fix = ((adopted > 0) | holes) & (filled > 0)
    if fix.any():
        rgb_crop = cv2.inpaint(rgb_crop, (fix * 255).astype(np.uint8), 3,
                               cv2.INPAINT_TELEA)
    repaint = (filled > 0) & (~solid) & (~fix)
    if repaint.any() and strong.sum() >= 5:
        med = np.median(rgb_crop[strong], axis=0)
        rgb_crop[repaint] = med
    rgba = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    fn = f"{prefix}_{k:02d}.png"
    cv2.imwrite(f"{OUT}/sprites/{fn}", rgba)
    w, h = int(x1 - x0), int(y1 - y0)
    ys2, xs2 = np.where(core > 0)
    return {"file": f"sprites/{fn}", "w": w, "h": h, "area": int(core.sum()),
            "strict": round(float(core[ys2, xs2].mean()), 3)}

def split_and_cut(mask, min_area, max_count, prefix):
    """split touching leaves: erode to separate, regrow within the original
    mask, then cut each leaf with a leaf-pixel alpha"""
    seeds = cv2.erode(mask, np.ones((3, 3), np.uint8))
    n, l, st, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    cands = sorted(((st[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
                    if st[i, cv2.CC_STAT_AREA] >= max(8, min_area // 2)),
                   reverse=True)
    out, seen_shapes = [], set()
    kk = np.ones((3, 3), np.uint8)
    for a, i in cands:
        if len(out) >= max_count:
            break
        seed = ((l == i).astype(np.uint8) * 255)
        grown = seed.copy()
        for _ in range(6):  # conditional regrow within the original mask
            grown = cv2.dilate(grown, kk) & mask
        grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        core = ((grown > 0) & (leafpix > 0))
        if core.sum() < min_area:
            continue
        ar = (st[i, cv2.CC_STAT_WIDTH] + 1) / (st[i, cv2.CC_STAT_HEIGHT] + 1)
        key = (int(core.sum()) // 40, round(min(ar, 1 / ar), 1))
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        entry = cut_leaf_sprite(core.astype(np.uint8) * 255, len(out), prefix)
        if entry:
            out.append(entry)
    return out

# single leaves from the removed airborne set, plus texture-only singles
# harvested from leaves that stay on the plate (over structures) and from
# the canopy interior — the pool needs shape diversity, not just 19
sp_air = split_and_cut(airborne_paper, min_area=22, max_count=44, prefix="leaf")
pool = list(sp_air)
extra_txt = split_and_cut(((airborne_static > 0).astype(np.uint8) * 255) | airborne_paper,
                          min_area=30, max_count=12, prefix="lx")
print(f"airborne sprites: {len(sp_air)}, texture extras: {len(extra_txt)}")
pool.extend(extra_txt)

# NOTE: harvesting canopy-interior disks was tried and removed — inside the
# dense canopy leafpix covers everything, so disks came out as full circles
# carrying gray shadow-gap texture (they rendered as gray plates in motion).
cy, cx = np.where(canopy0 > 0)
hsv_strict_g = (((hsv[:, :, 0] >= 3) & (hsv[:, :, 0] <= 31)) &
                (hsv[:, :, 1] > 95) & (hsv[:, :, 2] > 50))

# canopy cluster cutouts for the mega clouds (heavily blurred masses)
sprites_canopy = []
k = 0
tries = 0
while k < 26 and tries < 3000:
    tries += 1
    j = rng.integers(0, len(cy))
    y, x = cy[j], cx[j]
    r = int(rng.integers(16, 44))
    if y - r < 0 or y + r >= H or x - r < 0 or x + r >= W:
        continue
    disk = np.zeros((H, W), np.uint8)
    cv2.circle(disk, (int(x), int(y)), r, 255, -1)
    inside = ((canopy0 > 0) & (disk > 0)).sum()
    if inside < 0.55 * (disk > 0).sum() or inside == (disk > 0).sum():
        continue
    sub_m = ((canopy0 > 0) & (disk > 0))[y - r:y + r, x - r:x + r].astype(np.float32)
    alpha = cv2.GaussianBlur(sub_m, (0, 0), 1.4)
    rgba = cv2.cvtColor(img[y - r:y + r, x - r:x + r], cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
    fn = f"cl_{k:02d}.png"
    cv2.imwrite(f"{OUT}/sprites/{fn}", rgba)
    sprites_canopy.append({"file": f"sprites/{fn}", "w": 2 * r, "h": 2 * r})
    k += 1
print(f"canopy cluster sprites: {len(sprites_canopy)}")

# ---------- canopy layer RGBA ----------
ys, xs = np.where(canopy0 > 0)
bx0, bx1 = max(0, xs.min() - 4), min(W, xs.max() + 4)
by0, by1 = max(0, ys.min() - 4), min(H, ys.max() + 4)
alpha = cv2.GaussianBlur(canopy0[by0:by1, bx0:bx1].astype(np.float32) / 255.0, (0, 0), 1.2)
canopy_rgba = cv2.cvtColor(img[by0:by1, bx0:bx1], cv2.COLOR_BGR2BGRA)
canopy_rgba[:, :, 3] = (np.clip(alpha, 0, 1) * 255).astype(np.uint8)
cv2.imwrite(f"{OUT}/canopy.png", canopy_rgba)
canopy_bbox = {"x": int(bx0), "y": int(by0), "w": int(bx1 - bx0), "h": int(by1 - by0)}
print("canopy bbox:", canopy_bbox)

# ---------- erodible region (for measurement + erosion masks) ----------
er = cv2.GaussianBlur(erodible.astype(np.float32) / 255.0, (0, 0), 2)
cv2.imwrite("work/erodible.png", (er * 255).astype(np.uint8))
# binary protected mask aligned with erodible binary
erod_bin_true = erodible > 128
protected_bin = ((canopy0 > 0) & ~erod_bin_true).astype(np.uint8) * 255
cv2.imwrite("work/protected_roi.png", protected_bin)
eys, exs = np.where(erodible > 0)
erodible_bbox = {"x": int(exs.min()), "y": int(eys.min()),
                 "w": int(exs.max() - exs.min()), "h": int(eys.max() - eys.min())}

# ---------- manifests ----------
manifest = {
    "W": W, "H": H, "EXT": EXT,
    "girl": GIRL,
    "canopy_bbox": canopy_bbox,
    "erodible_bbox": erodible_bbox,
    "sprites_leaf": pool,
    "sprites_canopy": sprites_canopy,
}
with open("work/manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)

# debug visualization
vis = img.copy()
vis[airborne_paper > 0] = (0, 0, 255)      # red: removed airborne
vis[erodible > 0] = (255, 0, 0)            # blue: erodible canopy
vis[protected_canopy > 0] = (0, 255, 255)  # yellow: protected canopy
vis[airborne_static > 0] = (0, 255, 0)     # green: static airborne
cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), (255, 0, 255), 2)
cv2.imwrite("work/masks_vis.jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
cv2.imwrite("work/clean_plate_check.jpg",
            cv2.resize(plate, (plate.shape[1] // 2, plate.shape[0] // 2)),
            [cv2.IMWRITE_JPEG_QUALITY, 85])
print("done")
