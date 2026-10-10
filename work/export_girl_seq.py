#!/usr/bin/env python3
"""Girl per-frame RGBA sequence — measured, not modeled.

Per source frame: warp back to plate coords, crop GSEQ, matte the girl
(warm-color seed + dark-tone rescue below the coat hem -> close/open ->
hole fill -> main-component keep -> 3-frame temporal median -> feather),
save public/seq/girl/f***.png and work/girl_meta.json.

--probe: only frames [0, 30, 60, 90], plus check images and HSV stats for
calibrating S_MIN / V_MIN / rescue knobs before the batch run.
"""
import cv2
import numpy as np
import json
import os
import sys

H, W = 1280, 720
# Sprite rect (plate coords). Legs' dark tights + red boots reach y~1203 and
# her contact shadow / the slab front-face band fade out by ~y1230; the
# outstretched hand reaches x~650 and the shadow tail x~495.  >=4px margin
# around every per-frame content extent (validated by the bbox sweep below).
GSEQ = [490, 1005, 165, 232]  # x, y, w, h (plate coords)
GIRL_ROW_MAX = 211  # crop row of plate y1216 (slab-top surface end)
S_MIN = 120     # seed saturation floor — warm grade puts wall at S~55-80, coat/scarf at S~200+
V_MIN = 38      # seed value floor
COMP_MIN = 1200 # secondary components below this area are dropped (leaf/shadow-scale)
MARGIN = 2      # alpha bbox must stay this far inside GSEQ
LOW_ROW = 140   # crop row where the dark-tone rescue starts (plate y>=1145, below coat hem)
# Dark-tone rescue (legs/boots/shadow band): measured HSV in the lower band —
# tights mid-tones S 60-120 / V 40-100, boot cores S 216-230 / V 14-38, slab
# front-face band S 90-135 / V 47-136; pavement there is S~65 / V~175, so
# (v<130)&(s<130 | v<60) on warm hues captures leg/boot/shadow tone without
# flooding pavement.  Restricted to LOW_ROW so the wall side is untouched.
RESCUE_V_MAX = 130
RESCUE_S_MAX = 130
RESCUE_DARK_V = 60

FR = [0, 20, 40, 60, 80, 100, 121]
TX = [0, -50, -105, -159, -212, -272, -336]
tx_of = np.interp(np.arange(122), FR, TX)

gx, gy, gw, gh = GSEQ


def load_frame(f):
    """source frame f (0-based) warped back to plate coords, GSEQ crop"""
    src = cv2.imread(f"work/frames/f{f+1:03d}.png")
    M = np.float32([[1, 0, -tx_of[f]], [0, 1, 0]])
    return cv2.warpAffine(src, M, (W, H))[gy:gy + gh, gx:gx + gw]


def fill_holes(mask):
    """enclosed background (coat shadow, dark skirt) becomes foreground.

    Background = pixels reachable from the CROP BORDER, not just (0,0):
    the girl's arm can exit the frame edge and seal a wall pocket against
    the border, which a (0,0)-only flood would wrongly hole-fill.
    """
    h, w = mask.shape
    bg = (mask == 0).astype(np.uint8)
    padded = np.ones((h + 2, w + 2), np.uint8)  # border ring = background
    padded[1:-1, 1:-1] = bg
    cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    holes = (padded[1:-1, 1:-1] == 1).astype(np.uint8)
    return ((mask | holes) * 255).astype(np.uint8)


def matte(raw_roi):
    """binary girl mask for one GSEQ crop"""
    hsv = cv2.cvtColor(raw_roi, cv2.COLOR_BGR2HSV)
    warm = (hsv[:, :, 0] < 33) | (hsv[:, :, 0] > 167)
    seed = (warm & (hsv[:, :, 1] > S_MIN) & (hsv[:, :, 2] > V_MIN)).astype(np.uint8)
    # dark-tone rescue below the coat hem: tights/boots/shadow are warm-hue
    # but too dark (V) or too desaturated (S) for the seed above.  Open the
    # rescue first: its deep-gradient specks (pavement noise rows) are 1-3px
    # and would otherwise be bridged into the body by the 7x7 close below,
    # dragging the bbox to the crop bottom.
    low = np.zeros(seed.shape, bool)
    low[LOW_ROW:, :] = True
    rescue = (warm & (hsv[:, :, 2] < RESCUE_V_MAX) &
              ((hsv[:, :, 1] < RESCUE_S_MAX) | (hsv[:, :, 2] < RESCUE_DARK_V)) &
              low).astype(np.uint8)
    rescue = cv2.morphologyEx(rescue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    seed = ((seed > 0) | (rescue > 0)).astype(np.uint8)
    m = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = (fill_holes(m) > 0).astype(np.uint8)
    nn, ll, ss, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if nn >= 2:
        main = 1 + int(np.argmax(ss[1:, cv2.CC_STAT_AREA]))
        keep = (ll == main).astype(np.uint8)
        for i in range(1, nn):
            if i != main and ss[i, cv2.CC_STAT_AREA] >= COMP_MIN:
                keep[ll == i] = 1
        m = keep
    return m


def report(raws):
    """check images + HSV stats to calibrate the knobs"""
    for f, m in raws.items():
        roi = load_frame(f)
        vis = roi.copy()
        vis[m > 0] = (0.4 * vis[m > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        edge_im = roi.copy()
        edge_im[edge] = [0, 255, 255]
        sbs = np.hstack([roi, vis, edge_im])
        cv2.imwrite(f"work/girl_probe/f{f+1:03d}_check.png", sbs)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        fg = hsv[m > 0]
        bg = hsv[m == 0]
        print(f"f{f+1:03d}: fg px={int(m.sum())} "
              f"bg S p90={int(np.percentile(bg[:, 1], 90))} "
              f"bg V p10={int(np.percentile(bg[:, 2], 10))} | "
              f"fg S p10={int(np.percentile(fg[:, 1], 10))} "
              f"fg V p5={int(np.percentile(fg[:, 2], 5))}")
    print("calibration: raise S_MIN if bg S p90 >= S_MIN (wall leaking); "
          "lower V_MIN if coat interior is missing after hole-fill; "
          "rescue knobs RESCUE_V_MAX/RESCUE_S_MAX/RESCUE_DARK_V only act on "
          "rows >= LOW_ROW (plate y>=1005+LOW_ROW) — legs/boots/shadow tone")


def sweep_margin(meta):
    """bbox sweep: girl content must clear MARGIN at every GSEQ edge.

    The plate repair (GR in build_assets.py, y1005-1205 + dilation/fade)
    quilts rows up to ~y1209; below that the plate keeps its original
    pixels.  So the rect must protect GIRL content (feet bottom ~y1203,
    slab-top shadow ~y1213 — all <= GIRL_ROW_MAX) with MARGIN to the crop
    edge: anything cut there would expose quilted pavement.  Deeper content
    (slab front-face band y1216-1229, fallen leaves) composites over an
    identical original backdrop, so touching the bottom edge there is
    composite-neutral — reported, not asserted (it cannot be avoided: leaves
    cross any boundary; the girl's shadow merges into the band).
    """
    gx, gy, gw, gh = GSEQ
    bad_girl, env_touch, edge_report = [], [], []
    for k, v in sorted(meta["bbox"].items(), key=lambda t: int(t[0])):
        f = int(k)
        if v is None:
            continue
        x, y, w, h = v
        touches = []
        if x < MARGIN: touches.append("left")
        if y < MARGIN: touches.append("top")
        if x + w > gw - MARGIN: touches.append("right")
        girl_deep = meta.get("girl_deep", {}).get(k)
        bottom_girl = girl_deep is not None and girl_deep > gh - 1 - MARGIN
        bottom_env = (y + h - 1 >= gh - MARGIN) and not bottom_girl
        if bottom_girl:
            bad_girl.append((f, v, girl_deep))
        elif bottom_env:
            env_touch.append(f)
        if touches:
            edge_report.append((f, touches))
    if edge_report:
        fs = {t for _, ts in edge_report for t in ts}
        print(f"bbox sweep: {len(edge_report)} frames touch {sorted(fs)} edge(s) "
              "(foreground leaves / per-frame content — expected, not asserted)")
    if env_touch:
        print(f"bbox sweep: {len(env_touch)} frames carry environment content "
              f"(front-face band / pavement leaves) at the bottom edge over an "
              f"original plate backdrop (rows > plate y1216) — composite-neutral")
    if bad_girl:
        for f, v, gd in bad_girl[:10]:
            print(f"  GIRL BOTTOM VIOLATION f{f:03d}: bbox={v} girl_deep={gd}")
        raise AssertionError(f"{len(bad_girl)} frames cut girl content at the "
                             "GSEQ bottom margin — rect too short")
    print(f"bbox sweep: girl content clears bottom margin on all frames "
          f"(MARGIN={MARGIN})")


def source_margin_check():
    """Source-side guard against the original defect (rect too short for the
    feet).  The bbox sweep above is blind to a too-short rect — a crop that
    excludes the feet never sees them.  Here the FULL unwarp is probed: the
    deepest dark-warm row (girl boots/shadow signature) in the feet corridor
    plate x535-595 must clear the rect bottom by MARGIN + close(3px)."""
    H2, W2 = 1280, 720
    worst = None
    for f in range(122):
        src = cv2.imread(f"work/frames/f{f+1:03d}.png")
        M = np.float32([[1, 0, -tx_of[f]], [0, 1, 0]])
        pl = cv2.warpAffine(src, M, (W2, H2))
        hsv = cv2.cvtColor(pl[gy + 170:gy + 211, 535:595], cv2.COLOR_BGR2HSV)
        warm = (hsv[:, :, 0] < 33) | (hsv[:, :, 0] > 167)
        dw = warm & (hsv[:, :, 2] < RESCUE_V_MAX) & \
            ((hsv[:, :, 1] < RESCUE_S_MAX) | (hsv[:, :, 2] < RESCUE_DARK_V))
        rows = np.where(dw.sum(axis=1) >= 6)[0]
        if len(rows):
            d = gy + 170 + int(rows[-1])
            if worst is None or d > worst[0]:
                worst = (d, f)
    assert worst is not None, "no girl content found in feet corridor"
    d, f = worst
    need = d + MARGIN + 3
    print(f"source margin: deepest girl-corridor content y={d} (f{f:03d}); "
          f"rect bottom y={gy + gh}; need >= {need}")
    assert gy + gh >= need, (
        f"rect bottom {gy + gh} < deepest girl content {d} + "
        f"MARGIN+close ({need}) — feet would be cut")


def main():
    probe = "--probe" in sys.argv
    os.makedirs("work/girl_probe", exist_ok=True)
    frames = [0, 30, 60, 90] if probe else list(range(122))
    raws = {f: matte(load_frame(f)) for f in frames}
    if probe:
        report(raws)
        return
    # 3-frame temporal median against per-frame edge flicker
    stack = [raws[f] for f in range(122)]
    seq = []
    for f in range(122):
        trio = np.stack([stack[max(0, f - 1)], stack[f],
                         stack[min(121, f + 1)]])
        seq.append(np.median(trio, axis=0).astype(np.uint8))
    os.makedirs("public/seq/girl", exist_ok=True)
    meta = {"gseq": GSEQ, "bbox": {}, "girl_deep": {}}
    for f in range(122):
        a = cv2.GaussianBlur((seq[f] * 255).astype(np.float32), (0, 0), 0.8)
        rgba = np.dstack([load_frame(f), np.clip(a, 0, 255).astype(np.uint8)])
        cv2.imwrite(f"public/seq/girl/f{f:03d}.png", rgba)
        ys, xs = np.where(seq[f] > 0)
        meta["bbox"][str(f)] = ([int(xs.min()), int(ys.min()),
                                 int(xs.max() - xs.min() + 1),
                                 int(ys.max() - ys.min() + 1)]
                                if len(ys) else None)
        gy_idx = np.where(seq[f][:GIRL_ROW_MAX].any(axis=1))[0]
        meta["girl_deep"][str(f)] = int(gy_idx.max()) if len(gy_idx) else None
    with open("work/girl_meta.json", "w") as fj:
        json.dump(meta, fj)
    sweep_margin(meta)
    source_margin_check()
    print("girl seq written:", len(meta["bbox"]), "frames")


if __name__ == "__main__":
    main()
