# 女孩逐帧动画 + 树顶摆动 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 成片中女孩跟随原片逐帧动起来（抬手/转身/走步/围巾），树顶按实测曲线 ±1~2px 摆动。

**Architecture:** 延续"测量而非建模"管线：新增 `work/export_girl_seq.py` 从源视频逐帧抠出女孩 RGBA 序列；`build_assets.py` 在 plate 写盘前用墙面/地面 quilting 修补掉烤死的女孩；`build_canopy_seq.py` 把实测 sway 曲线烘进 canopy 序列（spawn/metrics 之后，保证既有产物逐字节不变）；Remotion 侧加一层逐帧 girl Img。Spec: `docs/superpowers/specs/2026-10-09-girl-motion-tree-sway-design.md`。

**Tech Stack:** Python `.venv`（opencv/numpy），Node 24 + Remotion 4，ffmpeg。

## Global Constraints

- 分辨率 720×1280 @24fps，122 帧，`DURATION=122`，不得改动。
- 一切运动参数从源视频实测（`work/frames/f001.png`~`f122.png`，view 坐标；反 warp 用 `M=[[1,0,-tx_of[f]]]`，`FR=[0,20,40,60,80,100,121]`、`TX=[0,-50,-105,-159,-212,-272,-336]`）。
- 确定性：所有随机必须来自带 seed 的 generator。**`build_assets.py` 的修改块必须用独立 seed 的局部 rng（g_rng），不得消耗全局 `rng`（seed 42）的抽取序列，否则下游 sprites/canopy 全部漂移。**
- 回归硬约束：Task 3 后 sprites/canopy.png 逐字节不变、plate 仅女孩区变化；Task 4 后 `canopy_spawn.json` 逐字节不变、`remain` 数组不变。
- 无新第三方依赖。
- 本项目"测试"= 管线惯例的 check 图 + 数值断言（如 `clean_plate_check.jpg`、`verify_metrics.py`），无 pytest。每个任务的验证步骤都给出确切命令和期望输出。
- 提交信息风格：`type: 描述`（feat:/fix:/docs:）。
- 所有命令的工作目录都是 `<repo root>`。

---

### Task 0: git 初始化 + 基线提交

项目尚不是 git 仓库，后续任务用 commit 做检查点，需要先建库。

**Files:**
- Create: `.gitignore`

**Interfaces:**
- Consumes: 无
- Produces: git 仓库，基线 commit 包含当前全部源码/脚本/文档（不含大文件产物）

- [ ] **Step 1: 写 .gitignore**

```gitignore
.venv/
node_modules/
.DS_Store
work/frames/
work/dbg_*
work/mine/
out/
deliverables/
*.pyc
```

- [ ] **Step 2: git init + 基线提交**

```bash
git init
git add -A
git commit -m "chore: baseline before girl-motion + tree-sway work"
```

Expected: 提交成功，`git log --oneline` 显示 1 条基线提交。

---

### Task 1: 女孩抠图 probe（3 关键帧标定）

先在 f1（站立垂手）/ f31（抬手转身）/ f61（走步）三帧上标定抠图阈值，再批量。check 图输出到 `work/girl_probe/`，人眼 1:1 验收。

**Files:**
- Create: `work/export_girl_seq.py`

**Interfaces:**
- Consumes: `work/frames/f*.png`（view 坐标源帧）、`FR`/`TX` 平移曲线
- Produces: `matte()`/`load_frame()` 函数（Task 2 复用同一脚本）；`GSEQ=[495,1005,145,200]` 矩形常量（Task 2 写入 `girl_meta.json`）

- [ ] **Step 1: 写脚本（完整文件）**

```python
#!/usr/bin/env python3
"""Girl per-frame RGBA sequence — measured, not modeled.

Per source frame: warp back to plate coords, crop GSEQ, matte the girl
(warm-color seed -> close/open -> hole fill -> main-component keep ->
3-frame temporal median -> feather), save public/seq/girl/f***.png and
work/girl_meta.json.

--probe: only frames [0, 30, 60], plus check images and HSV stats for
calibrating S_MIN / V_MIN / COMP_MIN before the batch run.
"""
import cv2
import numpy as np
import json
import os
import sys

H, W = 1280, 720
GSEQ = [495, 1005, 145, 200]  # x, y, w, h (plate coords), validated by bbox sweep
S_MIN = 42      # seed saturation floor — cream wall sits below, coat/scarf above
V_MIN = 38      # seed value floor
COMP_MIN = 400  # secondary components below this area are dropped (leaf-scale)
MARGIN = 2      # alpha bbox must stay this far inside GSEQ

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
    """enclosed background (coat shadow, dark skirt) becomes foreground"""
    h, w = mask.shape
    bg = (mask == 0).astype(np.uint8)
    reached = bg.copy()
    cv2.floodFill(reached, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 2)
    holes = (reached == 1).astype(np.uint8)
    return ((mask | holes) * 255).astype(np.uint8)


def matte(raw_roi):
    """binary girl mask for one GSEQ crop"""
    hsv = cv2.cvtColor(raw_roi, cv2.COLOR_BGR2HSV)
    warm = (hsv[:, :, 0] < 33) | (hsv[:, :, 0] > 167)
    seed = (warm & (hsv[:, :, 1] > S_MIN) & (hsv[:, :, 2] > V_MIN)).astype(np.uint8)
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
          "lower V_MIN if coat interior is missing after hole-fill")


def main():
    probe = "--probe" in sys.argv
    os.makedirs("work/girl_probe", exist_ok=True)
    frames = [0, 30, 60] if probe else list(range(122))
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
    meta = {"gseq": GSEQ, "bbox": {}}
    for f in range(122):
        a = cv2.GaussianBlur((seq[f] * 255).astype(np.float32), (0, 0), 0.8)
        rgba = np.dstack([load_frame(f), np.clip(a, 0, 255).astype(np.uint8)])
        cv2.imwrite(f"public/seq/girl/f{f:03d}.png", rgba)
        ys, xs = np.where(seq[f] > 0)
        meta["bbox"][str(f)] = ([int(xs.min()), int(ys.min()),
                                 int(xs.max() - xs.min() + 1),
                                 int(ys.max() - ys.min() + 1)]
                                if len(ys) else None)
    with open("work/girl_meta.json", "w") as fj:
        json.dump(meta, fj)
    print("girl seq written:", len(meta["bbox"]), "frames")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 probe**

Run: `.venv/bin/python work/export_girl_seq.py --probe`
Expected: 打印 3 行 `f001/f031/f061: fg px=... bg S p90=...` 统计 + calibration 提示；`work/girl_probe/` 出现 3 张 check 图。

- [ ] **Step 3: 标定阈值（如需要）**

判读规则（对 3 张 `f*_check.png`，中图=遮罩覆盖，右图=边缘）：
1. 墙面/地面大片变红（bg S p90 ≥ S_MIN）→ `S_MIN` 上调 +6，重跑 Step 2；
2. 大衣/围巾内部成洞且 hole-fill 没救回 → `V_MIN` 下调 8，重跑；
3. 女孩身旁悬空叶子被抠进 alpha（不接触她轮廓的红斑）→ `COMP_MIN` 上调至 1200，重跑；
4. 围巾细尾断裂 → `COMP_MIN` 下调 200，重跑。
每次改后重跑 probe，直到三帧均满足：她全身连续、无墙面泄漏、悬空杂斑 ≤2 个小点。

- [ ] **Step 4: 1:1 人眼验收**

Run: 打开 `work/girl_probe/f031_check.png` 全屏 1:1 检查（缩图会制造伪影假象）。
Expected: 边缘无白圈/灰晕，围巾轮廓完整，抬手姿态完整。不达标回到 Step 3。

- [ ] **Step 5: Commit**

```bash
git add work/export_girl_seq.py
git commit -m "feat: girl matte probe (3-frame calibration)"
```

---

### Task 2: 女孩 122 帧批量导出

**Files:**
- Modify: `work/export_girl_seq.py`（仅当 Task 1 标定改了阈值，改常量即可）
- Create: `public/seq/girl/f000.png`~`f121.png`、`work/girl_meta.json`（产物）

**Interfaces:**
- Consumes: Task 1 的脚本与标定好的 `S_MIN/V_MIN/COMP_MIN`
- Produces: `public/seq/girl/f***.png`（GSEQ 尺寸 RGBA）；`work/girl_meta.json`（`{"gseq":[x,y,w,h],"bbox":{...}}`，Task 3 的矩形来源、Task 5 的常量来源）

- [ ] **Step 1: 批量运行**

Run: `.venv/bin/python work/export_girl_seq.py`
Expected: 打印 `girl seq written: 122 frames`；`public/seq/girl/` 下 122 个文件。

- [ ] **Step 2: bbox 边界断言（防姿态出界）**

```bash
.venv/bin/python - <<'EOF'
import json
meta = json.load(open("work/girl_meta.json"))
gx, gy, gw, gh = meta["gseq"]
MARGIN = 2
bad = []
for f, b in meta["bbox"].items():
    if b is None:
        bad.append((f, "empty")); continue
    x, y, w, h = b
    if x < MARGIN or y < MARGIN or x + w > gw - MARGIN or y + h > gh - MARGIN:
        bad.append((f, b))
print("frames touching margin:", bad if bad else "none")
assert not bad, f"GSEQ too small, enlarge and rerun: {bad[:5]}"
EOF
```

Expected: `frames touching margin: none`。若断言失败：`GSEQ` 外扩 10px（相应 `build_assets.py` 的 `GR` 在 Task 3 同步外扩），重跑本步。

- [ ] **Step 3: 连续帧 contact sheet（查闪烁）**

```bash
.venv/bin/python - <<'EOF'
import cv2, numpy as np
tiles = []
for f in [0, 15, 30, 45, 60, 75, 90, 105, 121]:
    im = cv2.imread(f"public/seq/girl/f{f:03d}.png", cv2.IMREAD_UNCHANGED)
    bg = np.full_like(im[:, :, :3], 235)
    a = im[:, :, 3:].astype(np.float32) / 255
    comp = (im[:, :, :3] * a + bg * (1 - a)).astype(np.uint8)
    tiles.append(comp)
row = [np.pad(t, ((0, 0), (2, 2), (0, 0)), constant_values=255) for t in tiles]
cv2.imwrite("work/girl_sheet.jpg", np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("sheet written")
EOF
```

Expected: `work/girl_sheet.jpg` 生成；打开检查 9 个姿态连续变化、边缘无明显逐帧跳闪（时间中值应已压平；若仍闪，将 `S_MIN` 上调 4 重跑 Task 1 Step 2 + 本步）。

- [ ] **Step 4: Commit**

```bash
git add work/export_girl_seq.py work/girl_meta.json
git commit -m "feat: export 122-frame girl RGBA sequence"
```

（`public/` 视 .gitignore 策略：若 public/ 未被忽略则一并提交。）

---

### Task 3: plate 女孩区修补

`plate.png` 烤着静止女孩，叠加动画层会露出"旧姿势的影子"。在 `build_assets.py` 写盘前用 quilting 修补女孩矩形（墙上部用纸纹 bank、地面下部用 row-lock 地板 bank）。**插入点：`cv2.imwrite(f"{OUT}/plate.png", plate)`（约 line 377）之前。**

**Files:**
- Modify: `work/build_assets.py`（在 plate 写盘前插入一段）
- Test: 数值断言 + `work/girl_plate_check.png` 1:1 验收

**Interfaces:**
- Consumes: 现有全局 `leaf/pile/airborne_paper/paper_zone/img/plate/quilt/normalize_bank`、`GSEQ`（Task 2 的 `girl_meta.json`）
- Produces: 新 `public/plate.png`（仅女孩区变化）；`work/girl_plate_check.png`

- [ ] **Step 1: 备份现产物用于回归对比**

```bash
cp public/plate.png work/plate_prev.png
```

- [ ] **Step 2: 在 `cv2.imwrite(f"{OUT}/plate.png", plate)` 之前插入**

```python
# ---------- girl region repair ----------
# The girl becomes a per-frame sprite (work/export_girl_seq.py); the baked
# copy must leave the plate or her old pose ghosts out behind the animation.
# Wall part quilts from a paper bank, pavement part from a row-locked
# pavement bank.  Dedicated rng: must NOT shift the global rng sequence or
# every downstream asset (sprites/canopy) would drift.
GR = (495, 1005, 145, 200)   # keep in sync with GSEQ in export_girl_seq.py
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
pave_g = normalize_bank(_bank((SPLIT, H - 72), np.zeros((H, W), np.uint8)))
print(f"girl repair banks: paper={len(paper_g)} pave={len(pave_g)}")
assert len(paper_g) > 5 and len(pave_g) > 5, "empty quilt bank — widen zone"

def _repair_part(zone_m, bank, row_lock):
    fill = quilt((H, W), zone_m, bank, block=26, feather=9, row_lock=row_lock)
    soft = cv2.GaussianBlur(zone_m.astype(np.float32) / 255.0, (0, 0), 3)
    pf = plate.astype(np.float32)
    pf[:, :W] = pf[:, :W] * (1 - soft[..., None]) + fill * soft[..., None]
    return pf

pf = _repair_part(gr_wall, paper_g, False)
plate = np.clip(_repair_part(gr_pave, pave_g, True), 0, 255).astype(np.uint8)
cv2.imwrite("work/girl_plate_check.png",
            plate[gry - 24:gry + grh + 24, grx - 24:grx + grw + 24])
print("girl region repaired")
```

注意：`_repair_part` 第二次调用时 `plate` 已被第一部分覆盖——第二次 `pf = plate.astype(np.float32)` 重新读取，两段各自动融合，SPLIT 交界处 wall/pave 软遮罩各淡出 ~9px。

- [ ] **Step 3: 重跑资产链**

```bash
.venv/bin/python work/build_assets.py && .venv/bin/python work/harden_sprites.py
```

Expected: 打印含 `girl repair banks: paper=... pave=...` 和 `girl region repaired`；无 assert 报错。若 pave bank 不足 6 个：把 `_bank((SPLIT, H - 72), ...)` 的窗口改成 56px（`y1-72`→`y1-56`，切片同步改），重跑。

- [ ] **Step 4: 回归断言（只有女孩区允许变化）**

```bash
.venv/bin/python - <<'EOF'
import cv2, numpy as np
a = cv2.imread("work/plate_prev.png")
b = cv2.imread("public/plate.png")
assert a.shape == b.shape, "plate size changed?!"
d = np.abs(a.astype(int) - b.astype(int)).max(axis=2)
zone = np.zeros(a.shape[:2], bool)
zone[1005 - 18:1205 + 18, 495 - 18:640 + 18] = True
print("max diff outside girl zone:", int(d[~zone].max()))
assert d[~zone].max() == 0, "plate changed outside girl region — rng sequence shifted"
# sprites / canopy 不许变（harden 幂等性警告：build_assets 重生成后 harden 只跑一次）
print("OK: plate changed only in girl region")
EOF
```

Expected: `max diff outside girl zone: 0` + `OK`。非 0 → 插入代码消耗了全局 rng（检查是否误用 `rng` 而非 `g_rng`），修复后重跑 Step 3-4。

- [ ] **Step 5: 1:1 人眼验收**

打开 `work/girl_plate_check.png`：女孩应完全消失，墙面/地面纹理连续，无接缝、无亮斑、无结构残影。与 `work/plate_prev.png` 同区域并排对比。不达标：调 `_bank` 排除项或 `quilt(block=26, feather=9)` → `feather=14` 重跑 Step 3-5。

- [ ] **Step 6: Commit**

```bash
git add work/build_assets.py
git commit -m "feat: quilt girl region out of clean plate"
```

---

### Task 4: 树顶 sway 实测 + 烘入 canopy 序列

连续帧相位相关（相邻帧消解差异极小，相关干净），去平移、逐步截幅 ±0.8px、中值平滑、去均值后 cumsum 成逐帧偏移曲线，在 spawn/metrics 计算完**之后**重写序列帧。Remotion 零改动。

**Files:**
- Modify: `work/build_canopy_seq.py`（主循环后、meta 写盘前插入）

**Interfaces:**
- Consumes: `work/frames/f*.png`、`tx_of`、已写出的 `public/seq/canopy/f*.png`
- Produces: sway 后的 `public/seq/canopy/f***.png`；`work/canopy_meta.json` 增加 `sway_dx/sway_dy/sway_method` 字段；`canopy_spawn.json`、`remain` 保持逐字节不变

- [ ] **Step 1: 备份产物**

```bash
cp work/canopy_spawn.json work/canopy_spawn_prev.json
cp work/canopy_meta.json work/canopy_meta_prev.json
```

- [ ] **Step 2: 修改 meta 写盘 + 插入 sway 段**

把文件末尾的

```python
with open("work/canopy_meta.json", "w") as fj:
    json.dump({"remain": remain}, fj)
```

改为：

```python
# ---------- tree-top sway: measured, applied AFTER metrics ----------
# Consecutive-frame phase correlation on the canopy top strip (neighbor
# dissolve deltas are tiny), pan-compensated; clip per-step jumps, median-
# smooth, mean-removed cumsum -> per-frame offset vs f0.  Runs after
# spawn/remain are computed so those outputs stay byte-identical.
def measure_sway():
    y0, y1, x0, x1 = 70, 320, 20, 700
    prev = cv2.cvtColor(cv2.imread("work/frames/f001.png")[y0:y1, x0:x1],
                        cv2.COLOR_BGR2GRAY).astype(np.float32)
    steps = [(0.0, 0.0)]
    for f in range(1, 122):
        dtx = float(tx_of[f] - tx_of[f - 1])
        cur = cv2.imread(f"work/frames/f{f+1:03d}.png")[y0:y1, x0:x1]
        cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)
        a = cv2.warpAffine(prev, np.float32([[1, 0, -dtx], [0, 1, 0]]),
                           (x1 - x0, y1 - y0))
        (dx, dy), _ = cv2.phaseCorrelate(a, cur)
        # real sway steps are well under a pixel; dissolve/leaf events clip
        dx = float(np.clip(dx, -0.8, 0.8))
        dy = float(np.clip(dy, -0.8, 0.8))
        steps.append((dx, dy))
        prev = cur
    sm = np.array(steps, np.float32)
    k = 9
    pad = np.pad(sm, ((k // 2, k // 2), (0, 0)), mode="edge")
    sm = np.median(np.stack([pad[i:i + len(sm)] for i in range(k)]), axis=0)
    dx = np.cumsum(sm[:, 0]); dy = np.cumsum(sm[:, 1])
    dx -= dx.mean(); dy -= dy.mean()
    return dx.astype(np.float32), dy.astype(np.float32)

sway_dx, sway_dy = measure_sway()
print(f"sway range: dx [{sway_dx.min():.2f},{sway_dx.max():.2f}] "
      f"dy [{sway_dy.min():.2f},{sway_dy.max():.2f}]")
for f in range(122):
    p = f"public/seq/canopy/f{f:03d}.png"
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    M = np.float32([[1, 0, sway_dx[f]], [0, 1, sway_dy[f]]])
    cv2.imwrite(p, cv2.warpAffine(im, M, (im.shape[1], im.shape[0]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101))

with open("work/canopy_meta.json", "w") as fj:
    json.dump({"remain": remain,
               "sway_dx": [round(float(v), 3) for v in sway_dx],
               "sway_dy": [round(float(v), 3) for v in sway_dy],
               "sway_method": "consecutive phase-corr, clip±0.8/step, "
                              "median-9, mean-removed"}, fj)
print("done")
```

（替换原来末尾的 `print("done")`。）

- [ ] **Step 3: 重跑**

Run: `.venv/bin/python work/build_canopy_seq.py`
Expected: 打印原日志 + `sway range: dx [.., ..] dy [.., ..]` + `done`。sway 幅度应在 ±0.5~±2px 内（与既有实测 ±1~1.5px 一致）。若 >3px：测量被污染，退回 spec 兜底——正弦拟合（`sway_dx = 1.2*np.sin(np.arange(122)/122*2*np.pi*2)`，即 5 秒 2 个周期、幅 1.2px），并在 meta 的 `sway_method` 注明 `sinusoidal fallback`。

- [ ] **Step 4: 回归断言**

```bash
.venv/bin/python - <<'EOF'
import json
assert json.load(open("work/canopy_spawn_prev.json")) == \
       json.load(open("work/canopy_spawn.json")), "spawn points changed"
a = json.load(open("work/canopy_meta_prev.json"))["remain"]
b = json.load(open("work/canopy_meta.json"))["remain"]
assert a == b, "remain changed"
print("OK: spawn/remain identical")
EOF
```

Expected: `OK: spawn/remain identical`。失败 → sway 段被放进了主循环内，移到 meta 写盘前、主循环后。

- [ ] **Step 5: 输出帧 sway 抽检**

```bash
.venv/bin/python - <<'EOF'
import cv2, numpy as np
base = cv2.cvtColor(cv2.imread("public/seq/canopy/f000.png"),
                    cv2.COLOR_BGR2GRAY).astype(np.float32)
vals = []
for f in [20, 40, 60, 80, 100, 120]:
    g = cv2.cvtColor(cv2.imread(f"public/seq/canopy/f{f:03d}.png"),
                     cv2.COLOR_BGR2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(base, g)
    vals.append((f, dx, dy))
    print(f"f{f:03d}: dx={dx:+.2f} dy={dy:+.2f}")
mx = max(max(abs(dx), abs(dy)) for _, dx, dy in vals)
assert mx > 0.5, "sway not visible in output frames"
print("sway present, max", round(mx, 2))
EOF
```

Expected: 各帧 dx 在 ±2px 内摆动且方向有正有负，`sway present`。

- [ ] **Step 6: Commit**

```bash
git add work/build_canopy_seq.py
git commit -m "feat: bake measured tree-top sway into canopy sequence"
```

---

### Task 5: Remotion 接入女孩层

**Files:**
- Modify: `work/gen_constants.py`（导出 `GSEQ`）
- Modify: `src/constants.ts`（重跑 gen_constants 后自动更新，勿手改）
- Modify: `src/AutumnScene.tsx`（import `GSEQ` + camera group 内加一层）

**Interfaces:**
- Consumes: `work/girl_meta.json`（Task 2）、`public/seq/girl/f***.png`
- Produces: `GSEQ` 常量（`{x,y,w,h}`）；成片中 plate→canopy→girl→粒子的层序

- [ ] **Step 1: gen_constants.py 导出 GSEQ**

在 `gen_constants.py` 顶部 `meta = json.load(...)` 之后加：

```python
girl_meta = json.load(open("work/girl_meta.json"))
```

在 ts 模板 `export const GIRL = ...` 行之后插入一行：

```
export const GSEQ = {{x: {girl_meta['gseq'][0]}, y: {girl_meta['gseq'][1]}, w: {girl_meta['gseq'][2]}, h: {girl_meta['gseq'][3]}}};
```

- [ ] **Step 2: AutumnScene.tsx 加 import**

import 块 `CANOPY,` 之后加 `GSEQ,`：

```tsx
import {
  W,
  H,
  DURATION,
  PAN_FR,
  PAN_TX,
  CANOPY,
  GSEQ,
  SPRITES_LEAF,
  SPRITES_BOKEH,
  SPRITES_CLUSTER,
  DENS_FR,
  DENS_PX,
  WIND,
} from "./constants";
```

- [ ] **Step 3: camera group 内、canopy Img 之后加 girl 层**

在 `<Img src={staticFile(\`seq/canopy/...\`)} ... />` 的闭合 `/>` 之后、`</AbsoluteFill>` 之前插入：

```tsx
        <Img
          src={staticFile(
            `seq/girl/f${String(frame).padStart(3, "0")}.png`
          )}
          style={{
            position: "absolute",
            left: GSEQ.x,
            top: GSEQ.y,
            width: GSEQ.w,
            height: GSEQ.h,
          }}
        />
```

- [ ] **Step 4: 静态检查 + 重生成常量**

```bash
.venv/bin/python work/gen_constants.py && npx tsc --noEmit
```

Expected: `constants.ts written`；tsc 无输出（通过）。

- [ ] **Step 5: 女孩动作 probe 渲染**

```bash
./node_modules/.bin/remotion still Autumn out/probe_girl_f0.png --frame=0 --log=error
./node_modules/.bin/remotion still Autumn out/probe_girl_f31.png --frame=31 --log=error
./node_modules/.bin/remotion still Autumn out/probe_girl_f61.png --frame=61 --log=error
.venv/bin/python - <<'EOF'
import cv2
tiles = []
for f in [0, 31, 61]:
    im = cv2.imread(f"out/probe_girl_f{f}.png")
    tiles.append(im[985:1225, 455:680])
cv2.imwrite("work/girl_render_check.png", cv2.hstack(tiles))
EOF
```

Expected: `work/girl_render_check.png` 三联图——f0 站立、f31 抬手、f61 走步，姿态逐帧跟随源片；女孩身后无"旧姿势残影"（若出现重影 → Task 3 修补区没盖住底板旧女孩，回 Task 3 Step 5）。1:1 核验边缘无灰晕。

- [ ] **Step 6: Commit**

```bash
git add work/gen_constants.py src/constants.ts src/AutumnScene.tsx
git commit -m "feat: per-frame girl layer in Remotion scene"
```

---

### Task 6: 全量渲染 + 验收 + 交付归档 + README

**Files:**
- Modify: `README.md`
- Create: `out/autumn_final.mp4`（新版）、`deliverables/*`、`out/versions/*`（产物）

**Interfaces:**
- Consumes: Task 2-5 全部产物
- Produces: 新终版 + 验收记录（`work/qa_girl_motion.jpg`、`work/sway_check.log` 输出）

- [ ] **Step 1: 轮换 prev，渲染终版，封装音轨，归档**

```bash
cp out/autumn_final.mp4 out/autumn_prev.mp4
./node_modules/.bin/remotion render Autumn out/autumn_final.mp4 --log=error
ffmpeg -i out/autumn_final.mp4 -i inputs/reference-video.mp4 \
  -map 0:v -map 1:a -c:v copy -c:a copy -shortest -y deliverables/autumn-remotion-final.mp4
cp out/autumn_final.mp4 "out/versions/autumn_final_$(date +%m%d_%H%M).mp4"
```

Expected: 渲染 122 帧无错误；versions/ 新增带时间戳存档。

- [ ] **Step 2: 指标回归**

```bash
.venv/bin/python work/verify_metrics.py
```

Expected: 镜头平移/密度曲线/消解曲线指标与既有基线相当（无系统性劣化）。若消解指标异常 → 检查 sway 是否意外作用于 alpha 语义（sway 只是整体平移，不应影响 remain）。

- [ ] **Step 3: 女孩+摆动 vs 原片并排验收**

```bash
.venv/bin/python - <<'EOF'
import cv2, numpy as np, subprocess
FR = [0, 20, 40, 60, 80, 100, 121]
TX = [0, -50, -105, -159, -212, -272, -336]
tx_of = np.interp(np.arange(122), FR, TX)
rows = []
for f in [0, 30, 60, 90, 121]:
    cap = cv2.VideoCapture("out/autumn_final.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, mine = cap.read(); cap.release()
    src = cv2.imread(f"work/frames/f{f+1:03d}.png")
    # undo pan on source so both are plate coords
    srcp = cv2.warpAffine(src, np.float32([[1, 0, -tx_of[f]],[0,1,0]]), (720, 1280))
    gx, gy, gw, gh = 455, 985, 225, 240   # girl neighborhood crop
    rows.append(np.hstack([srcp[gy:gy+gh, gx:gx+gw],
                           mine[gy:gy+gh, gx:gx+gw],
                           np.full((gh, 6, 3), 255, np.uint8)]))
cv2.imwrite("work/qa_girl_motion.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("qa written")
EOF
```

Expected: `work/qa_girl_motion.jpg` 五行三列（左源片/右成片，均为 plate 坐标）。逐行 1:1 验收：姿态一致（f1 站立/f31 抬手/f61、f91 走步/f122 收尾）、树顶区域连续帧有微动。轻微相位差（±1 帧内）可接受。

- [ ] **Step 4: f0 不劣化断言**

```bash
ffmpeg -y -loglevel error -i out/versions/autumn_final_1009_1441.mp4 -vf "select=eq(n\,0)" -vframes 1 /tmp/old_f0.png
ffmpeg -y -loglevel error -i out/autumn_final.mp4 -vf "select=eq(n\,0)" -vframes 1 /tmp/new_f0.png
.venv/bin/python - <<'EOF'
import cv2, numpy as np
a = cv2.imread("/tmp/old_f0.png"); b = cv2.imread("/tmp/new_f0.png")
d = np.abs(a.astype(int) - b.astype(int)).max(axis=2)
mask = np.ones(a.shape[:2], bool)
mask[1005-18:1205+18, 495-18:640+18] = False   # girl zone may change
print("outside girl zone: mean", round(float(d[mask].mean()), 3),
      "max", int(d[mask].max()))
assert d[mask].mean() < 1.0 and d[mask].max() < 12, "f0 regressed outside girl zone"
print("OK: f0 unchanged outside girl zone (encode noise only)")
EOF
```

Expected: `OK`。sway 在 f0 为 0（曲线相对 f0 去均值后不以 f0 为零点——若 f0 sway 非零导致树冠区 max 偏高，属预期内 sway 位移，目测 `/tmp/new_f0.png` 与 old 并排无退步即可放行，把断言改为 mean < 1.0 only 并在提交信息注明）。

- [ ] **Step 5: README 更新**

- 管线 ④ 后加一行：`.venv/bin/python work/export_girl_seq.py        # → public/seq/girl/f*.png（女孩逐帧 alpha）`
- 目录结构 `public/` 注释补：`seq/girl 女孩逐帧序列`
- 已知妥协：删除"女孩为静态（源视频有走步/抬手动画）——可从源视频剪关键姿态做叠加。"，替换为：
  `- 女孩为逐帧抠像叠加（色度 matte），发丝级边缘与原片有差距；她的投影仍为静态。`
- 渲染小节补存档命令注释保持原样，另在"关键设计决策"追加一行：
  `- **动态元素实测**：女孩逐帧 matte、树顶 sway 曲线均测自源视频；plate 女孩区 quilting 修补，防旧姿势残影。`

```bash
git add README.md
git commit -m "docs: girl motion + tree sway in pipeline"
```

- [ ] **Step 6: 收尾清理**

```bash
rm -f out/probe_girl_f*.png /tmp/old_f0.png /tmp/new_f0.png
git status --short   # 确认无遗漏
```

Expected: 干净。向用户报告：新终版路径、QA 图路径、验收结论（含跳过/妥协项）。

---

## Self-Review 记录

1. **Spec 覆盖**：女孩逐帧（Task 1-3、5）、清底板（Task 3）、sway 烘入（Task 4）、常量+场景（Task 5）、验收 6 条（Task 6 Step 2/3/4 + Task 1 Step 4 边缘核验 + Task 5 Step 4 tsc + 渲染确定性由固定 seed 保证）、README（Task 6 Step 5）——全覆盖。
2. **Placeholder 扫描**：无 TBD/TODO；阈值标定给判据与调整方向，非占位。
3. **类型/命名一致性**：`GSEQ`（meta `gseq`）↔ Task 3 `GR`（注明同步）；`girl_meta.json` 结构在 Task 2 定义、Task 5 消费一致；`canopy_spawn_prev.json` 备份名前后一致。
