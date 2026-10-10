# video-replicate

视频复刻管线：给定一张静态原画 + 一段 AI 生成参考视频，用 Remotion 程序化重建该视频——
**不重画画，只重做运动；一切运动参数从源视频实测，而非建模猜测。**

首个案例：`inputs/` 中的纸雕风秋日街景（Pinterest 静态图 → AI 图生视频，5.08s，720×1280@24fps）。

## 目录结构

```
inputs/            参考素材（静态原图 + 参考视频）
deliverables/      成品输出（终版 / 上一版 / 新旧对比视频）
src/               Remotion 场景
  AutumnScene.tsx    场景本体（相机组 + 粒子系统 + 颗粒）
  constants.ts       自动生成的常量（相机路径/精灵清单/实测曲线），勿手改
  spawn_points.json  树冠消解边缘逐帧采样点（粒子出生点）
  initial_leaves.json 第 0 帧初始叶场（静态图真实叶位置）
work/              Python 资产管线 + 测量脚本 + 中间产物
public/            渲染消费的资产（底板/树冠序列/精灵/颗粒；seq/girl 女孩逐帧序列）
out/               渲染输出；out/versions/ 版本存档
analysis/          初版（纯矢量重绘路线）的遗留分析，仅存档
```

## 管线（按执行顺序）

环境：`.venv`（Python: opencv/numpy/scipy），Node 24 + Remotion 4。

```bash
# ① 相机标定：模板匹配/ECC/链式 ORB → 相机路径
.venv/bin/python work/calibrate_chain.py

# ② 资产管线：掩码分割 → 清底板 → 精灵 atlas → 树冠 RGBA
.venv/bin/python work/build_assets.py        # 生成 public/plate.png + sprites/ + canopy.png
.venv/bin/python work/harden_sprites.py      # alpha 硬化 + 孔洞填充（防灰晕/白圈）
.venv/bin/python work/build_extension_from_video.py  # 底板右扩展：源视频反 warp 时间中值
.venv/bin/python work/export_initial_leaves.py       # 第 0 帧真实叶位置
# 注：全量重跑需按此顺序跑两遍（build_assets 的地板修补 bank 依赖 ext_video_median.png）或先跑 build_extension 再重跑 build_assets

# ③ 源视频测量：消解曲线 / 密度曲线 / 光流风场
.venv/bin/python work/measure_source.py      # → work/motion.json

# ④ 树冠消解序列（测量式）+ 常量导出
.venv/bin/python work/build_canopy_seq.py    # → public/seq/canopy/f*.png (122帧)
.venv/bin/python work/export_girl_seq.py     # → public/seq/girl/f*.png（女孩逐帧 alpha）
.venv/bin/python work/gen_constants.py       # → src/constants.ts（消费 work/girl_meta.json）

# ⑤ 渲染 + 原声封装
npx tsc --noEmit
./node_modules/.bin/remotion render Autumn out/autumn_final.mp4 --log=error
ffmpeg -i out/autumn_final.mp4 -i inputs/reference-video.mp4 \
  -map 0:v -map 1:a -c:v copy -c:a copy -shortest -y deliverables/autumn-remotion-final.mp4
cp out/autumn_final.mp4 "out/versions/autumn_final_$(date +%m%d_%H%M).mp4"  # 版本存档
```

单帧预览：`./node_modules/.bin/remotion render Autumn out/probe.mp4 --frames=59-60`

## 关键设计决策

- **混合架构**：原画做底板，程序只做运动。矢量重绘路线已验证不可行（`analysis/` 存档）。
- **测量而非建模**：相机路径（纯水平左移）、树冠消解形态（逐帧叶色掩码）、风场（光流）、
  密度曲线全部实测自源视频。建模式消解曾导致树冠形态与原片不符。
- **精灵切割四防**：腐蚀分离防黏连、叶色 alpha 防投影烙印、距强色度核心距离约束 +
  外圈 0.4 羽化防 JPEG 色环晕圈（非叶心像素一律叶色中值重涂）、封闭孔洞/缝隙收编防纸底露馅。
- **确定性**：粒子全部 `random(seed)`，同代码必出同片。
- **验收**：渲染后抽帧并排 + 1:1 原生分辨率核验（缩图 JPEG 会制造伪影假象）+ 指标对比
  （镜头平移/密度曲线/消解曲线，`work/verify_metrics.py`）。
- **动态元素实测**：女孩逐帧 matte、树顶 sway 曲线、主角叶 SAM2 序列（4 片大叶逐帧抠像，
  位置逐帧实测）、树枝摆动（三分带光流中位数 + 羽化 warp 贴片，±2.5px）均测自源视频；
  plate 女孩区 quilting 修补，防旧姿势残影。
  SAM2 环境：`.sam2/`（transformers Sam2VideoModel + MPS，独立 venv，见 work/export_hero_seq.py）。
- **回归基线语义**：canopy 序列的 spawn/remain 参照 = 改动前脚本（024d6ab）在当前输入上的重跑结果，而非存储文件（0c67541 的旧 spawn/meta 由旧输入生成，不可复现）；plate 修补的参照 = `work/plate_prev.png`（0c67541 原始底板）。

## 可调参数

- `src/AutumnScene.tsx` 顶部：`MAX_ACTIVE`（粒子上限）、`POOL`、bokeh/mega/hero/skitter 各层
  数量、尺度、模糊、风场倍率；`TUMBLE` 叶子翻面参数（全层，实测自原片 f072-f086：
  周期 8-16 帧、侧对压缩至 0.3、边缘态变暗至 0.75）。mega 层已移除：原片末秒右上
  穿越的是中小清晰叶（2-8k px），弥焦大团与原片不符。
- `work/build_canopy_seq.py`：叶色判定阈值（影响消解激进度）。
- `work/measure_source.py`：测量口径。

## 素材版权声明

`inputs/` 中的静态原画来自 Pinterest（[pin 6227018031](https://huaban.com/pins/6227018031) 为检索来源之一），
版权归原画作者所有；AI 参考视频由该原画生成。二者仅作为本仓库的复刻研究对象与
学习演示用途，请勿商用；如原作者提出要求请联系移除。

## 已知妥协

- 女孩为逐帧抠像叠加（色度 matte），发丝级边缘与原片有差距；她的投影仍为静态。
- 逐叶轨迹是程序生成，统计特征对齐、不逐帧对齐（帧级 100% 原理上不可行）。
- 末段前景巨叶层的体积感略弱于源片。
