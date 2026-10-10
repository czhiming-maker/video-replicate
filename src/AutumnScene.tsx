import React, { useMemo } from "react";
import { AbsoluteFill, Img, useCurrentFrame, staticFile } from "remotion";
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
import spawnData from "./spawn_points.json";
import heroSeqData from "./hero_seq.json";
import initialLeaves from "./initial_leaves.json";

// ---------------- hero leaves: per-frame source sequences ----------------
// 2 hand-picked source leaves cut per-frame by SAM2 video segmentation
// (work/export_hero_seq.py); position/size are measured per frame, so these
// carry the source's real flipping/curling/motion-blur — no procedural
// tumble applied to this layer.
const HERO_SEQS = heroSeqData as Record<
  string,
  {
    f0: number;
    f1: number;
    pos: Record<string, { cx: number; cy: number; w: number; h: number }>;
  }
>;

// ---------------- deterministic random ----------------
const hash = (s: string): number => {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967295;
};
const rand = (seed: string, lo = 0, hi = 1): number =>
  lo + hash(seed) * (hi - lo);

// leaf tumble: measured from source f072-f086 — a big hero leaf's apparent
// aspect swings 1.0 -> 6.5 -> 1.4 over ~12 frames (edge-on foreshortening)
// and darkens as it turns edge-on.  Period/phase drawn per leaf from rand.
const TUMBLE = { tmin: 8, tmax: 16, squash: 0.3, dim: 0.75, expand: 0.12 };
const tumbleAt = (
  id: string,
  t: number
): { squash: number; expand: number; bright: number } => {
  const T = TUMBLE.tmin + rand(id + ":tT") * (TUMBLE.tmax - TUMBLE.tmin);
  const th = (6.283 * t) / T + rand(id + ":tP") * 6.283;
  const s = 0.5 * (1 + Math.cos(th)); // 1 -> 0 -> 1 over one cycle
  const squash = 1 - (1 - TUMBLE.squash) * s;
  return {
    squash,
    expand: 1 + TUMBLE.expand * s,
    bright: TUMBLE.dim + (1 - TUMBLE.dim) * (1 - s),
  };
};

const interp = (xs: number[], ys: number[], x: number): number => {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let i = 0;
  while (i < xs.length - 1 && xs[i + 1] < x) i++;
  const t = (x - xs[i]) / (xs[i + 1] - xs[i]);
  return ys[i] + t * (ys[i + 1] - ys[i]);
};

const panAt = (f: number): number => interp(PAN_FR, PAN_TX, f);
const MAXD = Math.max(...DENS_PX);

// ---------------- particle types ----------------
type Sprite = { file: string; w: number; h: number };
type P = {
  id: string;
  sprite: Sprite;
  scale: number;
  layer: number; // 0 far, 1 mid, 2 near
  blur: number;
  opacity: number;
  spawnFrame: number;
  x0: number;
  y0: number;
  vx: number;
  vy: number;
  ax: number;
  aw: number;
  ap: number;
  ay1: number;
  yw1: number;
  yp1: number;
  ay2: number;
  yw2: number;
  yp2: number;
  rot0: number;
  spin: number;
  flip: boolean;
};

const mkStormParticle = (id: string, sprite?: Sprite, scale?: number): P => {
  let spr: Sprite;
  let scl: number;
  let layer: number;
  let blur = 0;
  let opacity = 1;
  let spm: number;
  if (sprite && scale !== undefined) {
    spr = sprite;
    scl = scale;
    layer = scale > 1.3 ? 2 : scale > 0.7 ? 1 : 0;
    spm = 0.8 + layer * 0.22 + rand(id + ":v") * 0.2;
  } else {
    const r = rand(id + ":sp");
    spr = SPRITES_LEAF[Math.floor(Math.pow(r, 1.3) * SPRITES_LEAF.length)];
    const sz = rand(id + ":sz");
    if (sz < 0.5) {
      scl = 0.45 + rand(id + ":s0") * 0.45;
      layer = 0;
      opacity = 0.82 + rand(id + ":o0") * 0.18;
      spm = 0.72 + rand(id + ":v0") * 0.26;
    } else if (sz < 0.88) {
      scl = 0.9 + rand(id + ":s1") * 0.8;
      layer = 1;
      spm = 0.95 + rand(id + ":v1") * 0.3;
    } else {
      scl = 2.0 + rand(id + ":s2") * 1.4;
      layer = 2;
      blur = 0.5 + rand(id + ":b2") * 1.1;
      spm = 1.22 + rand(id + ":v2") * 0.45;
    }
  }
  return {
    id,
    sprite: spr,
    scale: scl,
    layer,
    blur,
    opacity,
    spawnFrame: -1,
    x0: 0,
    y0: 0,
    vx: WIND.vx * spm * (0.85 + rand(id + ":wx") * 0.3),
    vy: WIND.vy * spm * (0.65 + rand(id + ":wy") * 0.7),
    ax: 12 + rand(id + ":a1") * 26,
    aw: 0.05 + rand(id + ":w1") * 0.09,
    ap: rand(id + ":p1") * 6.283,
    ay1: 7 + rand(id + ":a2") * 16,
    yw1: 0.06 + rand(id + ":w2") * 0.1,
    yp1: rand(id + ":p2") * 6.283,
    ay2: 3 + rand(id + ":a3") * 9,
    yw2: 0.13 + rand(id + ":w3") * 0.14,
    yp2: rand(id + ":p3") * 6.283,
    rot0: rand(id + ":r0") * 360,
    spin: (rand(id + ":rs") - 0.5) * 7,
    flip: rand(id + ":fl") < 0.5,
  };
};

const posAt = (p: P, f: number) => {
  const t = f - p.spawnFrame;
  return {
    x: p.x0 + p.vx * t + p.ax * Math.sin(t * p.aw + p.ap),
    y:
      p.y0 +
      p.vy * t +
      p.ay1 * Math.sin(t * p.yw1 + p.yp1) +
      p.ay2 * Math.sin(t * p.yw2 + p.yp2),
    rot: p.rot0 + p.spin * t,
  };
};

const aliveAt = (p: P, f: number): boolean => {
  const { x, y } = posAt(p, f);
  return x > -120 && x < 840 && y > -120 && y < 1400;
};

// ---------------- initial field: real leaves from the static image ----------------
const pickSpriteFor = (id: string, w: number, h: number): [Sprite, number] => {
  const area = w * h;
  let best = SPRITES_LEAF[0];
  let bestScore = Infinity;
  for (const s of SPRITES_LEAF) {
    const sa = s.w * s.h;
    const ar = Math.min(w, h) / Math.max(w, h);
    const sr = Math.min(s.w, s.h) / Math.max(s.w, s.h);
    const score =
      Math.log(Math.max(sa, 20) / Math.max(area, 20)) ** 2 +
      (ar - sr) ** 2 * 4;
    if (score < bestScore) {
      bestScore = score;
      best = s;
    }
  }
  const scale = Math.min(
    2.4,
    Math.max(0.3, Math.sqrt(area / (best.w * best.h)))
  );
  return [best, scale * (0.92 + rand(id + ":vs") * 0.16)];
};

const INITIAL: P[] = initialLeaves.map((l, i) => {
  const id = `init${i}`;
  const [sprite, scale] = pickSpriteFor(id, l.w, l.h);
  const p = mkStormParticle(id, sprite, scale);
  p.spawnFrame = 0;
  p.x0 = l.x;
  p.y0 = l.y;
  p.rot0 = rand(id + ":ir") * 360;
  return p;
});

// ---------------- schedule: refill from canopy edge per measured density ----------------
const MAX_ACTIVE = 720;
const POOL: P[] = Array.from({ length: 1600 }, (_, i) =>
  mkStormParticle(`p${i}`)
);
let poolIdx = 0;
const scheduled: P[] = [...INITIAL];
const simActive: P[] = [...INITIAL];
const spawnMap = spawnData as unknown as Record<string, [number, number][]>;
for (let f = 0; f < DURATION; f++) {
  for (let k = simActive.length - 1; k >= 0; k--) {
    if (!aliveAt(simActive[k], f)) simActive.splice(k, 1);
  }
  const want =
    Math.round((MAX_ACTIVE * interp(DENS_FR, DENS_PX, f)) / MAXD) -
    simActive.length;
  const pts = spawnMap[String(f)] ?? [];
  const tx = panAt(f);
  for (let k = 0; k < want && poolIdx < POOL.length; k++) {
    const p = POOL[poolIdx++];
    p.spawnFrame = f;
    if (pts.length > 0 && rand(p.id + ":src") < 0.82) {
      const pt = pts[Math.floor(rand(p.id + ":pi") * pts.length)];
      p.x0 = pt[0] + tx + (rand(p.id + ":jx") - 0.5) * 56;
      p.y0 = pt[1] + (rand(p.id + ":jy") - 0.5) * 56;
    } else {
      p.x0 = rand(p.id + ":ax") * 940 - 60;
      p.y0 = 30 + rand(p.id + ":ay") * 1080;
    }
    scheduled.push(p);
    simActive.push(p);
  }
}

// ---------------- bokeh (foreground, out of focus) ----------------
// solid leaf sprites only: cluster cuts carry paper gaps that turn into
// murky fog when blurred.  The source's late frames are full of 50-250px
// foreground leaves, so the pool is large with two size tiers.
const BOKEH_SPRITES = SPRITES_BOKEH.length > 0 ? SPRITES_BOKEH : SPRITES_LEAF;
const BOKEH: P[] = Array.from({ length: 30 }, (_, i) => {
  const id = `bok${i}`;
  const sprite = BOKEH_SPRITES[Math.floor(rand(id + ":sp") * BOKEH_SPRITES.length)];
  const hero = i >= 22;
  const p = mkStormParticle(
    id,
    sprite,
    hero ? 3.2 + rand(id + ":sc") * 1.6 : 2.2 + rand(id + ":sc") * 1.4
  );
  p.spawnFrame = Math.round(3 + (i * 112) / 30 + rand(id + ":sf") * 7);
  p.x0 = -120 - rand(id + ":x") * 260;
  p.y0 = 60 + rand(id + ":y") * 1080;
  p.vx = 12 + rand(id + ":vx") * 9;
  p.vy = (rand(id + ":vy") - 0.4) * 7;
  p.blur = hero ? 2.8 + rand(id + ":bl") * 1.7 : 3 + rand(id + ":bl") * 3;
  p.opacity = 0.85 + rand(id + ":op") * 0.15;
  p.spin = (rand(id + ":rs") - 0.5) * 16;
  return p;
});

// ---------------- ground skitterers ----------------
const SKITTER: P[] = Array.from({ length: 12 }, (_, i) => {
  const id = `sk${i}`;
  const sprite = SPRITES_LEAF[Math.floor(rand(id + ":sp") * 14)];
  const p = mkStormParticle(id, sprite, 0.5 + rand(id + ":sc") * 0.5);
  p.spawnFrame = Math.round(i * 8 + rand(id + ":sf") * 6);
  p.x0 = -60 + rand(id + ":x") * 320;
  p.y0 = 1150 + rand(id + ":y") * 110;
  p.vx = 9 + rand(id + ":vx") * 7;
  p.vy = 0;
  p.opacity = 0.95;
  return p;
});

const skitterY = (p: P, f: number): number => {
  const t = f - p.spawnFrame;
  return p.y0 - Math.abs(Math.sin(t * 0.33 + p.ap)) * 14;
};

// ---------------- hero wave: the source's signature late-shot mass of
// huge blurred deep-red leaves sweeping past the camera ----------------
const HERO_SPRITES = SPRITES_BOKEH.length > 3 ? SPRITES_BOKEH : SPRITES_LEAF;
const HERO: P[] = Array.from({ length: 9 }, (_, i) => {
  const id = `hw${i}`;
  const sprite = HERO_SPRITES[Math.floor(rand(id + ":sp") * HERO_SPRITES.length)];
  const p = mkStormParticle(id, sprite, 3.4 + rand(id + ":sc") * 3.2);
  p.spawnFrame = Math.round(96 + i * 2.6 + rand(id + ":sf") * 3);
  p.x0 = -150 - rand(id + ":x") * 300;
  p.y0 = -40 + rand(id + ":y") * 700;
  p.vx = 11 + rand(id + ":vx") * 8;
  p.vy = 1 + rand(id + ":vy") * 4;
  // blur calibrated to source late leaves (Laplacian var 283-401: moderately
  // sharp, leaf shape always readable — 7-13px turned them into amorphous
  // blobs the source doesn't have)
  p.blur = 3.5 + rand(id + ":bl") * 2;
  p.opacity = 0.92;
  p.spin = (rand(id + ":rs") - 0.5) * 18;
  return p;
});
// ---------------- scene ----------------
export const AutumnScene: React.FC = () => {
  const frame = useCurrentFrame();
  const tx = panAt(frame);

  const stormEls = useMemo(() => {
    return scheduled
      .filter((p) => p.spawnFrame <= frame && aliveAt(p, frame))
      .sort((a, b) => a.layer - b.layer)
      .map((p) => {
        const { x, y, rot } = posAt(p, frame);
        const tw = tumbleAt(p.id, frame - p.spawnFrame);
        return (
          <Img
            key={p.id}
            src={staticFile(p.sprite.file)}
            style={{
              position: "absolute",
              left: x - (p.sprite.w * p.scale) / 2,
              top: y - (p.sprite.h * p.scale) / 2,
              width: p.sprite.w * p.scale,
              height: p.sprite.h * p.scale,
              transform: `rotate(${rot}deg) scaleX(${
                (p.flip ? -1 : 1) * tw.expand
              }) scaleY(${tw.squash})`,
              opacity: p.opacity,
              filter: `brightness(${tw.bright})${
                p.blur > 0 ? ` blur(${p.blur}px)` : ""
              }`,
              willChange: "transform",
            }}
          />
        );
      });
  }, [frame]);

  const bokehEls = BOKEH.filter((p) => p.spawnFrame <= frame && aliveAt(p, frame)).map(
    (p) => {
      const { x, y, rot } = posAt(p, frame);
      const tw = tumbleAt(p.id, frame - p.spawnFrame);
      return (
        <Img
          key={p.id}
          src={staticFile(p.sprite.file)}
          style={{
            position: "absolute",
            left: x - (p.sprite.w * p.scale) / 2,
            top: y - (p.sprite.h * p.scale) / 2,
            width: p.sprite.w * p.scale,
            height: p.sprite.h * p.scale,
            transform: `rotate(${rot}deg) scaleX(${tw.expand}) scaleY(${tw.squash})`,
            opacity: p.opacity,
            filter: `blur(${p.blur}px) brightness(${tw.bright})`,
          }}
        />
      );
    }
  );

  const heroEls = HERO.filter((p) => p.spawnFrame <= frame && aliveAt(p, frame)).map(
    (p) => {
      const { x, y, rot } = posAt(p, frame);
      const tw = tumbleAt(p.id, frame - p.spawnFrame);
      return (
        <Img
          key={p.id}
          src={staticFile(p.sprite.file)}
          style={{
            position: "absolute",
            left: x - (p.sprite.w * p.scale) / 2,
            top: y - (p.sprite.h * p.scale) / 2,
            width: p.sprite.w * p.scale,
            height: p.sprite.h * p.scale,
            transform: `rotate(${rot}deg) scaleX(${tw.expand}) scaleY(${tw.squash})`,
            opacity: p.opacity,
            filter: `blur(${p.blur}px) saturate(1.35) hue-rotate(-8deg) brightness(${(0.9 * tw.bright).toFixed(3)})`,
          }}
        />
      );
    }
  );


  const skitterEls = SKITTER.filter(
    (p) => p.spawnFrame <= frame && posAt(p, frame).x < 840
  ).map((p) => {
    const { x, rot } = posAt(p, frame);
    const y = skitterY(p, frame);
    const tw = tumbleAt(p.id, frame - p.spawnFrame);
    return (
      <Img
        key={p.id}
        src={staticFile(p.sprite.file)}
        style={{
          position: "absolute",
          left: x - (p.sprite.w * p.scale) / 2,
          top: y - (p.sprite.h * p.scale) / 2,
          width: p.sprite.w * p.scale,
          height: p.sprite.h * p.scale,
          transform: `rotate(${rot}deg) scaleX(${tw.expand}) scaleY(${tw.squash})`,
          opacity: p.opacity,
          filter: `brightness(${tw.bright})`,
        }}
      />
    );
  });

  const grainX = (hash(`gx${frame}`) - 0.5) * 12;
  const grainY = (hash(`gy${frame}`) - 0.5) * 12;

  return (
    <AbsoluteFill style={{ backgroundColor: "#cfc9ba", overflow: "hidden" }}>
      {/* camera group: plate + canopy in plate coords */}
      <AbsoluteFill
        style={{
          transform: `translateX(${tx}px)`,
          width: W + 380,
          overflow: "hidden",
        }}
      >
        <Img
          src={staticFile("plate.png")}
          style={{ position: "absolute", left: 0, top: 0, width: W + 380, height: H }}
        />
        {/* branch sway: measured per-frame warp of the trunk/branch region
            (work/export_branch_warp.py) — plate-coords overlay, alpha is the
            feathered branch mask so the wall never moves */}
        <Img
          src={staticFile(
            `seq/branch/warp_f${String(frame).padStart(3, "0")}.png`
          )}
          style={{ position: "absolute", left: 0, top: 380, width: 470, height: 900 }}
        />
        <Img
          src={staticFile(
            `seq/canopy/f${String(frame).padStart(3, "0")}.png`
          )}
          style={{
            position: "absolute",
            left: CANOPY.x,
            top: CANOPY.y,
            width: CANOPY.w,
            height: CANOPY.h,
          }}
        />
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
      </AbsoluteFill>

      {stormEls}
      {skitterEls}
      {[...Object.entries(HERO_SEQS)].map(([name, seq]) => {
        const p = seq.pos[String(frame)];
        if (!p) return null;
        return (
          <Img
            key={`hseq${name}`}
            src={staticFile(
              `seq/hero/${name}/f${String(frame).padStart(3, "0")}.png`
            )}
            style={{
              position: "absolute",
              // this layer sits OUTSIDE the camera group (view coords), so
              // the pan that the group applies to plate-coord layers must be
              // added manually here
              left: p.cx + tx - p.w / 2,
              top: p.cy - p.h / 2,
              width: p.w,
              height: p.h,
            }}
          />
        );
      })}
      {bokehEls}
      {heroEls}

      <Img
        src={staticFile("grain.png")}
        style={{
          position: "absolute",
          left: grainX,
          top: grainY,
          width: W,
          height: H,
          opacity: 0.055,
        }}
      />
    </AbsoluteFill>
  );
};
