#!/usr/bin/env python3
"""
Import AFUN eval_real predictions into static/results/ for the web gallery.

Each AffU prediction folder looks like:
  <results>/<scene>_<query_words>/
      meta.json       (with: ckpt, mode, input, query, frame, prep_dir)
      pred.npz        (mask, points_N50, score, motion_type, spline_ctrl, ...)
      pred_seg.png
      pred_3d.html

The matching scene input lives at:
  <input>/<scene>/
      cam_K.txt
      color/<frame>.png
      depth/<frame>.npy

This script copies/derives the per-entry files the website JS needs:
  static/results/<entry_id>/
      rgb.png                  -- the scene RGB at the prediction frame
      depth.npy                -- depth at that frame (for reference)
      intrinsics.json          -- {K, fx, fy, cx, cy, height, width}
      instruction.txt          -- the language query
      prediction.npz           -- copied (renamed) from pred.npz
      trajectory_50pts.txt     -- 50 points, one "x y z" per line
      trajectory_50pts.npy     -- same array, .npy for convenience
      point_cloud.npz          -- {points_xyz, colors_rgb} back-projected from RGB-D
      pointcloud_small.bin     -- downsampled binary used by the in-browser viewer
                                  (matches scripts/preprocess_pointclouds.py format)

Run from the project root:
  python3 scripts/import_eval_real.py \
      --pred-dir /nfs/turbo/coe-jungaocv/wzn/workspace/AffU/eval_real/results/combined_model_step_106800 \
      --eval-dir /nfs/turbo/coe-jungaocv/wzn/workspace/AffU/eval_real \
      --scene kitchen_1

Add multiple --scene flags to import several scenes in one shot.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# Mirror the binary layout used by scripts/preprocess_pointclouds.py so the
# in-browser viewer can read either source identically.
TARGET_POINTS = 10000
MASK_TARGET_FRAC = 0.15
Z_MIN = 0.05
Z_MAX = 8.0
SEED = 42


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return s or "query"


def list_scene_predictions(pred_dir: Path, scene: str) -> list[Path]:
    """All prediction folders whose name starts with `<scene>_`."""
    prefix = f"{scene}_"
    return sorted(p for p in pred_dir.iterdir() if p.is_dir() and p.name.startswith(prefix))


def load_intrinsics_txt(path: Path) -> np.ndarray:
    K = np.loadtxt(path)
    if K.shape != (3, 3):
        raise ValueError(f"unexpected intrinsics shape {K.shape} in {path}")
    return K.astype(np.float64)


def backproject(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float32)
    m = (z > 0) & np.isfinite(z) & (z < 8.0)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xs - cx) / fx * z
    Y = (ys - cy) / fy * z
    pts = np.stack([X[m], Y[m], z[m]], axis=1).astype(np.float32)
    cols = rgb[m].astype(np.uint8)
    return pts, cols


def filter_valid(points, colors):
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] > Z_MIN)
        & (points[:, 2] < Z_MAX)
    )
    return points[valid], colors[valid]


def project_mask_flags(points, K, mask_hw):
    H, W = mask_hw.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = points[:, 2]
    u = np.rint(points[:, 0] / z * fx + cx).astype(np.int32)
    v = np.rint(points[:, 1] / z * fy + cy).astype(np.int32)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    flags = np.zeros(len(points), dtype=np.uint8)
    u_clip = np.clip(u, 0, W - 1)
    v_clip = np.clip(v, 0, H - 1)
    sampled = mask_hw[v_clip, u_clip].astype(np.uint8)
    flags[inside] = sampled[inside]
    return flags


def mask_aware_downsample(points, colors, mask_flags, target, mask_frac, rng):
    n = len(points)
    if n <= target:
        return points, colors, mask_flags
    masked_idx = np.where(mask_flags > 0)[0]
    unmasked_idx = np.where(mask_flags == 0)[0]
    want_masked = min(int(round(target * mask_frac)), len(masked_idx))
    want_unmasked = min(target - want_masked, len(unmasked_idx))
    if want_unmasked + want_masked < target and len(masked_idx) > want_masked:
        want_masked = min(target - want_unmasked, len(masked_idx))
    pick_mask = rng.choice(masked_idx, want_masked, replace=False) if want_masked else np.array([], dtype=int)
    pick_unmask = rng.choice(unmasked_idx, want_unmasked, replace=False) if want_unmasked else np.array([], dtype=int)
    idx = np.concatenate([pick_unmask, pick_mask])
    rng.shuffle(idx)
    return points[idx], colors[idx], mask_flags[idx]


def write_pointcloud_bin(path: Path, points, colors, mask_flags):
    n = len(points)
    with open(path, "wb") as f:
        f.write(np.uint32(n).tobytes())
        f.write(points.astype(np.float32, copy=False).tobytes())
        f.write(colors.astype(np.uint8, copy=False).tobytes())
        f.write(mask_flags.astype(np.uint8, copy=False).tobytes())


def import_one(pred_folder: Path, eval_dir: Path, out_root: Path, rng):
    meta = json.loads((pred_folder / "meta.json").read_text())
    query = meta["query"]
    frame = meta["frame"]
    input_rel = meta["input"]
    scene_dir = (eval_dir / Path(input_rel).relative_to("eval_real")).resolve() \
        if input_rel.startswith("eval_real") else (eval_dir / input_rel).resolve()
    # Fallback: <eval_dir>/_prep/<scene_name>
    if not scene_dir.is_dir():
        scene_name = Path(input_rel).name
        scene_dir = (eval_dir / "_prep" / scene_name).resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene input dir not found: tried {scene_dir}")

    rgb_path = scene_dir / "color" / f"{frame}.png"
    depth_path = scene_dir / "depth" / f"{frame}.npy"
    K_path = scene_dir / "cam_K.txt"

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path).astype(np.float32)
    # Heuristic: depth is sometimes stored in millimeters. Anything with a
    # median above 50 (i.e. > 50 m if it were meters) is almost certainly mm
    # — convert to meters so the in-browser viewer's Z_MIN/Z_MAX filter works.
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size and np.median(finite) > 50.0:
        depth = depth / 1000.0
    K = load_intrinsics_txt(K_path)

    # The entry_id matches the source folder name (keeps cross-reference simple).
    entry_id = pred_folder.name
    out = out_root / entry_id
    out.mkdir(parents=True, exist_ok=True)

    # Copy / convert inputs.
    Image.fromarray(rgb).save(out / "rgb.png")
    np.save(out / "depth.npy", depth)
    H, W = depth.shape
    intr = {
        "K": K.tolist(),
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "height": int(H), "width": int(W),
    }
    (out / "intrinsics.json").write_text(json.dumps(intr, indent=2))
    (out / "instruction.txt").write_text(query.strip() + "\n")

    # Copy prediction.
    pred = np.load(pred_folder / "pred.npz", allow_pickle=True)
    np.savez(out / "prediction.npz", **{k: pred[k] for k in pred.files})
    traj = pred["points_N50"].astype(np.float32)
    np.save(out / "trajectory_50pts.npy", traj)
    with open(out / "trajectory_50pts.txt", "w") as f:
        for x, y, z in traj:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    # Point cloud.
    pts_all, cols_all = backproject(rgb, depth, K)
    np.savez(out / "point_cloud.npz", points_xyz=pts_all, colors_rgb=cols_all)
    pts, cols = filter_valid(pts_all, cols_all)
    mask_hw = pred["mask"].astype(bool) if "mask" in pred.files else None
    if mask_hw is not None:
        flags_full = project_mask_flags(pts, K, mask_hw)
    else:
        flags_full = np.zeros(len(pts), dtype=np.uint8)
    pts_d, cols_d, flags_d = mask_aware_downsample(
        pts, cols, flags_full, TARGET_POINTS, MASK_TARGET_FRAC, rng
    )
    write_pointcloud_bin(out / "pointcloud_small.bin", pts_d, cols_d, flags_d)

    n_masked = int(flags_d.sum())
    print(
        f"  {entry_id:<50}  query={query!r:<35}  "
        f"pts {len(pts_all):>7} -> {len(pts_d):>5}  masked-kept {n_masked:>4}"
    )

    return {
        "scene": meta.get("input", "").rstrip("/").split("/")[-1],
        "entry_id": entry_id,
        "query": query.strip(),
        "frame": frame,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path,
                    help="AffU eval_real results dir for a single checkpoint.")
    ap.add_argument("--eval-dir", required=True, type=Path,
                    help="AffU eval_real root (contains _prep/<scene>/).")
    ap.add_argument("--out", default="static/results", type=Path,
                    help="Output dir under the website project.")
    ap.add_argument("--scene", action="append", required=True,
                    help="Scene name prefix (e.g. kitchen_1). Can be passed multiple times.")
    args = ap.parse_args()

    out_root = (Path(__file__).resolve().parents[1] / args.out).resolve() \
        if not args.out.is_absolute() else args.out
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    imported = []
    for scene in args.scene:
        folders = list_scene_predictions(args.pred_dir, scene)
        if not folders:
            print(f"[warn] no prediction folders for scene={scene!r} in {args.pred_dir}", file=sys.stderr)
            continue
        print(f"Scene {scene}: importing {len(folders)} queries -> {out_root}")
        for f in folders:
            try:
                imported.append(import_one(f, args.eval_dir, out_root, rng))
            except Exception as e:
                print(f"  [error] {f.name}: {e}", file=sys.stderr)

    print(f"Imported {len(imported)} entries.")


if __name__ == "__main__":
    main()
