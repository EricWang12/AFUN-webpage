#!/usr/bin/env python3
"""
Import AFUN "generated_dual" predictions into static/results/ for the web gallery.

A generated_dual scene folder looks like:
  <scene>/
      <scene>.png              -- scene RGB (shared by every query)
      <scene>_depth.npy        -- depth map, shape (H, W); may be millimeters
      <scene>_cam.json         -- {fx, fy, cx, cy}
      <Query phrase>.npz       -- one per query: mask, points_N50, query, ...
      <Query phrase>.png       -- seg viz (not used by the website)
      <Query phrase>_3d.html   -- standalone viewer (not used by the website)

This produces, for each query, static/results/<entry_id>/ with the files the
website JS reads (rgb.png, depth.npy, intrinsics.json, instruction.txt,
prediction.npz, trajectory_50pts.{txt,npy}, point_cloud.npz, pointcloud_small.bin)
— the same layout as scripts/import_eval_real.py. The heavy lifting (backproject,
mask projection, mask-aware downsample, binary writer) is reused from that module.

Run from the project root:
  python3 scripts/import_generated_dual.py \
      --scene-dir static/generated_dual/wood_cabinet \
      --scene-id wood_cabinet

Then add a scene entry to static/results/scenes.json (this script prints a
ready-to-paste JSON block at the end).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse the shared pipeline so the binary format stays identical across importers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_eval_real import (  # noqa: E402
    SEED,
    TARGET_POINTS,
    MASK_TARGET_FRAC,
    backproject,
    filter_valid,
    project_mask_flags,
    mask_aware_downsample,
    write_pointcloud_bin,
)

# Files in a scene folder that are NOT per-query predictions.
NON_QUERY_NPZ = set()


def load_cam(scene_dir: Path, scene_id: str) -> np.ndarray:
    cam_path = scene_dir / f"{scene_id}_cam.json"
    cam = json.loads(cam_path.read_text())
    fx, fy = float(cam["fx"]), float(cam["fy"])
    cx, cy = float(cam["cx"]), float(cam["cy"])
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def find_query_npzs(scene_dir: Path, scene_id: str) -> list[Path]:
    """Per-query npz files: every .npz that carries a points_N50 trajectory."""
    out = []
    for p in sorted(scene_dir.glob("*.npz")):
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if "points_N50" in d.files and "mask" in d.files:
            out.append(p)
    return out


def import_one(npz_path: Path, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray,
               scene_id: str, out_root: Path, rng):
    pred = np.load(npz_path, allow_pickle=True)
    query = str(pred["query"]).strip().rstrip(".")
    stem = npz_path.stem  # e.g. "Open_the_left_cabinet_door"
    entry_id = f"{scene_id}_{stem}"

    out = out_root / entry_id
    out.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(out / "rgb.png")
    np.save(out / "depth.npy", depth)
    H, W = depth.shape
    (out / "intrinsics.json").write_text(json.dumps({
        "K": K.tolist(),
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "height": int(H), "width": int(W),
    }, indent=2))
    (out / "instruction.txt").write_text(query + "\n")

    np.savez(out / "prediction.npz", **{k: pred[k] for k in pred.files})
    traj = pred["points_N50"].astype(np.float32)
    np.save(out / "trajectory_50pts.npy", traj)
    with open(out / "trajectory_50pts.txt", "w") as f:
        for x, y, z in traj:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    pts_all, cols_all = backproject(rgb, depth, K)
    np.savez(out / "point_cloud.npz", points_xyz=pts_all, colors_rgb=cols_all)
    pts, cols = filter_valid(pts_all, cols_all)
    mask_hw = pred["mask"].astype(bool) if "mask" in pred.files else None
    flags_full = project_mask_flags(pts, K, mask_hw) if mask_hw is not None \
        else np.zeros(len(pts), dtype=np.uint8)
    pts_d, cols_d, flags_d = mask_aware_downsample(
        pts, cols, flags_full, TARGET_POINTS, MASK_TARGET_FRAC, rng
    )
    write_pointcloud_bin(out / "pointcloud_small.bin", pts_d, cols_d, flags_d)

    print(f"  {entry_id:<48}  query={query!r:<34}  "
          f"pts {len(pts_all):>7} -> {len(pts_d):>5}  masked-kept {int(flags_d.sum()):>4}")
    return {"id": entry_id, "text": query}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True, type=Path,
                    help="generated_dual scene folder (contains <id>.png, <id>_depth.npy, <id>_cam.json).")
    ap.add_argument("--scene-id", required=True,
                    help="Scene id / filename stem, e.g. wood_cabinet.")
    ap.add_argument("--out", default="static/results", type=Path)
    args = ap.parse_args()

    scene_dir = args.scene_dir.resolve()
    out_root = (Path(__file__).resolve().parents[1] / args.out).resolve() \
        if not args.out.is_absolute() else args.out
    out_root.mkdir(parents=True, exist_ok=True)

    rgb = np.array(Image.open(scene_dir / f"{args.scene_id}.png").convert("RGB"))
    depth = np.load(scene_dir / f"{args.scene_id}_depth.npy").astype(np.float32)
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size and np.median(finite) > 50.0:   # mm -> m (see import_eval_real)
        depth = depth / 1000.0
    K = load_cam(scene_dir, args.scene_id)

    if rgb.shape[:2] != depth.shape:
        print(f"[warn] rgb {rgb.shape[:2]} != depth {depth.shape}; "
              f"resizing RGB to depth resolution", file=sys.stderr)
        rgb = np.array(Image.fromarray(rgb).resize((depth.shape[1], depth.shape[0])))

    rng = np.random.RandomState(SEED)
    npzs = find_query_npzs(scene_dir, args.scene_id)
    if not npzs:
        print(f"[error] no query .npz found in {scene_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Scene {args.scene_id}: importing {len(npzs)} queries -> {out_root}")
    queries = [import_one(p, rgb, depth, K, args.scene_id, out_root, rng) for p in npzs]

    # Print a scenes.json block to paste in.
    block = {
        "id": args.scene_id,
        "name": args.scene_id.replace("_", " ").title(),
        "subtitle": f"One scene, {len(queries)} language queries"
                    if len(queries) > 1 else "One scene",
        "thumb": f"{queries[0]['id']}/rgb.png",
        "queries": queries,
    }
    print("\nAdd this entry to static/results/scenes.json:\n")
    print(json.dumps(block, indent=2))


if __name__ == "__main__":
    main()
