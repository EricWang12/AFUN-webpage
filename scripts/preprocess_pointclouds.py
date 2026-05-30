#!/usr/bin/env python3
"""
Build lightweight binary point clouds for the in-browser 3D gallery.

For every result folder under static/results/ that has a point_cloud.npz,
this writes pointcloud_small.bin with a downsampled cloud:

  uint32  N
  float32 positions[N, 3]     (XYZ in camera frame, meters)
  uint8   colors[N, 3]        (RGB, 0-255)
  uint8   mask_flag[N]        (1 if the projected pixel falls inside
                               the predicted affordance mask, else 0)

The mask_flag block lets the JS viewer recolor masked points in red so
the user can see which 3D region corresponds to AFUN's predicted mask.
The format stays tiny and trivial to read from JavaScript via
fetch + ArrayBuffer. Re-run this script after adding new result folders.

Usage:
  python3 scripts/preprocess_pointclouds.py
"""
import json
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(ROOT, 'static', 'results')

TARGET_POINTS = 10000      # downsample target — ~160 KB per cloud
MASK_TARGET_FRAC = 0.15    # try to fill at least this fraction with masked points
Z_MIN = 0.05               # discard invalid / behind-camera points
Z_MAX = 8.0                # discard far-away noise
SEED = 42


def filter_valid(points, colors):
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] > Z_MIN)
        & (points[:, 2] < Z_MAX)
    )
    return points[valid], colors[valid]


def mask_aware_downsample(points, colors, mask_flags, target, mask_frac, rng):
    """Downsample to `target` points. Oversamples mask points so that up to
    `mask_frac` of the kept points come from the masked region when possible.
    This keeps tiny masks (e.g. a small toaster lever) visible after random
    downsampling, which would otherwise drop them to <10 points."""
    n = len(points)
    if n <= target:
        return points, colors, mask_flags

    masked_idx   = np.where(mask_flags > 0)[0]
    unmasked_idx = np.where(mask_flags == 0)[0]
    n_mask   = len(masked_idx)
    n_unmask = len(unmasked_idx)

    want_masked   = min(int(round(target * mask_frac)), n_mask)
    want_unmasked = min(target - want_masked, n_unmask)
    # If we couldn't fill the unmasked quota (very small clouds), top up with
    # more masked points.
    if want_unmasked + want_masked < target and n_mask > want_masked:
        want_masked = min(target - want_unmasked, n_mask)

    pick_mask   = rng.choice(masked_idx,   want_masked,   replace=False) if want_masked   else np.array([], dtype=int)
    pick_unmask = rng.choice(unmasked_idx, want_unmasked, replace=False) if want_unmasked else np.array([], dtype=int)
    idx = np.concatenate([pick_unmask, pick_mask])
    rng.shuffle(idx)   # mix masked / unmasked in render order
    return points[idx], colors[idx], mask_flags[idx]


def project_mask_flags(points, K, mask_hw):
    """Project each (X,Y,Z) into pixel coords and sample mask_hw."""
    if mask_hw is None:
        return np.zeros(len(points), dtype=np.uint8)
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


def load_predicted_mask(folder):
    """Return predicted affordance mask as a bool HxW array, or None."""
    pred_path = os.path.join(folder, 'prediction.npz')
    if not os.path.exists(pred_path):
        return None
    pred = np.load(pred_path, allow_pickle=True)
    for key in ('pred_mask_hw', 'mask_hw', 'pred_mask'):
        if key in pred.files:
            m = pred[key]
            if m.dtype != bool:
                m = m.astype(bool)
            return m
    return None


def load_intrinsics(folder):
    path = os.path.join(folder, 'intrinsics.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        K = np.array(json.load(f)['K'], dtype=np.float64)
    return K


def write_bin(path, points, colors, mask_flags):
    n = len(points)
    with open(path, 'wb') as f:
        f.write(np.uint32(n).tobytes())
        f.write(points.astype(np.float32, copy=False).tobytes())
        f.write(colors.astype(np.uint8, copy=False).tobytes())
        f.write(mask_flags.astype(np.uint8, copy=False).tobytes())


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f'No results dir at {RESULTS_DIR}', file=sys.stderr)
        sys.exit(1)

    rng = np.random.RandomState(SEED)
    total = 0
    for name in sorted(os.listdir(RESULTS_DIR)):
        folder = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(folder):
            continue
        npz_path = os.path.join(folder, 'point_cloud.npz')
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path)
        points = data['points_xyz']
        colors = data['colors_rgb']
        n_in = len(points)

        # 1) drop invalid depth, NaNs, far noise
        points, colors = filter_valid(points, colors)

        # 2) compute per-point mask flag on the FULL valid cloud, so the
        #    mask-aware sampler can see the whole population.
        K = load_intrinsics(folder)
        mask_hw = load_predicted_mask(folder)
        if K is None or mask_hw is None:
            mask_flags_full = np.zeros(len(points), dtype=np.uint8)
            mask_note_src = 'no mask'
        else:
            mask_flags_full = project_mask_flags(points, K, mask_hw)
            mask_note_src = f'{int(mask_flags_full.sum())} masked-source'

        # 3) downsample with mask awareness
        points, colors, mask_flags = mask_aware_downsample(
            points, colors, mask_flags_full,
            TARGET_POINTS, MASK_TARGET_FRAC, rng,
        )
        n_out = len(points)

        out_path = os.path.join(folder, 'pointcloud_small.bin')
        write_bin(out_path, points, colors, mask_flags)
        size_kb = os.path.getsize(out_path) / 1024
        n_masked_kept = int(mask_flags.sum())
        print(
            f'  {name:<35}  {n_in:>7} -> {n_out:>5} pts  ({size_kb:6.1f} KB)  '
            f'{n_masked_kept:>5} masked-kept  ({mask_note_src})'
        )
        total += 1

    print(f'Done. Wrote {total} pointcloud_small.bin files.')


if __name__ == '__main__':
    main()
