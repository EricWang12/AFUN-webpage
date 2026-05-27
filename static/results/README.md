# Cherry-picked AFUN results

Three representative samples exported from the cherry-pick set
(`eval_3d/qual_rows/selected/combined_withgt.json`, kept under `Ours-106800`).

| Folder | Source dataset | Instruction | RGB resolution |
|---|---|---|---|
| `01_agibot_drawer/` | AgiBot (egocentric) | "Open the orange microwave on the desktop." | 640 x 480 |
| `02_droid_real_robot/` | DROID (real robot) | "Push the lever on the toaster downwards." | 1280 x 720 |
| `03_robomind2_open_oven/` | RoboMIND-v2 | "open oven" | 640 x 480 |

## Files in each sample folder

Inputs:
- `rgb.png` — input RGB image (`obs_frame.png` from the dataset).
- `depth.npy` — float32 depth map in meters, shape `(H, W)`. Invalid pixels are 0.
- `intrinsics.json` — `K` (3x3), `fx/fy/cx/cy`, and image height/width.
- `instruction.txt` — language task description.
- `gt_mask.png` — ground-truth affordance mask (when available).

Point cloud (back-projected from RGB-D using `K`):
- `point_cloud.ply` — binary PLY with per-vertex RGB color.
- `point_cloud.npz` — same data as `{points_xyz: float32 (N,3), colors_rgb: uint8 (N,3)}`.

AFUN prediction (model: `Ours-106800`, i.e. `p0.84-robot-sonata-geo-combined_v3` step 106800):
- `trajectory_50pts.txt` — 50 predicted 3D points in the camera frame (meters), one `x y z` per line.
- `trajectory_50pts.npy` — same `(50, 3)` float32 array.
- `prediction.npz` — full raw prediction (trajectory, mask, contact point, spline control points, score).
- `prediction_meta.json` — extracted scalars/lists from `prediction.npz` (contact point, motion type, score, spline control points, r0).

Optional:
- `viz_3d_interactive.html` — standalone Plotly viewer with the point cloud + 50-point trajectory.

`index.json` (top level) — machine-readable summary of all 3 samples.

## Reproducing the point cloud from the inputs

```python
import json, numpy as np
from PIL import Image

rgb = np.array(Image.open('rgb.png').convert('RGB'))
depth = np.load('depth.npy')                       # (H, W) float32, meters
K = np.array(json.load(open('intrinsics.json'))['K'])

H, W = depth.shape
ys, xs = np.mgrid[0:H, 0:W]
z = depth
m = (z > 0) & np.isfinite(z) & (z < 5.0)
fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
X = (xs - cx) / fx * z
Y = (ys - cy) / fy * z
pts = np.stack([X[m], Y[m], z[m]], axis=1)         # (N, 3)
cols = rgb[m]                                       # (N, 3) uint8
```
