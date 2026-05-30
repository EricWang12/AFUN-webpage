# AFUN prediction results (web gallery)

Data behind the **Prediction Results** section of the site. The gallery reads
`scenes.json`; everything else here is the per-query payload it fetches.

## Layout

```
scenes.json                     # manifest the gallery loads (scenes -> queries)
<scene>/                        # single-query scene: the 3 files live here
    rgb.png
    pointcloud_small.bin
    trajectory_50pts.txt
<scene>/<query>/                # multi-query scene: one subfolder per query
    rgb.png
    pointcloud_small.bin
    trajectory_50pts.txt
```

Current scenes: `kitchen` and `wood_cabinet` are multi-query; `microwave`,
`toaster`, `drawer`, `toaster_oven`, `lid`, `pot` are single-query.

## The three files the site uses

Per query, the in-browser viewer fetches only:

- `rgb.png` — input RGB (shown as the inset thumbnail + scene-card image).
- `pointcloud_small.bin` — downsampled cloud the WebGL viewer renders. Layout:
  `uint32 N`, `float32 positions[N,3]`, `uint8 colors[N,3]`, `uint8 mask_flag[N]`
  (`1` = inside the predicted affordance mask, recolored red). See
  `scripts/preprocess_pointclouds.py`.
- `trajectory_50pts.txt` — 50 predicted 3D points (camera frame, meters), one
  `x y z` per line; rendered yellow→blue.

## scenes.json

```jsonc
[
  {
    "id": "kitchen",                 // also the folder name
    "name": "Kitchen",               // label shown on the scene card
    "thumb": "kitchen/use_the_microwave/rgb.png",
    "queries": [
      { "id": "kitchen/use_the_microwave", "text": "Use the microwave" },
      ...
    ]
  },
  { "id": "microwave", "name": "Microwave", "thumb": "microwave/rgb.png",
    "queries": [ { "id": "microwave", "text": "Open the orange microwave..." } ] }
]
```

`id` is the path (relative to this folder) of the directory holding the three
files — so single-query scenes point at the scene folder, multi-query queries at
a `<scene>/<query>` subfolder.

## Adding results

1. Import a prediction with `scripts/import_eval_real.py` (eval_real format) or
   `scripts/import_generated_dual.py` (generated_dual format). These write a flat
   per-entry folder with the full payload (depth, point_cloud.npz, prediction.npz,
   intrinsics, …).
2. Keep the three web files in the `<scene>/[<query>/]` location above; the rest
   of the raw payload is not served — move it to `archive/results/` (where the
   original full exports already live) to keep this folder light.
3. Add or extend the scene's entry in `scenes.json`.

The raw inputs/predictions for every current scene are preserved under
`archive/results/<original_entry_name>/` (depth, point clouds, `prediction.npz`,
ground-truth masks, standalone Plotly viewers, and the legacy flat `index.json`).
