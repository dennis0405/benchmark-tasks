# Clip / film closing diagnostic

This is a standalone Newton 1.7 diagnostic. It starts with a pre-folded pouch
and an open, free clip already around the fold. The film has no attachment,
pinning or successful-pose correction. The table supports the pouch. It is
not an insertion, lifting, YAM, Isaac Sim or airtightness demonstration.

## Corrected root cause

The older native probe wrote `-.55` into the builder's joint default while
leaving the authored clip body poses closed. VBD's cached rest-angle reference
then became inconsistent. The corrected probe preserves closed model defaults
and opens only the separate episode State through FK. On CPU, Warp's NumPy
array can alias model memory, so the initial coordinate array uses `.copy()`.

The one-second bare-hinge A/B measured:

| Model defaults | Final hinge angle | Limits respected |
|---|---:|---|
| Consistent closed rest; separately opened episode | +0.00825 rad | yes |
| Older inconsistent defaults | +0.50562 rad | no |

This corrects the earlier attribution to VBD's rigid-joint path itself.
`hinge_rest_repro.py --inconsistent-rest` preserves the failing control.

## Declared validation gate

`verify_native_hold.py` requires all of the following, separately from the
competition scorer:

- Every physics state, including initial closure, checked for triangle/box
  interior intrusion at 0.1 micrometre tolerance and strict transverse film
  self-crossing; no detected event over the entire episode.
- A continuous three-second window (181 samples at 60 Hz) with active fold
  contacts on both jaw boxes, each spanning at least 80 mm laterally.
- Fold centroid motion relative to the clip at most 2 mm, root drift at most
  2 mm, hinge within [-0.56, 0.03] rad and variation at most 0.02 rad.
- Finite film coordinates throughout, and full physics-check cadence recorded.

These are diagnostic thresholds, not material calibration. Coplanar self-overlap
and motion between physics steps are not certified. No air, food, breakage,
external loading or lift capacity is validated. Contact-force telemetry is
uncalibrated and is used only as a contact witness.

## Run

Use the isolated environment and pinned upstream checkout described in
`FILM_MATERIAL_STATUS.md`. Do not silently re-sync the CUDA-12 Warp wheel back
to the CUDA-13 PyPI build on this host's driver.

```sh
cd /data/robocurve/robosimstudio
export PYTHONPATH=. JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/data/robocurve/envs/rss/bin/python \
  /data/robocurve/tasks/close-snack-bag/experiments/native_film_pouch.py \
  /fresh/output --device cuda:0 --nx 80 --ny 64 --seconds 5 \
  --folded --clip --verify-every-step --record-every 1 --substeps 20
/data/robocurve/envs/rss/bin/python \
  /data/robocurve/tasks/close-snack-bag/experiments/verify_native_hold.py /fresh/output
```

Optional independent rendering needs `pyrender==0.1.45` and `opencv-python-headless`
in addition to existing trimesh/imageio-ffmpeg. `render_native_clip.py` uses EGL
and recorded mesh/body coordinates. It does not interpolate or replace the
simulated motion; its replay runs at half speed. The orange film material is
for visibility, not a calibrated packaging appearance.

```sh
/data/robocurve/envs/rss/bin/python \
  /data/robocurve/tasks/close-snack-bag/experiments/render_native_clip.py /fresh/output
```

Full mesh recordings stay in `/data/robocurve/runs/` on Brev. Compact reports,
selected states and final video are preserved in Git. Keep the Brev instance
running, as requested by the user.
