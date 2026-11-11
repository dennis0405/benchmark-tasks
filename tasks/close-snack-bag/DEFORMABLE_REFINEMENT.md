# Deformable packet refinement — blocked on coupled contact stability

This research prototype is **not the submitted task**. The existing `arena_scene.py`,
contact scorer, and recorded YAM demo still use the rigid packet. None of the
fixtures or solver probes below is a YAM demonstration.

## Geometry and measurements

- The original two-fold reset shape overlapped the still-thick packet shoulder.
  Extending the flat mouth inboard (`--taper-start 0.12 --taper-end 0.14`)
  removes those crossings in the authored folded shape. The initial geometry
  regression checks this using Open3D; it does not prove dynamic safety.
- The mouth was narrowed to a 0.6 mm front/back gap, with an experimental
  0.1 mm effective surface thickness. Both numbers are uncalibrated. The body
  sits on a 25 mm board with a fixed rigid contents fixture; no air pressure is
  simulated. A narrow support and an edge press can be enabled separately.
- `surface_clip_probe.py` logs the direction of both folds across width and a
  contiguous-width heuristic, along with signed vertex-to-jaw distances. The
  -0.5 cosine threshold is a research heuristic, not an approved scorer.
  Neither node proximity nor node-to-node spacing establishes contact or the
  absence of intersecting triangles.
- `check_surface_intersections.py` stores Open3D candidate intersections and
  applies a second, stricter test that rejects boundary-only touches. This is
  a sampled, floating-point mesh check: zero crossings does not prove continuous
  collision freedom or absence of rigid-body penetration.

## Results

Results and selected sampled meshes are under `evidence/refinement/`.
Earlier results were produced by intermediate script versions and should not
be treated as bitwise-reproducible with the latest scripts.

1. **PhysX:** the corrected folded geometry has zero authored intersections,
   but the 160 x 16 surface produced 298 strict candidate-pair crossings after
   the first 1/960 s physics step, and 4,096 after 0.25 s. A flat-packet control
   likewise developed 5,075 at 0.25 s. A more isotropic 80 x 64 grid reduced
   that flat-control count to 775, but did not eliminate it. Self-collision,
   explicit filter distance, CCD, 64 iterations and four collision-pair
   updates were enabled in the fine-step probe. Pair counts are diagnostic,
   not physical penetration depths. The pressure fixture and force-controlled
   clip insertion did not keep the clip on the folded mouth; this is not a
   robot experiment.
2. **Newton VBD alone:** the corrected 80 x 64 packet produced zero detected
   triangle crossings at the sampled 0.004, 0.25 and 0.5 s points, for both
   flat and directly initialized folded states. With no clip the folded state
   reopened. These are single short trials, not proof of long-term stability.
   The corresponding results, node snapshots, intersection reports and source
   hashes are saved in `newton-thin-flat/`, `newton-thin-folded/`, and
   `newton-thin-sources.sha256`; the probe source used for those trials is
   archived as `newton-thin-source.py.txt`.
3. **Newton VBD with clip:** the clip USD loads, but in the standalone VBD
   probe its root and revolute joint did not move after initialization. A
   dynamic clip and YAM therefore require a rigid-solver coupling, not merely
   successful articulation loading.
4. **MJWarp/VBD coupling:** a clip away from the packet moved and closed under
   gravity, but the manually coupled two-way case near the packet immediately
   diverged. At reset, the proposed opened clip at x=0.180 m, z=0.043 m had
   no detected surface intersections with its jaw/handle boxes and no packet
   vertices inside them (`initial-clip-clearance.json`); this does not prove
   collision-free simulation after the first step. A one-way coupling control
   remained finite, but cannot validate clamping because it omits the
   deformable reaction on the clip. With two-way proxy coupling, increasing
   the per-body contact buffer from 4,096 to 65,536 removed the overflow
   warning and reducing the soft margin to 0.3 mm reduced the nominal contact
   envelope, yet the clip still diverged during the first 0.05 s. Another
   80 x 64 coupled run was interrupted after its first step failed to finish
   within two minutes. Coupled results remain invalid, not success evidence.

The installed Arena deformable asset rejects contact sensors against
deformables. Thus the old rigid-contact-force scorer cannot be reused without
an independently validated deformable contact/geometry strategy. See also
NVIDIA's [PhysX deformable collision settings](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab_physx/isaaclab_physx.sim.schemas.html).

## Reproduce representative probes

Each command requires a **new output folder**. The 80 x 64 Newton tests need
the Isaac Lab-Arena environment, GPU, and Open3D; runtime and dynamics are
not guaranteed to match another software version.

```bash
source /data/IsaacLab-Arena/yam.env
cd /data/benchmark-tasks/tasks/close-snack-bag
python -m unittest -v test_surface_mesh test_surface_intersections test_scoring
python initial_clip_clearance.py
python newton_surface_probe.py /data/snack-runs/new-flat
python newton_surface_probe.py /data/snack-runs/new-folded --folded
python check_surface_intersections.py /data/snack-runs/new-folded/surface-snapshots.npz
python newton_surface_probe.py /data/snack-runs/new-coupled --folded --clip --coupled \
  --clip-x 0.18 --clip-z 0.043 --nx 40 --ny 32 --duration 0.2 \
  --soft-ke 8000 --soft-kd 0.01 --soft-contact-margin 0.0003 --shape-gap 0.0005
```

Next, stabilize two-way rigid/deformable contact with validated forces and no
capacity overflows; test all folded layers and triangle/rigid intersections
through 3 s after release; integrate a supported Newton coupling into YAM;
then develop a nodal scorer and record a fresh YAM demo. Real packet material
calibration is still outstanding.
