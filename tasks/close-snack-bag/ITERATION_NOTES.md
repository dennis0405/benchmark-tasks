# Newton snack task — current engineering iteration

This document supersedes older experiment conclusions in
`DEFORMABLE_REFINEMENT.md` and `POUCH_REDESIGN.md` where they differ. The
experimental systems below are not yet wired into the Arena task, have not
been calibrated to a real bag or chip, and do not establish a YAM demo or score.
Several retained result files were generated with intermediate script versions;
use the source and flags in this file for new experiments, not exact replay.

## Air diagnostics

The current `pouch_air_warp.py` subtracts the dynamic volume of contents
 tetrahedra from the gas volume and rejects inverted content elements. Optional
`--air-channel` uses the narrowest paired-skin cross section as a leak proxy;
it is not CFD and has not been validated against actual folding or a clip.

A manually sealed (not physically clipped) control with 16 substeps, rather
than 4, reached the ground within 0.1 s and remained there through 1 s. Four
sampled film meshes had no detected self-crossings and the sampled contents
nodes stayed inside. The mouth visually remained open despite the manually
imposed zero leak; this is a numerical control, not a demonstration of a seal.
It does not establish that all pressure/solver stability issues are resolved.

## Pre-fragmented chips

`breakable_chip_builder.py` creates four chips as eight tetrahedral halves,
joined by six springs per chip. `ChipFractureMonitor` checks extension every
solver substep on the GPU and permanently releases all seam springs when its
threshold is exceeded. This is a single predefined fracture plane, not a
validated physical potato-chip material; the 0.6 mm threshold is a provisional
numerical control value. Chips are not yet connected to the submitted scene.

In 0.4 s air-open controls with 16 substeps and a 0.6 mm threshold, no chips
broke without manipulation, while a 0.3 N balanced diagnostic bend broke only
chip 0; the extra force stopped once it broke. Before changing the threshold,
a 0.15 mm limit broke chip 0 during settling in both air and no-air controls.
A standalone tensile and bend test also distinguished low and high loads. These
are simulation checks, not experimentally measured failure forces. The direct
load is a diagnostic fixture, not a gripper or physically justified crushing.

## Clip contact and YAM

`newton_surface_probe.py` now calls `sim.forward()` after writing the clip
articulation state and reads VBD body poses from Newton's live state. The old
articulation-view readings were stale: the claim that the clip did not move
under VBD was incorrect. A free-clip control with refreshed initial FK fell
and closed; an otherwise similar control without the refresh diverged.

`pouch_clip_probe.py` supports a bounded external-force fixture and a press
plate. This is a diagnostic and not a YAM policy. A 1.2 s press-assisted trial
with particle-only rigid-soft contact preserved most of the folds, but only the
lower jaw contacted and sampled packet triangles crossed that jaw (12 strict
pairs at 0.5 s). Therefore it failed contact validation. Subsequent full
surface contact and other settings require independent measurement.
`check_clip_intersections.py` checks sampled film triangles against actual
recorded jaw poses, including triangle/box crossing where no vertex is inside.
A zero result at samples does not prove continuous collision freedom.

`yam_newton_smoke.py` loaded both YAM arms into Newton VBD; with a 0.15 rad
joint1 target, the grippers moved about 16 mm in world Y relative to an
otherwise identical zero-target run at 0.4 s. This shows one command affected
the physical links, not that the full YAM controller, contacts, or camera
pipeline can execute the task. `yam_newton_ik_smoke.py` is a further position
tracking probe, not yet evidence of a task demo.

`deformable_scoring.py` is a unit-tested, disconnected state machine. It
requires valid observations for fold, both jaw contacts, coverage, hands clear,
containment and fracture before applying the provisional score. It must not be
used as the competition scorer until those measurements are validated in the
integrated simulation and 3 s of actual post-release behavior are observed.

## Controls

All paths passed as outputs must be new. Activate the Arena environment and
run from `tasks/close-snack-bag`:

```sh
python -m unittest -q test_pouch_air_warp test_crease_model test_breakable_chip_builder \
  test_breakable_chip_probe test_deformable_scoring test_clip_intersections test_clip_force_fixture
python breakable_chip_probe.py /data/snack-runs/chip-bend-low.json --force .0001 --direction z --duration .05
python breakable_chip_probe.py /data/snack-runs/chip-bend-high.json --force .03 --direction z --duration .05
python pouch_breakable_probe.py /data/snack-runs/bag-no-load --duration .4 --air \
  --substeps 16 --iterations 30 --self-radius .0002 --self-margin .0004 \
  --collision-interval 1 --break-extension .0006
python pouch_breakable_probe.py /data/snack-runs/bag-load --duration .4 --air \
  --substeps 16 --iterations 30 --self-radius .0002 --self-margin .0004 \
  --collision-interval 1 --break-extension .0006 --damage-force .3
python yam_newton_smoke.py /data/snack-runs/yam-control --delta 0
python yam_newton_smoke.py /data/snack-runs/yam-command --delta .15
```

Selected numerical outputs and sampled meshes are in `evidence/iteration-nov12`.

## Additional controls

The full rigid-soft surface-contact trial (`pouch-clip-fullcontact-v1`) saved
8 samples over 1.2 s with no strict film self-crossings or film/clip box
crossings detected. The clip contacted only the lower jaw and did not achieve
the required two-sided grip; the press plate was also present. This is not a
successful release, seal, or 3 s hold. Enabling full-surface contact increased
the collision contact capacity to roughly 380k. Static/particle-only controls
are not substitutes for full surface tests.

The YAM IK test explicitly resets both arms' full joint state and forwards
kinematics. It then applies Newton inverse-dynamics gravity compensation at
each physics step. In one 20 mm Cartesian movement test, initial FK matched
within 0.1 micrometre, the planning error was 0.014 mm and the final physical
TCP errors were approximately 3.08 and 3.12 mm. Without gravity compensation
the final tracking error was approximately 72 mm. No object contact or task
action was tested. Reproduce with:

```sh
python yam_newton_ik_smoke.py /data/snack-runs/yam-ik-new
```
