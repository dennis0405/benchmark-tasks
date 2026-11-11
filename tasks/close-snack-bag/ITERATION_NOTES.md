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

## Clip and chip follow-up

Two fixture-driven controls used full-surface contact and four breakable
chips, without the air model. Neither is a robot demonstration. In v2, both
jaws had contacts at 0.8 and 1.0 s, and all four chips remained intact; after
the clip fixture was released at 1.2 s, the clip slipped away and the two-fold
heuristic fell to zero by 1.6 s. The eight saved samples had no strict
film/film or film/clip crossings. This is a sampled check only.

V3 used a wider, faster pressing plate. Two chips broke by the first 0.25 s,
and the two-fold heuristic was only 0.22 at 1.5 s. It is a failed control,
not evidence of a safe manipulation. Provisional fracture parameters are not
measured physical properties. The compact logs for both controls are under
`evidence/iteration-nov12/clip-followup/`.

V4 initialized the already-folded mouth with the experimental crease rest
angles (`--initial-crease-yield .25`). At 1.6 s the fold-reversal heuristic
was 0.93 and no chips were broken, but the clip had closed on top of the bag:
there was no observed upper-jaw contact and no demonstrated clamping. The
33 sampled configurations had no strict film/clip box crossings. This does
not establish continuous collision freedom. The initial crease is imposed,
not an observed robot folding action. V5 explores
a lower-first clip trajectory with an additional side support that is moved
away after release; it must be evaluated independently for collisions,
contact, integrity and retention.

In V5, the open clip stalled at x≈0.209 m, outside the folded mouth, and only
the lower jaw was observed contacting the packet. The support also settled on
the floor and moved only 12 mm laterally during the release interval. Although
no chips broke, this was not a successful insertion or a clean release. No
strict film/clip crossings were detected in its 20 sampled configurations.

`yam_snack_plan.py` and `yam_soft_proxies.py` are unfinished robot experiments.
The convex per-link deformable-contact proxy ran briefly with stationary arms;
neither accurate finger/film contact nor the task plan has been validated.
The `--sample-every` option records intermediate probe states for later analysis;
it does not itself make the disconnected scorer valid.

In the v8 support-only control, the 40 g fixture stayed at its target height
for 0.3 s away from the bag. V9's corner support nevertheless stayed near the
floor when interacting with the folded packet, and the clip again stalled at
the packet edge. V10 began with the wide support overlapping the packet and
broke two chips; this run is invalid. `support_clearance.py` conservatively
checks the folded surface against the reset support box and rejects that
configuration before simulation. A collision-free v11 corner pinch lost its
grip while the upper fixture rose; the lower support remained near the floor.
All these are fixture diagnostics with experimental material parameters.

In v13, the corner support began clear of the packet at z=14 mm, but by
0.2 s had been pushed to z≈5 mm and its measured fixture force approached
the 3 N cap. The clip stalled and only the lower jaw contacted; no chips broke.
Reducing the soft-contact stiffness from 8000 to 2000 in the 0.6 s v14 control
still produced the same support loss. These are not validated force estimates
for a real bag; sampled fixture commands do not measure all contact forces.
Saved fixture poses now allow sampled triangle/box crossing checks against
the press and support in addition to the articulated clip.

V16 used a 5 N support limit and soft contact stiffness 2000. The sampled
fixture checker detected 17 strict film/support crossings at 0.1 s and 19 at
0.5 s. No vertices happened to be inside the box, illustrating why a
vertex-only check is insufficient. This trial is invalid due to penetration;
it cannot establish a feasible grip. The checker is sampled, not continuous.
With stiffness 8000 and a stronger position gain in v18, the support remained
near z=10 mm and all chips survived, but the sampled checker still found 3–9
strict film/support crossings after 0.1 s. V18 is also invalid.
