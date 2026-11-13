# Unfolded pouch research — 2026-11-14

**WIP. No completed deformable sealing demonstration or public task score yet.**
The older rigid Arena implementation remains in `task.py` / `arena_scene.py`.
These experiments use RoboSimStudio + MuJoCo + Newton VBD; they do not yet
establish public Isaac/Arena compatibility.

## Verified results

| Trial | Observed result | Limit |
|---|---|---|
| `rigid-clip-high-carry-v6` | YAM grasps handle ends, opens spring, lifts, rotates, carries, holds 3 s; target error 0.218 mm / 0.0172 rad | No film |
| `rigid-thin-blade-v7` | Same sequence with the thin lower blade, 434 frames, finite, no controller failure | No film; not sealing |
| `fixed-clip-closure-ke1e6-v1` | Passive upper-jaw closure on four film layers; 3 s / 7,200 substeps, zero scoped intersections | Lower clip body fixed to a fixture |
| `fixed-clip-closure-ke1e4-v2` | Same fixture for 4 s / 9,600 substeps, zero scoped intersections | Fixed base; not free-tool retention |
| `unfolded-supported-press-v7` | Continuous unfolded start; folding proceeds; rejected at frame 494, with 428 substep BOX hits | Narrow neck support side-edge failure |
| `unfolded-full-support-v8` | Continuous unfolded start, folding and commanded press sweeps (zero measured press load); median fold 2.63 rad, 90% folded columns; 90,960 substeps with zero detected crossings/BOX hits | Frictional transfer to the board edge failed; no clip attempt |
| `press-component-force-v2` | Resumed physical checkpoint, 28,080 substeps with zero scoped hits | Nominal press missed the folded patch; fold gate failed |
| `release-component-sliding-v3` | Resumed pre-release checkpoint, 33,720 substeps with zero scoped hits | Fold gate failed on the former narrow support |

Compact results, selected actual states, frozen sources and explicit scopes
are under `../evidence/unfolded-20261114/`. Full raw trajectories remain on
Brev. Selected snapshots do not certify intervening steps.

The former 120 mm neck support left 20 mm of pouch on either side unsupported.
Rejected triangles were at Y=+/-60 mm, around its upper side edges. That narrow
apron was useful for the abandoned side regrasp but is unnecessary for table
insertion. The full-width candidate now supports the whole 160 mm pouch.
Its reset finger/fixture BOX check passed 48 pairs; mesh/capsule contacts are
outside that check.

## Current implementation

- A sewn 200 x 160 mm pouch starts lying down, mouth flat and unfolded.
  Bottom and sides are welded; there is no initial imposed fold, particle
  attachment or object pose write during actions. The authored ready reset
  places open hands around the rim.
- YAM's pads are 12 x 4 x 80 mm. The wrist follows the folding arc at the same
  angle. The former 1.45x wrist lead requested about 5% geometric free-span
  extension at mid-turn. Actual edge extension is separately measurable with
  `inspect_membrane.py` and must not be confused with that planned demand.
- `raised_flat` shifts the pouch 120 mm forward and uses a 240 mm raised
  support. `--mouth-support full --support-extension-mm 14` places the board
  front 14 mm ahead of the intended crease, preserving reset finger clearance.
- `select_press_patch` measures projected overlap of the four actual material
  ply groups. If a strip is upright, it explicitly falls back to its measured
  rim. It keeps the pad footprint on the support.
- Pressing completes XY travel with 50 mm bottom clearance, then descends
  vertically. A sampled force limit and slow admittance stroke replace an
  unconditional downward push. Sampled simulator loads are not hardware force
  measurements or substep-integrated loads.
- The `admittance` support option keeps a pressing hand in contact while the
  other hand works. Contact floor and pad/board friction experiments remain
  under validation. The baseline fixed-height pad slipped during edge transfer.
- The clip has divergent levers and end bosses, a 0.8 mm lower-blade surrogate,
  stepped thin toe, 100 mm jaws, 0.4 mm closed gap and an uncalibrated 0.2
  Nm/rad passive spring. The tool is squeezed by its handle ends, not motorized.
- After folding, the planned controller slides the folded neck to the edge
  through left-pad contact, enters below the overhang, raises the blade clear
  of the board, then advances and releases. The four-ply clip retention stage
  is not yet complete.
- `ToolFeedback` adds the preceding substep's full film reaction wrench to
  clip bodies. This is explicit staggered coupling, not an implicit two-way
  solver. The fixed-base closure fixture does not prove free-body stability.

## Restartable component diagnostics

Each frame now records film position/velocity, rigid pose, joint position/
velocity and actual plastic rest angles. A successful pre-clip gate writes
`clip-entry.npz` plus metadata and control targets. `--clip-component-seed`
restores a physical component reset, rebuilds solver caches, and marks
`initial_fold_imposed=true`. It is never an uninterrupted end-to-end proof.

Saved partial-stage reset seeds also include the exact source frame and control
trace. `--component-start` selects raw-rim release, centre pressing, edge drag
or clipping. Geometry and material profiles must match. An intentional contact
material comparison requires the explicit `--override-contact-materials` flag
and is recorded in the arguments; the final demonstration must rerun from the
unfolded reset with the selected settings.

For a resumed rack contact, the strict rigid preflight is retained alongside
an additional 10 micrometre rack-only surface inset check. The observed settled
contacts were 2.2–3.6 micrometres deep. This does not relax film checks, opposing
clip-part checks, or the strict fresh-scene preflight.

## Material scope

All values are uncalibrated hypotheses. The 80 micrometre mass thickness gives
0.08 kg/m². Numerical contact skins are larger: 0.3 mm particle radius and
0.5 mm self-contact interaction distance. An actual multilayer wrapper still
needs measured thickness, bending, friction and crease recovery; anisotropy,
delamination and film tearing are not represented.

- `derived`: E=100 MPa, rest-angle recovery 0.1/s (older baseline).
- `permanent`: E=100 MPa, zero recovery.
- `laminate`: E=1 GPa, zero recovery; the current comparison.
- `paper_matched`: upstream numerical coefficients, not a measured laminate.

The 1 GPa hypothesis is below one manufacturer's 2.507 GPa MD value for a
[BOPP product](https://www.derflex.com/flexible-BOPP-packaging-films.html).
That 20.7 micrometre cigarette film is not our snack wrapper; it supplies a
scale comparison only. See Toray's [measurement explanation](https://www.films.toray/en/knowledge/kno_002_2.html).

The current research bag is empty. Contents, fracture/spill penalties, air,
public runner integration and Dream inference remain outstanding.
`implicit_pouch_air.py` is an unintegrated candidate: four reference/kernel
bridge tests pass on CPU and CUDA, but no pouch pressure/vent/convergence
validation is claimed yet.

## Commands and acceptance

Pinned RSS: `4e52e1a76437ef1296028afea744437f6e9469c0`. Use the existing
CUDA-12 Warp environment on this L40S/565-driver instance; do not `uv sync`
CUDA-13 packages into it. Check the driver before restoring on other GPUs.

From `/data/robocurve/robosimstudio`, the current full-support baseline is:

```sh
PYTHONPATH=. JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /data/robocurve/envs/rss/bin/python \
  /data/robocurve/tasks/close-snack-bag/experiments/fold_pouch_robot.py \
  /fresh/output --ready-start --drive --clip --stage seal --hz 20 \
  --physics-hz 2400 --workspace raised_flat --material-profile laminate \
  --clip-style thin_lower --table-insert --release-style sliding \
  --mouth-support full --support-extension-mm 14 --support-hold static \
  --press-floor-mm 2.5 --check-substeps --frames 3000
```

This baseline's edge transfer failed; it is not a successful task recipe.
The `edge-drag-contact-v9` component had no film load: a released cursor
was re-seated on the measured hand every tick, cancelling its small descent.
`edge-drag-contact-v10` keeps the continuous force-control cursor moving and
requires measured contact before transfer. Its regression test and 17 related
geometry/material/air-kernel tests passed. Physical retention remains unproved.
The component compares pad friction 1.2, board
friction 0.2, a 0.5 mm approach floor and admittance support against its actual
folded checkpoint. Those coefficients are experimental, not calibrated.

`assess_unfolded.py` requires a genuine full episode, an unfolded start,
controller completion without failure, all substeps checked, no scoped hits,
measured four-ply jaw coverage >=96 mm, retained fold, loaded jaws and both
hands clear continuously for >=3 s. It always leaves `submission_score`
unknown while air/contents and the public scorer are absent.

The current independent observer covers strict transverse self-crossings and
film penetration into BOX, capsule and sphere solids after every solved physics
substep. Capsule checks use exact segment/triangle minimum distances. It excludes
coplanar overlap, other collider types and motion within a substep. Older results
without `checked_round_shapes` only checked BOXes; their scope is unchanged. The renderer
uses actual recorded coordinates without interpolation; printed outer film and
silver inner lining affect appearance only. A core-demo caption requires an
assessment matching the exact recording and trace hashes.

## Press contact diagnosis and current retry

The old horizontal pinch wrist pose was also used for pressing. Its rear
finger capsules extend below the selected pad and hit the 240 mm support
before the pad reaches the pouch. At the v10 checkpoint, one lower-finger
capsule was in board contact while the pad remained about 24 mm above it.
A continuous cursor fix alone therefore did not establish film load.

The upstream forward-pointing paper presser reverses the closing-axis sign
and was unreachable in this workspace (169 mm offline position error). A
156-pose FK/IK comparison found a forward-pointing, 25-degree-down variant
with the original closing-axis sign reaches the desired tip within a
micrometre. This is kinematic evidence only. `press-drag-tip-v12` and
`unfolded-tip-press-v11` now test its actual collision and contact behaviour.
All BOX/capsule/sphere finger geometry is used to place the pressing tip;
the tip's side sweep ends at Y=70 mm, inside the 80 mm pouch half-width.

`fold_pouch_robot.py` now saves physical component seeds at stage boundaries
under `seeds/`, including target controls, velocities and crease rest angles.
These reduce repeated prefix simulation and remain labelled component resets.


## Latest contact-driven iterations

No completed sealing demo yet. The preceding "live v10" paragraph is now
historical. Current scripts use a direct wedge path and a different press order.

| Trial | Outcome |
|---|---|
| `unfolded-tip-press-v11` | Real pitched finger contact; 690 frames / 82,800 substeps, zero scoped hits; edge drag peeled the flap open and failed |
| `unfolded-wedge-v12` | 591 frames / 70,920 substeps, zero scoped hits; centre press while supported did not retain the fold |
| `press-drag-tip-v13` | Actual side press retained ~2.69 rad / 90% fold; edge drag failed; restart seeds preserved |
| `wedge-clip-v14` | Handle grasp and lift succeeded; low simultaneous rotation/translation slipped; 35,040 substeps, zero scoped film hits |
| `rigid-wedge-carry-v8` | Low direct carry slipped without any film |
| `rigid-wedge-carry-v9` | Opening-stop preload rejected: -0.407 rad exceeded the allowed opening check; this preload was reverted |
| `rigid-wedge-via-high-v10` | High rotation at (0.66,0,1.04), then lower target (0.606,0,1.008), held 3 s; 571 frames, finite, no failure. Final target error 5.46 mm / 0.0741 rad; no film |
| `rigid-wedge-friction3-v11` | Higher friction did not meet the pose check |
| `sequential-rim-press-v16` | Released both raw grips, pressed the measured free rim across Y; reached ~3.00 rad / 100% folded columns with zero scoped hits. Interrupted after the valid fold prefix because the arm-clearance policy needed updating |
| `wedge-high-route-v17` | High route blocked with the left pressing arm present. Recorded cross-arm capsule pair 24/91 (bodies 4/13) at ~23 micrometres penetration. This is rigid arm contact, not film penetration |

`clear-arm-wedge-v18` rejected an unreachable fixed park.
`clear-arm-wedge-v19` resumes v16's pre-grasp seed, selects a reachable side
park by offline FK/IK, then attempts high carry and direct insertion.
It is a component reset, never an end-to-end success claim.

The current wedge policy releases both raw rim grips, presses the actual free
flap end vertically, and sweeps the pressing tip from Y=-45 to +70 mm without
pulling the flap toward the board edge. The FK/IK sweep check passes all six
sampled contact/hover poses (worst position residual 1.99 mm). The left arm is
then cleared to avoid the right arm's transport corridor. The clipped hold
must still be verified and the whole sequence rerun from the unfolded reset.

A 95 mm opening is centred about the closed handle ends before gripper
retreat. Opening about the earlier squeeze midpoint can leave one 86 mm-wide
closed end obstructed. This release correction is implemented, not yet a
validated free-tool retention result.

### Force observations and material scope

`--record-force-means` adds a read-only sum over every physics substep.
`--force-readout mean` additionally chooses that signal for pressing control;
recording alone leaves the last-substep controller unchanged. The core
assessment now requires full force windows, loaded jaws and unloaded fingers
through the continuous 3 s hold. The focused suite passes 27 tests, including
CPU/CUDA geometry and impulse-averaging cases.

The actual pouch mass is 5.615 g, zero particles have zero mass, and gravity
is -9.81 m/s2. Static force-balance auditing is unresolved: rest trials report
roughly 23 N board reaction despite a 0.055 N pouch weight and near-zero
momentum change. A two-ring self-contact filter and reducing the self-contact
skin from 0.5 to 0.1 mm did not resolve this. Treat reported forces as solver
observations, not calibrated physical loads. The existing material/spring
claims remain uncalibrated; the source of this large residual needs isolation.
The legacy 0.5 mm self-contact skin and zero-ring filter remain the default;
changes require explicit flags and a contact-settings override for old seeds.

A held-clip component reset was rejected by strict BOX preflight: observed
MuJoCo pad/handle overlaps were 56-73 micrometres. That check was not relaxed.
Retry from the saved pre-grasp seed instead of the held-tool seed.

All raw runs stay on Brev. Code, selected evidence and prior reset seeds stay
on `feature/close-snack-bag`; no Dream run or submission has been made.

The updated side park passed dynamically in v19; the fold retained ~2.92 rad
and 100% folded columns with the left arm clear. `unfolded-clear-wedge-v13`
is the current fresh, uninterrupted reset attempt using that policy.

A full-shape reaction audit also reports ~26 N total reaction, so the earlier
force-balance discrepancy is not just omitted palm contacts. All robot-body
reaction sums are now recorded, and the core hands-free gate requires them to
be unloaded. The physical interpretation/convergence of these solver forces
is still unresolved. No force scaling or arbitrary reaction cap was applied.
