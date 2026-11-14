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

### Low transport diagnosis and pressure-pad comparisons

The v19 component failed during low descent, before film insertion. Native
MuJoCo contact reconstruction (`audit_rigid_contacts.py`) found the lower toe
on the board edge. The outside descent was moved to at least 100 mm before
the board front. With this clearance and the left arm parked, rigid-only
pre-grasp resets v16/v17 still missed the flat low pose by 26/37 mm. Their
contacts were the intended handle/finger pairs; the wrist saturated a joint
limit. Planning from the **current** grasp, rather than the earlier lift
transform, exposes this reachability change as the clip tilts in the pads.

An 8-degree yaw/pitch comparison and damped pose corrections reduced the
rigid-only low-pose error to about 2 mm. Tight settling and preserving the
cursor through corrections still failed the unchanged 1 mm precision gate:
`rigid-tight-angled-v22` ended at 2.4 mm/0.031 rad; the stronger passive spring
comparison v23 ended at 1.7 mm/0.011 rad. Neither is a sealing result.
Unreachable hand motions now stop the tool route instead of being silently
treated as completed moves. Rigid traces include actual and commanded hand
poses for this diagnosis.

The `thin_strong` asset changes only passive hinge stiffness from 0.2 to
0.6 N m/rad. Its analytical handle-end load remains below 5 N over the
authored range; this is not a hardware measurement. Experimental `thin_broad`
and `thin_aligned` assets additionally widen the pressure pads from 16 to
40 mm along X; the latter aligns the upper pressure face with the lower one
at the nominal -0.32 rad opening. These comparisons retain free tool dynamics,
the same masses, contact checks and gripper effort limit. They have not yet
established retention. A wider-pad trial resumed at the old tool parking pose
was rejected for rail interference; `--reset-tool` explicitly uses the new
asset's authored parking pose at a rigid-only reset, never during an action.

Coordinate-only `--rebase-z` and the isolated no-self-contact rest audit did
not solve force balance: total rest reactions remained about 12 N. The
default coordinate frame and enabled self-contact are unchanged. The pinned
RSS `shape_contact_forces` documentation itself warns that a resting sheet's
weight is not read correctly. Loads remain numerical observations, not a
physical calibration. The focused CPU/CUDA suite passed 28 tests before the
two additional pressure-pad geometry cases.

The later `rigid-integral-aligned-v28` passed the unchanged 1 mm / 0.02 rad
low-carry gate at a flat root target (0.576,0,0.9694): measured error was
0.459 mm / 0.00190 rad, followed by a 3 s watched hold. It used a recorded
pre-grasp robot reset, no film, and the selected tool's authored parking
pose. Compact evidence is in `evidence/unfolded-20261114/rigid-aligned-v28`.
The controller integrates object-pose errors into the previous **command**;
rebuilding small corrections from the measured hand kept reintroducing a
roughly 0.8 mm loaded-hand tracking bias. Regression tests cover that bias
and a failed-hand-motion refusal. The focused suite now passes 32 tests.

`film-aligned-insert-v20` tests the same policy on the recorded folded film,
with an explicit clip-asset/parking reset. `unfolded-aligned-v14` starts from
the unfolded reset with the aligned-pad asset. They are running candidates,
not completed sealing evidence. The final film/clip contact and hands-free
assessment remain required before calling either a completed demo.

### Continuous blade-height servo

`film-aligned-insert-v20` was stopped at 1,244 frames: the blade stalled at
the board edge before touching film. Native rigid contact reconstruction at
frame 1,080 found the toe against the board (~7.5 micrometres compliant
penetration). The film observer recorded 149,280 substeps with zero scoped
hits. The fresh `unfolded-aligned-v14` prefix was also stopped at 1,238 frames,
with a retained ~2.72 rad fold and 148,560 substeps without scoped hits.
Neither interrupted prefix is a completed demo.

The new insertion servo measures the free tool every control frame, rotates
the TCP command about its measured root, and integrates object-pose error
into the previous command. It pauses forward reference motion when the
actual lowest blade corner lacks clearance and bounds forward lead to
0.8 mm. It never moves cloth or tool states directly and leaves the
independent film intersection checks unchanged.

`rigid-height-servo-v29` crossed the board edge to X=0.66 and held 3 s without
a controller error (1,169 frames). Its target clearance was 0.4 mm. A
read-only control-frame audit gives a minimum conservative lower-blade
clearance of 0.362 mm for the 253 insertion/hold frames overlapping the board.
This is a sampled convex-corner bound, not a substep swept-volume proof.
Compact evidence is in `evidence/unfolded-20261114/rigid-height-v29`.

That height is too high for reliable entry below the bottom film: measured
film vertices near entry start about 0.300 mm above the board, whereas the
0.2 mm toe's upper face would be 0.600 mm above it. A 0.040 mm clearance
comparison puts that upper face at 0.240 mm. The smaller forward-motion
height guard is a trajectory setting, not a relaxation of film intersection
or final four-ply capture requirements.

`rigid-low-clearance-v30` instead failed the unchanged approach gate at
6.4 mm. Native reconstruction found pressure-pad/wrist and open-finger
contacts. `thin_compact` retains the 0.6 N m/rad spring and upper pad alignment
but restores the 16 mm pressure-pad length, avoiding the large 40 mm pad
envelope. Its full joint-range opposing-box geometry case passes.
`rigid-compact-low-v31` then completed the 40 micrometre target-clearance
insertion and 3 s watched hold without failure (1,152 frames). The sampled
lower-bound clearance over the board was positive, with a 6 micrometre minimum
over 268 insertion/hold frames; it is not a substep proof. The film v21
attempt was stopped before contact to make these geometry changes. The
compact-tool film continuation is `film-compact-low-v22`, using the measured
fold from v14 with an explicit asset/parking reset. No new film retention or
submission result is claimed yet.

The separate closure stress fixture now supports a free root on a passive
heel stand. An instant spring release with split MuJoCo/VBD feedback became
non-finite in its first control frame, including a 10 kN/m film-contact
comparison. A no-feedback negative control remained finite but forced jaws
through the layers; it is not an acceptable retention solution. The recorder
now saves invalid final states instead of trying to normalize a zero
quaternion. Intersection counts after non-finite states are not physical
penetration measurements. A fully coupled VBD fixture comparison is being
tested independently; none of these stress fixtures is a YAM task demo.

### Unified solver port (ongoing, no completed film demo)

`film-compact-low-v22` reached first film contact with the compact blade but
stalled before capture. A declared component reset opened the two right
fingers by 0.3 mm each and lifted the free tool 0.02 mm; particle positions,
velocities and plastic rest angles were preserved. v24 remained geometrically
clear but stalled. v25 permitted 5 micrometres of native board contact in the
motion controller; the unchanged film observer then rejected actual tool/film
intersections at frame 79. This is a failure, not a retained seal.

The alternative unified Newton VBD path now uses intrinsic rigid/film contact,
read-only contact observables, body-pose-derived joint encoders, and an explicit
passive spring/damper with equal/opposite torques. The imported MuJoCo-only
spring is disabled in this path to avoid double actuation. The standalone
passive heel-stand closure v9 was finite and had zero observed intersections
for one second with expanded rigid contact capacity. It is neither an
unsupported hold nor a YAM task demonstration.

The first unified robot comparison v32 sagged because VBD does not consume the
imported MuJoCo per-body gravity-compensation attribute. The adapter now applies
the same authored coefficients as COM forces, retaining gravity on the free
tool and film. Native YAM actuator gains and mimic constraints are preserved.
Rigid and film tests of this port are ongoing. New full recordings also save
body velocities, since a body-primary solver cannot be resumed faithfully by
projecting compliant body poses through forward kinematics.

Focused geometry/material/controller suite: 34 tests passed on the Brev CPU/GPU.
These checks do not establish successful film capture or physical calibration.

The unified supported closure `unified-readout-v12` completed 65 control frames
(3.25 s, 3,900 observed substeps), finite with no observed film intersections.
Its heel stand remains present. Readout forces are uncalibrated and strongly
biased, so the observed geometry is not proof of realistic contact loads.
Selected evidence is `unified-supported-closure-v12`.

Robot port comparisons found that imported YAM finger constraints are legacy
sparse mimic rows, which VBD ignores. Translating them to supported joint-owned
metadata caused large arm disturbances and a film intersection during the first
pinch (v3); it is not the current execution path. The current experimental
gripper instead applies bounded opposed actuator forces plus a passive
centering spring/damper through `Control.joint_f`. This is an explicitly
compliant approximation to the transmission, not exact native MuJoCo mechanics.
The active force limit is 5 N per slide; passive spring reactions are separate.
Its CPU/CUDA checks verify force limits and dissipative transmission damping.
The v37 rigid comparison physically grasped and lifted the free clip; carry
and insertion verification are ongoing.

A rebased world and 1 mm top skin split preserve the board's outer solid and
leave all colliders active. This did not by itself fix the resting force
readout bias (v5). A lower contact stiffness comparison is ongoing. CUDA graph
capture is active (v5: one eager frame, nine replayed frames); this is not a
Python eager-loop bottleneck. Full unified film work is several seconds per
20 Hz control frame and is being profiled before tuning iteration counts.

### Contact ownership, profiling, and native robot comparison

The unified v6 run stopped at frame 70 on 77 observed finger/film intersections.
The native VBD rigid-soft DAT option is off by default and requires solver-owned
collision checkpoints. `OwnedVBD` provides that ownership and removes the
redundant RSS main detector. DAT v10 completed 173 frames / 10,380 observed
substeps without intersections, but failed the fold gate (0.79 rad) after
losing the rim grip. The declared contact continuation v11 also stayed
geometrically clear but lost the tool grasp before completing insertion.
These are failures; no full folding/sealing demonstration is claimed.

A Nsight kernel profile of the unified rest case measured 46.6% in body/particle
contact reduction and 19.2% in body/body reduction. Newton uses four strided
threads per body. The opt-in `vbd_parallel.py` specialization sets both the
kernel and launch width to 32, with exact source hash guards and no installed
file edits. A known 100-contact reduction returns the expected -100 N (error
below 0.001 N). Iteration counts and geometry checks remain unchanged. Rest
trajectories differ because parallel atomic accumulation is not bitwise
reproducible; this setting needs fresh trajectory validation. Matched final
throughput comparisons remain pending; no general speedup is certified.

The body-primary port changes YAM response: native MuJoCo armature, force
limits and mimic behavior are not all supported by VBD. A second research
configuration, `hybrid`, keeps the native MuJoCo robot and puts the free clip
in the film's VBD island. The stock staggered two-way proxy coupler carries
reactions; there is no extra delayed `ToolFeedback` force. The passive clip
spring is applied once to its Newton-owned bodies. Contact readout integration
and retained-grip tests are ongoing. v42 reached the handle and opened the
clip but slipped during lift; a force-limited squeeze-hold comparison follows.

The 8,000-contact known-sum microbenchmark measured 2.461 ms at four threads
versus 0.349 ms at 32 threads (five batches of 200 captured replays), with the
same accumulated force. This 7.05x ratio applies only to that contact-reduction
microbenchmark, not an entire scene. Compact numbers are in
`evidence/unfolded-20261114/performance/contact-reduction-benchmark.json`.

A read-only per-substep contact-capacity monitor now persists overflow events,
even if later frames become quiet. The focused suite passed 39 tests; the
new button asset additionally passes its full-range geometry and analytical
squeeze-force checks. These do not imply a completed task episode.

Hybrid v18 with 10x native arm position gains reached the pre-press fold gate
(2.31 rad, 64% columns), unlike the weaker servo comparisons. It later failed
on observed finger/film intersections during pressing at frame 516. A clean
frame-515 checkpoint was extracted for declared repeat-press diagnostics.
The model retains original native actuator effort limits; position stiffness
and damping are explicitly changed, not calibrated against hardware.

A host IK route search found the previous end-grasp parking location outside
the precise reachable range; moving the clip and its passive rack 40 mm in X
makes the tested grasp/lift/flat route kinematically feasible. This is IK-only
evidence, not collision or dynamics certification. Native reconstructed
contacts during squeezing also found contact on slanted finger-wing colliders.
`thin_buttons` retains the jaw and spring but adds narrow, protruding end pads
to seat on YAM's 12 mm main pads. End-pad centres are read from actual collider
transforms by the controller instead of hardcoded coordinates. Physical grip
and carry tests remain ongoing.

The hybrid contact-material assignment was corrected: only clip shapes use
10 kN/m. Finger and board shapes keep the original 1 MN/m values. A fresh
hybrid v20 nevertheless stopped on a board intersection at frame 2, so it is
not a validated fix. The fully coupled, DAT-enabled path is also being
retested with stronger arm tracking, 2,400 Hz physics and the reachable button
grip. No current end-to-end fold/clip/release/3 s success exists.

### Grip patch and thin-contact comparisons

The squeeze target now anchors the free root at the start of closing while
following the relative lever midpoint. This removes the previous root-chasing
feedback. v50/v51/v52 open the hinge but still fail the 50 mm lift check.
v52 records actual rigid contact forces: approximately 2 N per handle, initially
on slanted wing meshes rather than the main flat pad. The narrow 2 mm buttons
also provide little moment resistance while carrying the forward-heavy tool.

The revised button contact patch is 16 x 8 mm: narrower than the 12 mm main
finger width, with a longer torque-bearing patch. Spring stiffness is now
0.95 Nm/rad; the full-range analytical end-force check remains below 5 N.
Eight geometry/force-bound tests pass; these are not a physical calibration.
v54 tests this asset with exactly the original 50 mm lift and carry guards.

Rigid contact-history matching alone did not fix slipping. A separate legacy
soft-contact diagnostic v53 also fails lift; it is not selected as a solution.
The DAT/log-barrier/interval full film run v22 stops at frame 119 with 53
observed right-finger/film intersections and no contact-buffer overflow.
The next explicit comparison stiffens only finger contacts, leaving the light
tool's contact stiffness unchanged. Width-resolution options keep 40 cells in
the fold direction; a 16-cell width / 1,200 Hz run checks faster iteration
without relaxing the independent intersection tests. Both full episodes start
unfolded, without imposed folds or tool attachments.

The core assessor now also requires complete per-substep contact-capacity
observation and zero persisted overflow events. Six hold/negative-control
tests pass. No completed seal or final demonstration video is available yet.
