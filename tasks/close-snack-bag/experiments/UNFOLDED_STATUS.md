# Unfolded pouch research — 2026-11-14

**Work in progress. No completed deformable sealing demonstration yet.**
`task.py` / `arena_scene.py` still preserve the older rigid Arena task. These
experiments use the pinned RoboSimStudio + MuJoCo + Newton VBD environment,
not a verified replacement for the public Isaac/Arena runner.

## Verified components and rejected trials

| Trial | Observed result | Scope |
|---|---|---|
| `unfolded-self-contact-v7b` | 194 frames; 14 selected states without detected strict crossings/BOX intersections | Hands continue holding the fold; selected snapshots only |
| `unfolded-ramp-substeps-v8` | 14 transient BOX hits in 8,400 substeps | Rejected despite a clear final snapshot |
| `unfolded-short-lay-v9` | 16,320 substeps with zero detected hits | Failed the earlier pre-press motion gate |
| `unfolded-stack-frame-v4` | 74,520 substeps with zero detected hits | Nominal side regrasp missed the actual film |
| `unfolded-surface-stack-v6` | 89,040 substeps with zero detected hits; all four ply groups inside the measured finger gap | Rejected by an instantaneous contact-force gate; no lift or retention proof |
| `rigid-clip-high-carry-v6` | Grasp, spring opening, lift, turn, carry, 3-second hold; final target error 0.218 mm / 0.0172 rad | **Rigid tool component only: no film** |
| `raised-workspace-smoke-v1` | Two control frames, 240 substeps, finite and zero detected hits | Reset smoke only, not robot manipulation |

The last-substep force observable is not a time-integrated load. In the
surface-stack trial, the final left-finger result was `[0, 0.355] N` while the
preceding 30 control samples averaged `[3.93, 3.21] N`. The current controller
uses a 0.75-second observation window, checks actual four-ply surface capture,
and still requires a physical lift/retention test. No attachment is used.

The revised parking stand ends 15 mm before the clip hinge. A previous longer
stand overlapped the rigid jaw at reset and launched the tool. This was a scene
error; those trials are rejected. `check_rigid_tool.py` now rejects initial
clip/BOX interference, including opposing jaw/handle geometry. Its scope does
not include robot mesh/capsule contacts.

## Current mechanics under test

- Initial state: a sewn 200 x 160 mm pouch, closed bottom/sides, open flat
  mouth. No initial imposed fold or particle attachment. The authored
  `--ready-start` reset places open hands around the unfolded rim.
- YAM pads are 12 x 4 x 80 mm. A 90-degree wrist turn leaves a finite upright
  strip inside the grip; release and pressing must finish folding both panels.
- `raised_forward` moves the bag 80 mm forward and uses a 200 mm support.
  Fifteen nominal FK waypoints were reachable within 0.004 mm in the offline
  plan. Actual dynamic reach/contact still need validation.
- A narrow 120 mm neck apron gives lower-finger access while a full-width
  board supports the pouch body. The clip parking rack is at Y=-300 mm.
- The new clip has divergent levers, end bosses, 100 mm jaws, a 0.4 mm closed
  gap, and an uncalibrated 0.2 Nm/rad passive spring. End squeezing opens its
  hinge; no motor or external pose command actuates the clip.
- The latest follow-up press sweeps along the fold with a central pad after
  releasing the raw rim. Side grasp planning clips actual film triangles
  against the pad footprint; it does not depend on a vertex falling inside.
- `ToolFeedback` routes the previous substep's full contact wrench onto the
  clip. This is explicit staggered coupling, not an implicit two-way solver.
  Loaded contact and timestep convergence remain to be proved.

The live complete trials `unfolded-raised-permanent-v1` and
`unfolded-raised-laminate-v2` were launched before the newest central pressing
path. Their frozen sources, rather than the current files, define each trial.

## Materials and limitations

All material values are **uncalibrated hypotheses**. Thickness is 80 micrometres
and areal density 0.08 kg/m². Collision thickness is larger: a 0.3 mm particle
radius and 0.5 mm self-contact interaction distance are numerical contact skins.

- `derived`: E=100 MPa, plastic-rest recovery 0.1/s; previous baseline.
- `permanent`: same stiffness, zero recovery; crease rest changes remain.
- `laminate`: E=1 GPa, zero recovery; a comparison for a less floppy wrapper.
- `paper_matched`: upstream numerical coefficients; not a measured laminate.

The 1 GPa comparison is below the 2.507 GPa MD modulus listed for one
[BOPP product](https://www.derflex.com/flexible-BOPP-packaging-films.html).
That product is a 20.7 micrometre cigarette film, **not our snack package**;
it provides scale only. Toray explains the tensile-modulus measurement in its
[film mechanics reference](https://www.films.toray/en/knowledge/kno_002_2.html).
An actual multilayer wrapper needs thickness, bending, friction and recovery
measurements; anisotropy/delamination/tearing are not represented here.

The research scene is currently **empty**. Internal air, contents, fracture,
spill penalties, Arena integration and Dream inference remain outstanding.
`assess_unfolded.py` reports only an empty-pouch mechanics assessment and always
leaves `submission_score` unknown. It must never borrow the rigid task score.

## Reproduction

Use the pinned RSS commit `4e52e1a76437ef1296028afea744437f6e9469c0` and the
existing CUDA-12 Warp environment. Do not run `uv sync` into that environment.
From `/data/robocurve/robosimstudio`:

```sh
PYTHONPATH=. JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /data/robocurve/envs/rss/bin/python \
  /data/robocurve/tasks/close-snack-bag/experiments/fold_pouch_robot.py \
  /fresh/output --ready-start --drive --clip --stage seal --hz 20 \
  --physics-hz 2400 --workspace raised_forward --material-profile permanent \
  --check-substeps --frames 3000
```

The controller records actual film, rigid, joint-position and joint-velocity
states. Every run freezes its sources and parameters. Outputs must be new
folders. A completed command generator or `all_finite` is not success.

`substep_geometry.py` checks strict transverse self-crossings and film/BOX
intersections after every physics substep. It excludes coplanar overlap,
non-BOX colliders and motion between solved states. `clip_capture.py` measures
actual four-ply coverage between jaw planes. A final success claim additionally
needs at least three continuous seconds with both hands clear, retained fold,
loaded jaws and all the scoped geometry checks passing.
