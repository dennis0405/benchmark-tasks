# Mouth grasp, lift and hold only

User scope: remove folding and clip insertion; verify that both hands really
grasp the unfolded mouth and lift its material. This is an RSS/Newton component
test, not the Arena scorer or a whole-bag suspension test. The hands start
open around the mouth; no particle is attached, and only robot cursor/grip
requests change during the episode.

## Active observation contract: v3

The user explicitly requested that small footprint excursions no longer stop
the test. The same lifting commands now run through the hold unless physical
release or invalid/intersecting physics is observed. This is a component-test
change, not a change to the public task scorer.

- Pad-footprint flags, original point escape and accumulated displacement are
  logged and displayed; they neither stop nor reject retained contact.
- Physical separation is checked against all BOX/CAPSULE/SPHERE colliders on
  both finger bodies, including the wing geometry, not just the 12 mm pad.
  Separation over 10 mm, or separation over 5 mm together with absent force,
  must persist for 0.25 s before declaring a hand has released the film.
- An incomplete position target is recorded rather than aborting the hold.
- For the positive 3 s verdict, both hands need opposed mean contact presence,
  both mouth walls within 3 mm of actual finger primitives, and the original
  tracked material on both walls at least 40 mm above its acquisition height.
- Intersections, nonfinite states and incomplete collision-capacity data still
  invalidate the test. No artificial object attachments or position writes.

The original 50 mm lift and 35-degree tilt are unchanged in v76. The new
`verify_contact_hold.py` independently recomputes material rise and distances
from recorded particle/body states and reads primary mean force vectors;
cursor targets and footprint flags cannot establish a pass. Tests cover a
retained grasp despite escaped fiducials, real separated hands, and film at
the end of a long pad far from the jaw centre. Fifteen focused tests pass.
v76 completes its diagnostic run at frame 148: right-hand release is confirmed
from five consecutive separated observations. Contact is zero, with nearest
film-particle/finger separation increasing from 19.7 to 28.3 mm. The left hand
also has zero contact and 24.2 mm separation at the terminal state. The script
does not stop on footprint escape; it stops on actual separation. Three-second
bimanual retention fails. The independent raw-state audit agrees with the
recorded heights/distances and confirms the right-hand release interval.

All 17,760 physics steps have zero observed intersections and no contact
overflow. One cracker seam failure is recorded. The 7.45-second diagnostic
video, full raw recording, exact execution source and audit are preserved in
`evidence/pinch-lift-20261114/pinch-contact-hold-v76`. No further test is running;
Brev remains on. This was a fresh episode with the same settings/commands as
v74, not a resumed solver state.

## Historical observation contract: v2

- Both the front and back mouth wall must occupy at least 50 mm2 of each
  measured main-pad cavity. Both opposed contact readouts must be present.
- Five central material vertices from each wall are selected at acquisition,
  separately for each hand. All must stay inside that hand's pad cavity.
- The original material on each wall must rise at least 40 mm.
- At the raised pose, hold for at least three continuous seconds with no more
  than 3 mm additional p90 displacement of the original material in each hand.
- Preserve finite states, every-substep independent geometry/capacity checks,
  no observed intersections, an uninterrupted run and completed commands.

Protocol v1 rejected any displacement over 3 mm from acquisition while lifting.
v68/v69 stop at 32/35-39 mm rise; v70 stops at 26-27 mm. These are failures of
that strict v1 test, not proof that the cloth had left the fingers. A raw-box
inspection of v70 finds all five central vertices per wall still in the pads.
Protocol v2 therefore reports initial seating separately and measures holding
creep from hold entry. It adds original central-vertex containment; it does not
substitute newly encountered cloth or infer success from robot motion. Old
results remain failed, with their original evaluator in each source archive.

v67 at 1,200 Hz detects a contents/film intersection and is rejected. Further
tests use the unchanged 2,400 Hz geometry guards. v71's 12 mm inward translation
does not improve its v1 result. v72 tests the 6 mm-inset grasp with protocol v2;
v73 additionally tilts the held strip by 35 degrees while lifting, without a
fold-over trajectory. v72 stops at 131 frames on original landmark escape;
v73 stops at frame 63 on a contents/film intersection before lifting.

The repeated 35-degree trials finish as failed component tests: loaded v74
stops at 137 frames, empty v75 at 133. In v74, original film actually rises
36.4-36.9 mm; raw physical pad transforms confirm 5/5 original central points
inside on each left wall, and 4/5 plus 5/5 on the two right walls. One point
exceeds the main pad's 6 mm half-width by about 29 micrometres. This is a
conservative footprint-gate failure, NOT proof of complete physical release.
No three-second hold is observed, so retention remains unverified.

v74 checks all 16,440 physics steps with zero observed intersections, no
contact-capacity overflow, and no recorded cracker seam failure. This is
one diagnostic run, not robustness or physical-material validation. Its
6.9-second recorded video, complete raw states (9.97 MiB), frozen sources
and raw geometry audit are under `evidence/pinch-lift-20261114`.

No further trial is running after this review checkpoint. Brev remains on.

Focused tests: 12 pass, including moving-empty-hand, single-wall, lost-original-
landmark, ongoing hold creep, missing-frame and interruption controls. The
contact-force values are qualitative and uncalibrated. The film and predefined
cracker seams are also uncalibrated; fracture/spill scoring is outside this
component verdict and must be assessed separately before a full task demo.

## Reproduce a fresh test

Use the existing pinned Python environment and RSS checkout. Do not blindly
sync dependencies: this server needs the CUDA-12-compatible Warp installation.
From `/data/robocurve`, use a fresh execution-copy and output directory:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/data/robocurve/mps/pipe
export CUDA_MPS_LOG_DIRECTORY=/data/robocurve/mps/log
export PYTHONPATH=/data/robocurve/robosimstudio
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 JAX_PLATFORMS=cpu
/data/robocurve/envs/rss/bin/python \
  tasks/close-snack-bag/experiments/launch_isolated.py \
  /data/robocurve/experiment-code/NEW-NAME fold_pouch_robot.py runs/NEW-NAME \
  --stage pinch_lift --drive --ready-start --coupling unified \
  --contents --contents-mode brittle --workspace raised_flat \
  --material-profile laminate --yield-curvature 10 \
  --mouth-half-gap-mm .2 --self-radius-mm .1 --self-filter-rings 0 \
  --grasp-inset-mm 6 --lift-pitch-deg 35 --lift-advance-mm 0 \
  --arm-gain-scale 10 --finger-contact-ke 1000000 --rigid-contact-ke 10000 \
  --rigid-contact-threads 32 --body-particle-buffer 16384 \
  --mouth-support full --support-extension-mm 14 --board-skin-mm 1 --rebase-z \
  --hz 20 --physics-hz 2400 --check-substeps \
  --force-readout mean --record-force-means --frames 420 --checkpoint-every 30
```

`pinch-lift-assessment.json` is written automatically with raw-recording and
trace hashes. Render only the recorded trajectory, with visible tracking dots:

```bash
PYOPENGL_PLATFORM=egl /data/robocurve/envs/rss/bin/python \
  tasks/close-snack-bag/experiments/render_unfolded.py runs/NEW-NAME \
  --stride 1 --detail mouth_front --translucent-robot-detail
```

The close-up translucency and vertex dots are display aids, explicitly labelled
in-video. No recorded position is interpolated or edited.
