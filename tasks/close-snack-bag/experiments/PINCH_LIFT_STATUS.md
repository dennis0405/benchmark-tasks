# Mouth grasp, lift and hold only

User scope: remove folding and clip insertion; verify that both hands really
grasp the unfolded mouth and lift its material. This is an RSS/Newton component
test, not the Arena scorer or a whole-bag suspension test. The hands start
open around the mouth; no particle is attached, and only robot cursor/grip
requests change during the episode.

## Observation contract

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
fold-over trajectory. Their results are pending at this checkpoint.

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
  --grasp-inset-mm 6 --lift-pitch-deg 0 --lift-advance-mm 0 \
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
