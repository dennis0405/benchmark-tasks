# Unfolded pouch research — 2026-11-14

**Not a completed sealing demo or a submission-ready Arena replacement.**
The recorded rigid task remains in `task.py` / `arena_scene.py`. The new
experiments run YAM through pinned RoboSimStudio, MuJoCo and Newton VBD.

## Observed evidence

- Brev `robocurve-snack-film` restarted with an idle L40S. Previous padded
  clip work is preserved at `d5e68ca`.
- The pouch starts unfolded, with no particle attachment or object pose writes
  during actions. The curated `--ready-start` layout puts open grippers around
  the mouth at reset, on a 100 mm block.
- A 90-frame pinch control recorded bilateral finger contact at frames 60 and
  90, about 4.3–5.5 N per side in the last-substep simulator observable.
  This is not measured hardware force or a sustained-hold test. Compact raw
  evidence and frozen sources: `../evidence/unfolded-20261114/pinch-90/`.
- An arc control reached a median fold angle of 2.92 rad, then lost the fold.
  Five selected states (0, 90, 330, 390, 420) had no detected strict transverse
  self-crossings. Coplanar overlap and unsampled events are not covered.
- A later sampled box check found 30 pouch/board triangle intersections at
  reset, and zero at nine later samples. The new reset is 0.7 mm higher, with
  fail-fast initial box checks. The old run is not collision-free evidence.
- A free clip and experimental reaction hook stepped 30 frames with finite
  states, away from the pouch. This does not validate loaded clip contact.

## Mechanics and outstanding work

`deformable_scene.py` exposes `make_environment` / `read_object_states` for the
research runner. Matching names do not establish shared Isaac/Arena support.
Driver, controller, sampled geometry checker and recorded-state renderer are
under `experiments/`.

The default `derived` material is an uncalibrated 80 micrometre isotropic
equivalent. Optional `paper_matched` is a numerical bending reference from the
pinned paper example, not a measured snack laminate. Open-hand reset postures
are in `assets/yam_ready_pose.json`. Restore them after warm-up with
`reset_joint_q=True`. VBD-only controls require structural FK consistency but
do not support YAM effort limits; they are not the chosen robot path.

The controller keeps whichever hand still has bilateral contact while the
other presses, then exchanges support. This motion remains under validation.
`tool_feedback.py` adds the previous substep's full contact wrench to the clip's
rigid solve. Routing tests and an inactive-contact smoke passed; loaded contact
and timestep convergence remain unverified.

No completed 3-second clip hold, deformable task score, contents/air integration,
Dream run, media upload or submitted PR exists in this continuation. The rigid
rollout's score must not be used as the deformable score.

## Fresh diagnostic

From the pinned RSS checkout and existing CUDA-12 Warp environment:

```sh
PYTHONPATH=. JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /data/robocurve/envs/rss/bin/python \
  /data/robocurve/tasks/close-snack-bag/experiments/fold_pouch_robot.py \
  /fresh/output --ready-start --drive --clip --frames 2400
```

The research driver records at 60 Hz; `--clip` uses 40 physics substeps per
frame, otherwise 20. Outputs must be fresh directories, with frozen sources.
Check `controller_failure`, motion, contacts and geometry; `all_finite` is not
task success. Final reproduction still needs the public 20 Hz path.
