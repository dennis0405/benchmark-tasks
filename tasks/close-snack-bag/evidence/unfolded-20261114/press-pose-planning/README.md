# Offline press-pose planning

`press-pose-grid-v2.json` contains 156 left-arm tip-pose queries, 49 accepted
by the stated FK/IK tolerance. This is not dynamic or collision evidence.
The proposed forward 25-degree-down wrist keeps the original closing-axis
sign and reaches the tip at (0.70, 0.07, 0.97) m with 0.45 micrometre
position residual. No robot or cloth state was prescribed during a rollout.

Model: pinned RoboSimStudio YAM, `raised_flat`, thin clip, 19 mm finger
slide. Initial joint vector: observed full-support-v8 step 653, preserved in
`../component-resets/edge-drag-seed-v8/`. Reproduce using
`experiments/plan_press_poses.py SEED OUTPUT.json --grid`.
