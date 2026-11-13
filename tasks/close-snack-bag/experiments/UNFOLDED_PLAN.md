# Unfolded-mouth continuation — 2026-11-14

The user selected a sewn snack pouch lying on the table, its mouth flat and
unfolded. The robot must grasp, fold, press and apply the clip. Do not use the
pre-folded reset or a prescribed particle trajectory as a robot demonstration.

1. Preserve the previous stationary clip probes (checkpoint `d5e68ca`).
2. Prove a YAM plus sewn-film scene, then one contact-driven fold before adding
   the second fold and clip. Use real gripper contacts, no particle attachment,
   no object pose writes during rollout. Check the actual shared runner API;
   naming an RSS module `task.py` does not establish Arena compatibility.
3. Validate mouth fold geometry, release retention and intersections separately.
   The old 80 mm loaded span / every-pad gate is a diagnostic, not task success.
4. Connect tool grasp/open/insert/release, then a 3-second hands-free hold.
   Retain failed and no-clip controls; a successful probe is not a full demo.
5. Integrate contents/spill/damage and clearly bounded air/seal proxies; record
   real trajectories, setup image and video. Update task metadata only to match
   the scene actually validated.

The initial research scene uses pinned RSS paper/YAM one-way coupling with
finger-slide force feedback. That is an upstream folding reference path, not
verified general two-way clip-film physics or Isaac Sim integration. Test the
latter before claiming a completed, submission-ready task.

Use fresh run directories under `/data/robocurve/runs/`. Full raw trajectories
stay on the instance; retain compact results and final videos in this branch.
Measure solver/checking/recording time before optimizing. Keep the instance
running unless the user requests stop.
