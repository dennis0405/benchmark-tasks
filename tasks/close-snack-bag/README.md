# Close a snack bag with a spring clip — experimental work in progress

The unfolded-mouth YAM continuation runs in a separate RoboSimStudio research
scene; its evidence and remaining limitations are in
[UNFOLDED_STATUS.md](experiments/UNFOLDED_STATUS.md). It does not replace the
rigid Arena task or establish a deformable submission score.

**Not ready for submission.** Instruction: “Keep the chips from going stale.” The 160 mm wide packet rests on a 25 mm support board on the table, leaving its mouth overhanging the front edge, and starts with its mouth folded twice. It should gradually reopen unless it is pressed down or held shut with a clip. One YAM arm holds the fold while the other squeezes a spring clip's handles, inserts the folded mouth between its jaws, releases the clip, and withdraws. A 100 mm-wide main clip, a wide clothespeg-style alternative, a small paper clip and a rubber-band-shaped distractor are part of the intended scene. Both wide clips use the same scoring rule.

For environment versions, setup on a new GPU machine, preserved evidence and
the remaining Dream-1 steps, see [RESTART.md](RESTART.md).

An experimental deformable-surface packet and isolated contact probes are documented in [DEFORMABLE_EXPERIMENT.md](DEFORMABLE_EXPERIMENT.md). They are not connected to `arena_scene.py`, the scorer, or the YAM demo: the folded-mouth/clip tests have not yet shown reliable retention without unacceptable penetration. The existing robot video and score below concern only the rigid-panel prototype.

The follow-up solver and intersection investigation is in [DEFORMABLE_REFINEMENT.md](DEFORMABLE_REFINEMENT.md). Newton VBD avoided detected surface crossings in short single-packet probes, but coupled clip contact remains unstable and is not ready for YAM integration.

The packet's printed `CHIPS` label, shaded pillow-like surface and crimp stripes are original procedural graphics. The visible skin is non-colliding. The body is a rigid, hollow shell with a partially open front and four thin rigid food proxies; the folded mouth consists of three thin rigid panels and two spring-loaded hinges. It is not continuous deformable film, the band is a rigid approximation, the food proxies cannot fracture, and air leakage and packet tearing are not simulated. Physical values and robot-force limits have not been measured on real items. The simulation can test clip placement and retention, not whether real food stays fresh or breaks.

## Implementation

- `prototype.xml`: MuJoCo geometry and spring mechanisms; the motors are for isolated tests only.
- `arena_scene.py`, `scenekit.py`, `assets/`: Isaac Lab-Arena scene, monitoring and experimental scorer. The task's initial joint positions are restored once after simulator settling; no object poses or joints are written during robot actions.
- `demo_robot.py`: a contact-only, bimanual YAM controller that uses absolute 14-D joint commands and measured object states. A single scripted run reached the candidate full score; it is not a learned policy.
- `physical_probe.py`: isolated retention diagnostic that directly places the clip at reset. It is not a robot rollout and its video must not be used as a demonstration.
- `validate_physics.py`: simulator controls for both clips, 70 mm overlap, an unclipped bag, an escaped food proxy, and a fallen packet. These deliberately place objects directly at reset; they are not robot rollouts.
- `contact_monitor.py`: PhysX contact reporting for both jaws of each wide clip against the packet's rigid bodies. It samples the last physics substep of each control step and fails closed if unavailable.
- `contact_probe.py`: an isolated PhysX contact diagnostic that directly places a clip at reset; this is not a robot demonstration.
- `generate_mjcf_assets.py`, `design_bag.py`: procedural source for the converted assets and appearance.
- `test_scoring.py`: synthetic geometry/score checks, including the alternative clip, released hold, open clip, and spill override. They do not establish physical scoring validity.
- `evidence/scripted/`: the actual trial's result, setup screenshot and three camera videos; these files still need to be uploaded to the task board for a submission.
- `evidence/contact-demo/`: a newer robot trial under the contact-based scorer, including a setup frame, three 640 x 352 H.264 videos and source hashes.
- `evidence/flat-chip-demo/`: the current rigid-approximation task's recorded YAM rollout, setup frame, source hashes and physical scoring controls.

The YAM controller squeezes the clip handles, lifts and carries the clip, inserts it over the folded mouth, then withdraws both arms. The older recording in `evidence/scripted/` scored 1.0 in 404 steps with the previous geometry-only scorer, before contact reporting was added. Earlier trials failed. One scripted run is not a success-rate estimate and does not validate the simulator's real-world fidelity or the scorer's resistance to all exploits.

The earlier fixed-layout robot trial in `evidence/contact-demo/` reached 1.0 in 404 steps with the contact-based scorer, without a spill violation. It used the prior visual skin and spherical food proxies. Both simulated jaws had to remain in contact during the 60-step release hold. On the Brev L40S instance it took 74 seconds through completion, excluding video encoding; the result records hashes of the Python source and USD assets and confirms they were unchanged during the run. The packet starts folded: the robot holds the existing folds, rather than making two new folds from an unfolded packet.

The current scene's scripted YAM rollout in `evidence/flat-chip-demo/` reached
1.0 in 403 steps, with no detected spill or fall. It took 74.4 seconds before
video encoding on an L40S, and the recorded Python and USD hashes match this
revision. The fixed-layout trial is not a success-rate estimate. After changing
the food proxies to thin boxes, independent simulator placement controls gave
0 without a clip, 1 with each of the two wide clips, 0.5 for partial overlap,
and 0 for either a displaced food proxy or a fallen packet. These deliberately
placed controls are not robot rollouts. Food fracture, actual leakage and a
physical material calibration are outside this approximation.

On the current v5 clip, a **separate** placement diagnostic held the prepositioned clip and folded panels for more than 3 seconds and triggered the candidate 1.0 state; this diagnostic writes the initial clip pose and therefore is not evidence of robot feasibility. In a MuJoCo joint-actuation check, 0.07 N m opened the jaw tip about 22 mm. An idealized 18 mm handle lever corresponds to approximately 3.9 N at that torque; this is not a measured gripper force. The available MuJoCo pull comparison remains inconclusive and is not proof of retention.

On the contact-scored scene, the placement controls in `evidence/controls/physical-controls.json` yielded 0 for no clip, 1 for the main clip and the wide peg after the hold, 0.5 for a 70 mm overlap, and 0 for a food proxy outside the packet. `fallen-control.json` records a separate packet-fall test yielding 0. The prior geometry-only controls are retained separately. These placement diagnostics are not robot demonstrations. `contact-probe.txt` records a separate pre-integration measurement of normal forces on both jaws.

## Scoring under development

The intended score is 0 for a fallen packet or spilled contents; 0.5 if a clip grips the folded mouth; and 1.0 if it covers at least 96 mm of the 160 mm mouth and keeps it folded for 60 consecutive control steps at 20 Hz after both grippers withdraw. The current scorer requires the geometric checks and at least 0.001 N of simulated normal contact force from each jaw to a packet body, at separation <= 0.2 mm. It samples contact once per control step, and returns zero if contact monitoring fails. It uses link-origin distances as an approximation to gripper clearance. The 3-second counter assumes 20 Hz; the task limit is 2,400 control steps. Contact thresholds are simulation values, not real-world calibration. Adversarial false positives and actual physical retention remain under investigation.

## Run the experimental controller

```bash
source /data/IsaacLab-Arena/yam.env
cd /data/benchmark-tasks/tasks/close-snack-bag
python -m unittest -v test_scoring
python demo_robot.py /data/snack-runs/attempt --record
python validate_physics.py /data/snack-runs/physical-controls.json
python contact_probe.py
```

The runner writes its actual result and camera videos. It accepts `--cam-width 640 --cam-height 352` for a larger recording and includes SHA-256 hashes for the Python and USD files and whether they changed during the run. The recorded 404-step trials use a fixed layout with `object_jitter=0`; they are not a multi-seed success rate. Both use the Arena environment from the simulation walkthrough at 20 Hz. Record a new run whenever the task code or assets change.

## Dream-1 evaluation

After securely configuring `~/.dreamlabs/credentials` on the instance as described in the [simulation walkthrough](https://tasks.robocurve.org/walkthrough), run:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=arena_scene.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Keep the chips from going stale." --scorer task_score \
  --max-steps 2400 --store-frames --log-dir ~/yam-eval/snack-dream
for log in ~/yam-eval/snack-dream/*.json; do inspect-robots video "$log"; done
```

**No Dream-1 score or media ID has yet been recorded.** The `task.json` is a draft; upload the Dream run, scripted demo and setup screenshot to the task board and add the resulting media ID before submitting. A nonzero scorer result is provisional until validated against contact, failure controls and the actual motion video. Work remains on the band and food models, real-world spring calibration, clip retention, scorer validation, Dream-1 evaluation, and final submission metadata.
