# Close a snack bag with a spring clip — experimental work in progress

**Not ready for submission.** Instruction: “Keep the chips from going stale.” The 160 mm wide packet rests on a 25 mm support board on the table, leaving its mouth overhanging the front edge, and starts with its mouth folded twice. It should gradually reopen unless it is pressed down or held shut with a clip. One YAM arm holds the fold while the other squeezes a spring clip's handles, inserts the folded mouth between its jaws, releases the clip, and withdraws. A 100 mm-wide main clip, a wide clothespeg-style alternative, a small paper clip and a rubber-band-shaped distractor are part of the intended scene. Both wide clips use the same scoring rule.

The packet's printed `CHIPS` label, shaded pillow-like surface and crimp stripes are original procedural graphics. The visible skin is non-colliding. The body is a rigid, hollow shell with a partially open front with four rigid spherical contents; the folded mouth consists of three thin rigid panels and two spring-loaded hinges. It is not continuous deformable film, the band is a rigid approximation, the contents are not individual chips, and air leakage and packet tearing are not simulated. Physical values and robot-force limits have not been measured on real items. These limits must be addressed or disclosed before a submission.

## Implementation

- `prototype.xml`: MuJoCo geometry and spring mechanisms; the motors are for isolated tests only.
- `arena_scene.py`, `scenekit.py`, `assets/`: Isaac Lab-Arena scene, monitoring and experimental scorer. The task's initial joint positions are restored once after simulator settling; no object poses or joints are written during robot actions.
- `demo_robot.py`: a contact-only, bimanual YAM controller that uses absolute 14-D joint commands and measured object states. A single scripted run reached the candidate full score; it is not a learned policy.
- `physical_probe.py`: isolated retention diagnostic that directly places the clip at reset. It is not a robot rollout and its video must not be used as a demonstration.
- `generate_mjcf_assets.py`, `design_bag.py`: procedural source for the converted assets and appearance.
- `test_scoring.py`: synthetic geometry/score checks, including the alternative clip, released hold, open clip, and spill override. They do not establish physical scoring validity.
- `evidence/scripted/`: the actual trial's result, setup screenshot and three camera videos; these files still need to be uploaded to the task board for a submission.

The YAM controller squeezed the clip handles, lifted and carried the clip, inserted it over the folded mouth, then withdrew both arms. On the current assets and candidate scorer, one recorded trial (`/data/snack-runs/final-test/`) reached 1.0 in 404 control steps, with no spill violation. Earlier trials failed. One scripted run is not a success-rate estimate and does not validate the simulator's real-world fidelity or the candidate scorer's resistance to all exploits.

On the current v5 clip, a **separate** placement diagnostic held the prepositioned clip and folded panels for more than 3 seconds and triggered the candidate 1.0 state; this diagnostic writes the initial clip pose and therefore is not evidence of robot feasibility. In a MuJoCo joint-actuation check, 0.07 N m opened the jaw tip about 22 mm. An idealized 18 mm handle lever corresponds to approximately 3.9 N at that torque; this is not a measured gripper force. The available MuJoCo pull comparison remains inconclusive and is not proof of retention.

On the final scene, a YAM no-action control scored 0 while the two fold angles moved away from their initial folded values. A separate placement diagnostic for the wide peg also triggered the same candidate 1.0 after the hold; it is not a robot demonstration of using the peg.

## Scoring under development

The intended score is 0 for a fallen packet or spilled contents; 0.5 if a clip grips the folded mouth; and 1.0 if it covers at least 96 mm of the 160 mm mouth and keeps it folded for 60 consecutive control steps at 20 Hz after both grippers withdraw. The current code approximates jaw contact using joint angle, alignment and distances rather than measured contact forces. It uses link-origin distances as a conservative approximation to gripper clearance. The score has only synthetic tests; contact, retention, spillage and false positives must be validated in real simulator runs before it is trusted. The 3-second counter assumes 20 Hz; the task limit is 2,400 control steps.

## Run the experimental controller

```bash
source /data/IsaacLab-Arena/yam.env
cd /data/benchmark-tasks/tasks/close-snack-bag
python -m unittest -v test_scoring
python demo_robot.py /data/snack-runs/attempt --record
```

The runner writes its actual result and camera videos. The recorded 404-step trial is one fixed-layout trial with `object_jitter=0`; it is not a multi-seed success rate. It used the Arena environment from the simulation walkthrough at 20 Hz, 320 x 176 camera frames, and the candidate scorer. Record a new run whenever the task code or assets change.

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
