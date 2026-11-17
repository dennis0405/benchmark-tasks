# Separate ceramic finishing media from steel washers

Eight angle-cut ceramic rods and two steel washers rest together on a supported circular screen. Use the two YAM arms to separate the rods into the receiver, keep the washers in the screen, and return and release the screen. The screen, rods and rings are free rigid bodies. The policy sends joint targets and gripper commands to the standard arms.

## Environment

The native runs use Isaac Sim 6.1.0.0, IsaacLab-Arena revision `c7edf3a9d5d8276565208f72422f8f8a0806d6cf` and the YAM integration at `0059d02a1a91bc6834c1d0846022891df0c78730`. Start the runner with the Arena environment's Python interpreter and the YAM package installed. The task uses NumPy and Shapely for procedural geometry and ImageIO with the FFmpeg plugin for recording.

Control runs at 20 Hz and physics at 240 Hz. The PGS solver is configured on this scene. The same screen with the original TGS setup became unstable under the mixed load at reset. The sieve has 61 circular openings on a 30 mm pitch. Each 20-sided approximation has a 27 mm vertex diameter and 26.67 mm across the flats. The receiver bottom generates contacts within 8 mm to resolve fast landings while its rest surface stays at its modeled elevation. The ceramic rods are 9.525 mm in diameter and 22.225 mm between their oblique face centers. Each steel ring has a 30 mm outside diameter, an 8 mm bore and a 4 mm thickness. Friction and mass properties are simulation assumptions. All objects are reusable after reset.

## Scoring

Only a rod that traverses one aperture and is caught in the receiver earns recovery credit. The grader uses the moving physical screen as its reference and observes every completed physics substep. Recovery contributes up to 0.90. With all eight rods recovered and the two washers retained, a continuous three-second released dwell on both supports contributes the remaining 0.10. The bowl or washers leaving their permitted volumes and invalid crossings are checked separately from the recovery fraction.

## Run and check

Run these commands inside this folder with the IsaacLab-Arena Python environment active. Shapely and the YAM package must be installed in that environment. The script defaults to the fixed trial seed 8261. It writes the final native score, the 20 Hz actions and poses, and the 240 Hz rigid-body poses beside the requested report.

```bash
HEADLESS=1 python demo.py --device cuda:0 --render-video --report results/demo.json
HEADLESS=1 python demo.py --device cuda:0 --idle-steps 120 --report results/idle.json
HEADLESS=1 python demo.py --device cuda:0 --replay results/demo.trace.json --report results/replay.json
python -m unittest test_passage test_outcome
```

For another policy, start Isaac Lab, call `task.make_env(device="cuda:0", enable_cameras=True)`, and call `env.reset(seed=8261)`. Send one finite 14-dimensional YAM action per `env.step(action)`. Each result's info dictionary contains `sieve_score`. Call `env.reset()` for a new episode to clear passage history, faults and dwell. The wrapper limits an episode to 4,800 control actions. The demo only commands arm joints and gripper openings.

The scene uses simulated rigid contact and assumed material coefficients. No physical hardware evaluation is included.

## Inspect Robots

The task-local adapter is declared in `native-runtime.json`. Register it with the RoboCurve workflow native runtime before constructing the YAM embodiment. The adapter changes scene construction and preserves YAM reset, observations, action layout and score reporting. Use a fixed layout with `object_jitter=0` and `control_hz=20`. Dream-1 used the calibrated top and two wrist cameras at 320 × 176 pixels. The overview recording shows the same episode from outside the robot.

## Recorded results

| Policy | Final score | Rods recovered | Washers retained | Control steps | Simulated time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dream-1 (`dreamlabs/dream-1-beta`) | 0.4592 | 4/8 | 2/2 | 4,800 | 240 s |
| Scripted bimanual YAM | 1.0 | 8/8 | 2/2 | 597 | 29.85 s |

Dream-1 reached the action limit with four rods in the receiver. Its recorded score was 0.45918367346938777. The scripted demo recovered all eight rods and kept the released screen on both supports for 4.4125 seconds. Neither run triggered a scoring fault.
