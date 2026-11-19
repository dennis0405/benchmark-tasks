# Install a clip-on wheel balance weight

A bimanual YAM mounts a 29 g spring-clip weight at the green marks on a stationary rim flange. The left gripper supports the weight while the right swings a 120 g wheel-weight hammer. The benchmark supplies the balance correction and target location in advance.

## Rig and reset

Use one fresh process per trial. The asset contains the bench fixture, rim, green marks, rests, hammer, crown and two independently sprung jaws. All positions and collision shapes are in metres in `assets/rig.usda`. `task.py` places the fixture origin at (0.31, 0, 0.845) m and sets gravity, cameras and physics through the YAM environment. Load the asset in its authored pose and allow the runner to settle the objects. Both runs recorded here use seed 0 and a fixed layout. A tabletop reproduction would require a secured flange, compatible clip, tool rests and a force gauge at the crown. Actual clip geometry, friction and release loads require calibration on physical hardware.

## Reproduce

The recorded host ran Isaac Sim 6.1.0 with CPU PhysX at 2,000 Hz and RTX camera rendering. From the task directory, mount it read-only at /task in the YAM Isaac image and mount a separate writable output directory. With an available GPU, inside the configured container run:

```sh
cd /outputs
/isaac-sim/python.sh /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none
```

The runner records one set of videos, a result and the measured trajectory per episode. In the same environment replay a native episode with:

```sh
/isaac-sim/python.sh /task/runtime/rescore.py --task /task/task.py /path/to/episode/trajectory.jsonl --expect /path/to/episode/result.json
```

The shared runner also exposes `--policy dream`. It uses the instruction, top and wrist cameras and measured robot state. The overview camera records the review video. Live model access requires operator credentials and an authorized budget. The portable runtime includes a generic spending guard. The recorded call used the same policy adapter with the operator’s account reservation guard.

## Scoring and evidence

| Cumulative score | Observable condition |
| --- | --- |
| 0.25 | Both left finger pads previously held the weight away from the mark, then support it on the mark with measured toe or jaw contact against the flange. The crown must fall within 6 mm of both horizontal target coordinates and in the allowed height band. |
| 0.50 | Both right pads hold the handle and the hammer nose strikes the clip bridge. Impulse from nose-to-bridge contacts with closing speed above 0.12 m/s must total more than 0.002 N s within the control frame. At each contact the observer checks relative vertical body speed over the immediately preceding eight physics samples (4 ms). It samples throughout the episode. |
| 1.00 | Both toes previously passed under the lip before loading. With both hands clear and the released hammer supported on its rack, hold 4 N for 60 consecutive whole control frames with rim contact. Both toes must stay below the height limit, the weight must stay on the mark and the crown must drift at most 1 mm from the first eligible loaded position. |

The virtual tester arms when the hands and hammer are clear from step 355 onward. Force increases by 2 N/s up to 4 N. The force stops if the crown moves more than 12 mm from the ramp start or the test times out. The episode ends at success or 490 steps (24.5 simulated seconds). The 4 N load is a benchmark choice. No production or road-use acceptance load has been established.

Both rack posts must register load-bearing contacts against the hammer during each qualifying control frame. The contacts must be predominantly vertical (absolute vertical normal component above 0.8) and carry vertical impulse. Hammer linear speed must stay below 10 mm/s and angular speed below 0.25 rad/s at each sample. Losing either support or disturbing the hammer breaks the consecutive-frame count.

The fixed episode limit requires clearing the fixture by roughly 19.5 s to leave time for the 2 s ramp and 3 s hold. The left and right hands have assigned roles. Numerical thresholds describe the simulated fixture.

## Occupational context

Mounting a weight on a rim is an explicit Core task for tire repairers and changers. The broader automotive service technician occupation includes wheel service with balancing machines. In May 2025 the US OEWS estimated 704,640 employees in the latter occupation. This figure covers the entire occupation. [O*NET tire repair tasks](https://www.onetonline.org/link/details/49-3093.00) · [O*NET automotive service tasks](https://www.onetonline.org/link/details/49-3023.00) · [BLS May 2025 employment](https://www.bls.gov/news.release/archives/ocwage_05152026.pdf)

## Recorded results

Scripted YAM seed 0 scored 1.00 in 479 steps (23.95 s). Dream-1 seed 0 scored 0.00 in 490 steps (24.50 s), stop reason step_limit. The independent missed-rack control released the hammer en route to the rack and peaked at 0.50.
