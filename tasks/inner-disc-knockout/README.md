# Remove only the center knockout from an electrical enclosure

Open the smaller conduit entry in a fixed, de-energized bench enclosure. The slug and the larger annulus start flush with the front panel. The two YAM grippers can reach opposite faces. Pushing past the slug after it releases can also tear out the surrounding annulus.

## Setup and scoring

Use seed 0 in a fresh process. The fixture center is (0.32, 0, 0.866) m in world coordinates. The 198 × 168 × 3 mm front panel carries a 63 mm diameter, 2.6 mm thick slug and a separate 68 mm ID, 103 mm OD, 2.5 mm thick annulus. A fixed tray catches the slug below the opening. Reset both breakable joints and allow 30 settling frames before policy actions. Rigid geometry, a limited revolute joint and a breakable fixed joint approximate scored steel. Commercial break loads and sheet deformation have not been measured.

| Score | Measured conditions |
| --- | --- |
| 0.5 | While robot contact with the slug is reported, the slug bends more than 14° relative to the ring and the two inner anchor positions remain less than 4 mm apart. This credit is retained until outer-ring damage. |
| 1.0 | After the bend, PhysX reports an inner-tab fracture and the inner anchors separate by more than 9 mm. For 20 consecutive control frames, the slug and ring centers must then be more than 45 mm apart, each tool center more than 72 mm from the fixture center, and no robot contact with the fixture may be reported. |
| 0.0 | Outer-tab fracture, absolute ring tilt above 8°, or outer-anchor separation above 5 mm cancels all credit. An untouched or insufficiently bent slug also scores zero. |

The scorer accepts either arm and either bending direction. The distance condition measures displacement of the slug center. It does not test an inserted conduit or require the slug to land on the tray.

## Reproduce a trial

The GPU renders the cameras in Isaac Sim 6.1.0. CPU PhysX steps at 500 Hz, and the robot accepts 20 actions per second. Each episode ends on success or after 240 actions (12 simulated seconds). The scripted run uses measured joint and break events to withdraw the gripper. The model receives the top and two wrist RGB images, measured joint/gripper state and the instruction.

For an existing configured YAM Isaac image, mount this task at /task and a writable output directory at /outputs. Run inside the container with a visible RTX GPU.

```sh
/isaac-sim/python.sh /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none
```

The portable build helper uses pinned sources. Run `python3 runtime/build_image.py` in the task folder to build the image. For a model trial, use the same command with `--policy dream`, an owner-only credential file supplied through `DREAM_CREDENTIALS` or the SDK environment and your configured spending ledger via `--budget-path`. Replay a recorded trajectory inside the same environment with the following command.

```sh
/isaac-sim/python.sh /task/runtime/rescore.py --task /task/task.py /path/to/episode/trajectory.jsonl --expect /path/to/episode/result.json
```

## Occupational context

Electricians install conduit and route insulated wiring between boxes. This benchmark isolates the step of opening the intended entry diameter. The U.S. Bureau of Labor Statistics estimated 821,000 electrician jobs in 2025. [O*NET Electricians, 47-2111.00](https://www.onetonline.org/link/details/47-2111.00) · [BLS Occupational Outlook Handbook](https://www.bls.gov/ooh/construction-and-extraction/electricians.htm)

## Recorded evidence

The scripted YAM run scored 1.0 in 174 actions (8.70 s). At 7.40 s the inner tab fractured. The slug landed on the tray 83.8 mm from its starting center. A trial stopped after the first bend scored 0.5 with the slug still attached at 20.5°. A longer push broke the outer tab at 6.45 s and scored 0.0. All three traces replay to their recorded scores.

Dream-1 scored 0.0 in all 240 actions (12.00 s). Five returned action chunks covered the single episode. Neither tab broke and the simulator recorded no robot contact with the center slug.
