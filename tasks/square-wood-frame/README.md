# Square a racked wooden frame with a diagonal clamp

A fixed 7° skew in a 400 × 300 mm four-rail cabinet frame leaves one diagonal about 59 mm longer than the other. The robot must pick up the sliding clamp, seat its jaws on diagonally opposite corner blocks, narrow its span, then leave both hands clear while the clamp holds the frame square. The clamp starts off the frame. The two ends are connected to each other by a passive sliding joint. No fixture, jaw or rail is driven by the controller.

## Run and reset

Mount this task directory at /task inside the Isaac Sim 6.1.0 bimanual YAM environment. Use a writable output directory and one new process per episode. The included runtime uses CPU PhysX at 300 Hz and RTX rendering. The robot receives 20 joint/gripper commands per second. The maximum trial duration is 40 simulated seconds.

```sh
cd /outputs
/isaac-sim/python.sh /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none
```

Use the same command with `--policy dream` in a model-authorized environment. The policy receives the task instruction, three camera observations and robot joint/gripper state. It receives no diagonal lengths, contact flags or score. The scripted demonstration uses simulated corner positions to decide when to stop.

Replay the scorer against each saved episode with:

```sh
/isaac-sim/python.sh /task/runtime/rescore.py --task /task/task.py /path/to/trajectory.jsonl --expect /path/to/result.json
```

## Scoring

| Cumulative score | Requirement |
|---|---|
| 0.25 | Gripper contact with the clamp has occurred. Both ends have moved at least 60 mm from their start positions and both jaw faces contact the initially longer diagonal's corner blocks. Either clamp orientation counts. |
| 0.50 | With both jaws seated, their span has decreased at least 5 mm from first qualifying placement and the absolute difference between diagonals is at most half its initial value. |
| 1.00 | After reaching 0.50, both jaws remain seated, the diagonals agree within 1 mm and all four angles are within 0.2° of 90°. Both hands stay open and clear for at least three seconds (61 consecutive sample times at 20 Hz). |

Seating combines a PhysX jaw-face/corner contact with a corner-center test in each jaw's coordinates. The lateral offset must be under 12 mm, the height between 8 and 16 mm, and the absolute face-to-corner gap at most 1 mm. Hand clearance requires each finger to open more than 27 mm, each hand TCP to stay more than 60 mm from both grip centers and no arm-to-clamp contact in the control interval. Partial credits persist. An interrupted hold restarts the final timer. The experiment ends on success or at 800 steps.

## Real-world context and fidelity

Carpenters build and repair cabinets, doors and frameworks. Comparing diagonals and clamping across the long one is a woodworking method for correcting a loose carcase during assembly. A tabletop counterpart can be checked with a tape or two diagonal gauges, a square and a timed hands-off observation. [O*NET carpenters, 47-2031.00](https://www.onetonline.org/link/details/47-2031.00) · [Diagonal cabinet clamping](https://www.finewoodworking.com/2020/10/15/workshop-tip-shopmade-clamp-squares-up-any-carcase)

The fixture uses rigid rails, ideal corner pins, rounded corner blocks and a sliding clamp with passive drag. The benchmark checks static retention on a frictional support. It does not quantify real clamping force, elastic springback or retention under an external pull.

The US May 2025 OEWS estimate was 670,090 carpenters. This is the headcount for the occupation. Squaring a cabinet is a step within its cabinet and framework assembly duty. [BLS national employment estimate](https://www.bls.gov/news.release/archives/ocwage_05152026.htm)

## Fixture dimensions for a tabletop reproduction

The frame measures 400 × 300 mm between the corner pin centers. The four wood rails are 31 mm wide and 24 mm thick; the end tongues extend to the pins. The 28 mm diameter round seating blocks rise 48 mm above the mat. The simulated mat is 600 × 720 × 8 mm. Place the frame with the long diagonal along the line joining the clamp jaws. The clamp starts parallel to that diagonal, displaced 105 mm sideways. The grip pads are 36 × 14 × 51 mm and sit 102 mm above the mat at their centers. Each jaw face is 39 × 12 × 32 mm. The 19 × 11 mm bar connects a fixed jaw to a sliding jaw. For a physical replica, the clamping direction and the adjustable jaw must allow about 45 mm of closing travel from reset. Real material friction and hand-applied holding force would need to be measured on that rig.

## Recorded trials

Seed 0: Dream-1 scored 0.00 in 800 steps (40.00 s; step_limit). Scripted YAM scored 1.00 in 643 steps (32.15 s; success). An early-release physical control retained an 18.03 mm diagonal difference and scored 0.50 when its saved contact and pose trace was rescored.
