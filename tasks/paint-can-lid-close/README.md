# Close the paint can

Steady the can with one hand, tap the loose lid down around the rim with the rubber mallet, return the mallet to the cloth, and release the work. The trial has a fixed initial state and a limit of 1,800 control steps at 20 Hz. Isaac Sim 6.1.0 runs PhysX on CPU at 480 Hz and renders the three policy cameras and the observer camera on an RTX GPU.

## Repeatable dry setup

Use an unused or fully cleaned and dried one-US-gallon steel ring-and-plug can (168 mm diameter, 190.5 mm height), a matching metal lid and a dry folded canvas cloth (620 × 500 × 4 mm). Secure clean steel ballast low in the can, centring it horizontally, and weigh the can and ballast together to 2.40 kg. Keep the groove and tool faces clean and dry. The simulated can's centre of mass is 40 mm below its mid-height. The side tab is a rigid 14 × 76 × 18 mm pinch surface on the +y side with its centre 121 mm from the can axis and 110 mm above the cloth. Use the grippable end to steady the can.

With x away from the arm bases and y to the left, the cloth centre is (370, −20) mm, can centre (340, −5) mm and mallet head centre (400, −240) mm. The head lies on its cylindrical side, its handle pointing 120 degrees from +x. The rubber head is 51 mm in diameter and 134 mm long. The 0.31 kg mallet has a 25 mm wide wooden handle reaching 255 mm from the head centre. Match the initial screenshot with the arms open.

## Judging at the bench

1. Mark the eight sector centres at 45-degree intervals. With a matching lid fully seated, record each local lip height relative to the adjacent can rim using a depth gauge or a straightedge and feeler gauge. Put millimetre reference marks in a fixed side camera view. Reset each sector to 4 mm proud of its seated height. If a used lid is permanently bent, replace it and measure its seated reference again.
2. From synchronized overhead and side views, identify which part of the rubber face hit each sector, whether both grasps were maintained, and whether the strike drove the lip down. A transfer across the boundary can seat two sectors. A later press or a strike on a previously seated sector earns no new sector. Verify each credited height within 1 mm of its reference; remove the credit if it rises out of tolerance.
3. Award 0 for 0–2 sectors, 0.25 for 3–5, 0.5 for 6–7, and 0.75 for all eight. Full score also requires the mallet stationary on the cloth and both open hands at least 50 mm clear continuously for 3 s with all eight sectors remaining seated. Cap at 0.5 if displacement from the start exceeds 15 mm or tilt exceeds 8 degrees. A tool dropped below the bench or a toppled can scores zero.

In simulation each sector has a 4 mm slide with two spring equilibria. Contact is measured by the solver. A hit requires the near-vertical head to approach faster than 0.12 m/s and deliver more than 0.012 N·s of rapid vertical contact impulse with both grips present. The 2 mm engagement midpoint must be crossed in the impact frame or the following 50 ms frame. The current insertion must reach 3 mm. A released or reopened sector must qualify again. The steel fit is represented by a 4 kN/m spring, without sheet-metal denting. The impact resistance and the enlarged mallet have not been calibrated against physical hardware. For a physical transfer check, film a glancing pass, a light touch, and a successful tap on the specified dry can; measure the impact speed and local depth before and after each. No physical calibration result is claimed.

## Run and inspect

Mount this directory read-only at /task and a separate writable directory at /outputs in the yam-inspect-isaac:6.1.0 image. Enable the GPU for camera rendering. From the image run:

    /isaac-sim/python.sh /task/run.py --policy scripted --episodes 1 --seed 0 --device cpu --viz none

For the policy episode use the same command with --policy dream and --budget-path pointing to an owner-only episode reservation mounted with a writable receipts directory. DREAM_CREDENTIALS identifies the owner-only Dream key file on the evaluation host. Scoring and sector measurements are withheld from the policy. Each episode saves the camera recordings, the source hashes, trajectory.jsonl, and result.json. Use the paired result and recording from that episode.

## Recorded trials

Seed 0, revised task on Blackwell. Dream-1 scored 0.00 in 1,800 steps (90 s). Scripted YAM scored 1.00 in 768 steps (38.4 s). Both trials used the same task, seed, scoring code and camera setup. The overview camera is for human review. Dream receives the top camera, both wrist cameras and the joint/gripper state. The corrected score was also checked against a hover-only contact control and delayed-press, pre-seated and reopening tests.

The live scene is constructed by task.make_environment from rim.can and rim.sector. The shared paint geometry module retains descriptions of the original wet prop. Only the dry version above is used in these trials.
