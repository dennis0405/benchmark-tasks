# Turn a pizza in place and return the turning peel

Use the turning peel to rotate the pizza half a turn on the oven deck. Keep the pizza centered and topping-side up, then leave the blade on the front landing shelf and the shaft on the rear cradle. The asymmetrical toppings show its orientation. Either direction of rotation earns the same score.

## Why turn a pizza?

In a home pizza oven, the edge closest to the heat can brown while the opposite edge is still pale. A cook slips a turning peel against the rim and gives the pizza a half-turn so the paler edge faces the heat. The familiar skill is making that turn in a tight space while the pizza stays centered on the stone and the toppings stay upright. The cold setup below isolates the peel handling.

## Fixed setup

The stone deck is 48 × 78 cm, its top is 100 cm above the floor, and its opening is 13 cm high. The 30 cm pizza starts at world XY (0.95, −0.26) m. The 17 cm round blade starts at (0.62, −0.35) m. A 5.5 × 4.8 cm rear stand supports the shaft 34 cm behind the blade. Two 5 mm tall lips leave a 30 mm opening around the 16 mm wide shaft. The blade and shaft are one free 400 g rigid body. Both parallel-gripper YAM arms are present. The right gripper starts open above the handle.

The simulated pizza has a total mass of 283 g. Its crust has a central body, 16 compliant rim segments and 12 compressible contact pads. Toppings are fixed visual landmarks. The simulation evaluates manipulation of a set crust. It does not model heat, cooking, topping loss or empirically calibrated food mechanics. The roof and one wall use a visual cutaway and retain their colliders.

## Score

The grader samples yaw and resolved peel-to-food contact at 240 Hz. A rotation step qualifies when the peel applies more than 0.015 N outside the pizza's upper surface and at most 0.015 N to the upper surface. Side and underside contact both qualify. Tool contact anywhere on the top face or upper bevel counts toward the top load, including shaft and blade contact. Each contact is evaluated in the moving food link's frame. The top classification uses the core's 15 mm top plane and each hinged rim's upper bevel starting 4 mm above its local origin. It allows 0.5 mm for solver skin and requires a contact normal with an absolute vertical component of at least 0.35. These are simulator classification tolerances.

The grader adds signed pizza yaw changes only on qualifying steps. Credited progress is the smaller of the total net turn and qualifying net turn in the same direction, floored at zero. Reversed strokes cancel. At least two robot links must have loaded the peel above 2 N simultaneously at some point in the episode.

| End state | Score |
| --- | ---: |
| Less than 45 degrees of credited progress | 0 |
| At least 45, 90 or 135 degrees of progress | 0.2, 0.4 or 0.6 |
| Absolute net turn of 175–185 degrees with at least 165 degrees of credited progress | 0.8 |
| Same half-turn, with a continuous qualifying park at the end for at least 3 s | 1.0 |

Loss of deck support (vertical contact below 0.001 N), center excursion over 50 mm, pizza tilt over 15°, or direct robot-to-food contact above 0.1 N sets the episode score to zero. A parked peel requires vertical contact above 0.1 N on each support, robot and pizza contact below the 0.02 N threshold, and a blade center within 30 mm horizontally and 5 mm vertically of its starting position. The shaft axis must be within 8° of its starting direction and the blade within 5° of level. The grader tracks both the blade center and a shaft point 34 cm behind it. More than 2 mm of motion from the dwell anchor restarts the clock. Grasping the peel again or losing a support cancels the dwell.

## Cold reproduction setup

Prepare one pizza per physical attempt. The quantities below define the proposed test specimen. The recorded trials used the simulated pizza.

1. **Make the crust.** Use plain refrigerated wheat pizza dough from the same product and lot for a batch. Start with a 280 g portion. Roll it to a 300 mm circle on baking parchment, dock the center with a fork and leave the outer 15 mm intact. Bake bare on a sheet pan in a preheated 220 °C conventional oven for 10 minutes, then check at one-minute intervals until the center is set and the underside is dry. Cool uncovered on a wire rack until a probe inserted horizontally into the center reads 20–22 °C. Remove the parchment. Use intact bases measuring 300 ± 3 mm across two perpendicular diameters and weighing 250 ± 3 g after cooling. If a pilot base misses the mass range, adjust the next dough portion by the cooled mass difference and record the portion and bake time for the batch.
2. **Add the orientation landmarks.** Spread 23 ± 1 g of plain tomato paste in a centered 196 mm circle, leaving the outer rim bare. Punch four red bell-pepper discs and one green disc with a 32 mm cutter. Pat dry and trim the flesh side to give 10 ± 1 g for all five together. Press each disc flat into the paste. Measured from the pizza center in millimeters along the deck's world X and Y axes, place the green disc at (0, 63) and the red discs at (−55, −25), (44, −40), (35, 17) and (−40, 35). Keep the underside dry and free of loose flour, oil and parchment.
3. **Check and reset.** Weigh the assembled pizza on a tared scale. The target is **283 ± 5 g including toppings**. Confirm the center is at 20–22 °C immediately before each start and use a dry stone at the same temperature in an unheated mock oven. Brush crumbs from the stone and wipe the peel dry. Align the pizza center with the fixed setup mark and point the green disc along +Y. Reset the peel on both supports and open both grippers. Log mass, temperature and a setup photo before starting. Use a fresh matched pizza for every attempt, including retries, so previous peel strokes cannot soften or tear the next rim.

Mark the initial center and a radial line on the crust using food coloring. Put a ruler and angle scale in the deck plane. Record synchronized overhead and oven-opening views that resolve the rim and both supports. Mark the blade center and a point 34 cm down the shaft.

For a physical trial, use the following video measurements as contact proxies.

1. Start at the first frame of motion after reset. Confirm that the gripper closes on the peel and the peel moves with it. Measure the greatest center excursion and tilt, and check that some crust stays on the deck throughout. If the robot touches the food, the whole crust lifts clear, the center moves more than 50 mm or tilt exceeds 15°, record zero.
2. Use the radial line in the overhead view to unwrap yaw. In the opening view, mark each interval when the tool bears on the side or underside and moves the pizza without rubbing its top. Sum the signed angle changes over those intervals. Exclude top-rubbing and free-motion intervals. Use the smaller of the net yaw and the eligible yaw in the final direction in the score table. If occlusion prevents judging a decisive contact, mark the trial unscorable and reposition the camera before repeating.
3. For the final three seconds, check that the grippers are clear, the blade is level within 5°, the shaft direction is within 8° of its start, and the peel rests on both supports. Measure blade-center displacement from its start (at most 30 mm horizontally and less than 5 mm vertically) and verify that neither marked point moves more than 2 mm during the hold.

These visual tests define a cold-hardware reproduction protocol. The recorded trials used the simulator's contact sensors. Agreement between video proxies and instrumented physical force measurements has not been tested.

## Asset provenance

The runtime loads the I2RT Robotics YAM model, converted from URDF/meshes to USD by the YAM Arena integration. The model's bundled `SOURCE_COMMIT` identifies upstream revision `120c3c81400171174604e503943f8d1ebc891058` and `LICENSE.i2rt` is MIT, copyright I2RT Robotics. The integration's README lists no license for the integration itself. The task folder references the installed runtime and does not redistribute the robot model or integration. Oven, rack, peel and food geometry are built by the task scripts. [I2RT model source](https://github.com/i2rt-robotics/i2rt/tree/120c3c81400171174604e503943f8d1ebc891058) · [YAM Arena integration](https://github.com/kairosiann/yam-isaac-lab-arena/tree/0059d02a1a91bc6834c1d0846022891df0c78730).

## Running

Use an RTX-capable Isaac Sim machine with the IsaacLab-Arena Python environment, the YAM USD asset and the YAM integration installed. The tested Arena revision is `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, its Isaac Lab submodule is `91a0906f93802eb859192dcb0bca879f891e92d6`, and the YAM integration is `0059d02a1a91bc6834c1d0846022891df0c78730`. The renderer uses ImageIO, FFmpeg and Pillow. SciPy supplies the rigid-transform and IK calculations.

Run inside this folder. The scene has one fixed layout. The public trial allows 5,000 control actions at 20 Hz (250 simulated seconds). GPU wall time depends on rendering and hardware.

```bash
mkdir -p results
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python demo.py --video --camera-stride 4 --out results/yam.json
python demo.py --arena --video --camera-stride 4 --out results/policy-controls.json
```

The first command drives the simulated joints with gravity and known-payload torque compensation. The second sends 14-dimensional absolute joint/gripper targets to the public environment and folds payload compensation into the position targets. Both read physical contacts and poses. The controller never moves the food or peel by writing a pose. Each run writes the final grade, control samples, and an H.264 recording beside the JSON. Camera recordings use simulation time.

For Inspect Robots, register `native-runtime.json` with the RoboCurve task-local runtime loader and load `task.py` through `PizzaEmbodiment`. Use `control_hz=20` and `object_jitter=0`. The fixed overhead camera sees the oven and tool rack. The two wrist cameras follow the grippers. The adapter's reset settles an open right gripper above the handle and an open left gripper at READY. Every episode resets the crust joints' bodies, the core, peel and scorer. An idle policy earns zero. The scripted policy can access full simulation state. A Dream score must come from a separate rollout using the policy camera views and joint observations.

Run `python check_inspect.py` for two fixed-scene resets, three RGB views and a 14-D idle-policy check without a model call. Run `python -m unittest -v test_parking` for the rear-support, robot-contact and shaft-motion scoring regressions.

## Validation

Run `python -m unittest -v test_parking test_food_contact` for scoring regressions. Run `python check_shaft_rub.py --out results/shaft.json` for the separate native contact control. Its fixture moves the shaft across the upper surface of the freely simulated pizza and supplies prior grasp as a test precondition. The public YAM task runs with the ordinary free peel and measured grasp history.

## Recorded results

| Recording or check | Grader | Result |
| --- | --- | --- |
| Dream-1 live rollout shown in the board model video | Previous whole-peel contact rule | 0.0, 5,000 actions, step limit |
| Offline replay of those exact Dream-1 actions | Current side/underside rule | 0.0, 5,000 actions, seed 1415158438 |
| Scripted YAM trial shown in the board demo video | Current rule | 1.0, 3,267 public 14-D actions |
| Inverted shaft rubbing the top of the free crust | Current rule | 0.0 despite 119.36° of rotation |

The YAM demo turned the pizza 182.52° and received credit for 178.51°. Maximum center travel was 18.77 mm. Minimum deck load was 0.243 N. The released peel remained supported for 5.233 s with 0.0003 mm maximum measured drift during that interval.

The board model video comes directly from the original Dream-1 episode. Controller, physics scene, action interface and instruction are unchanged. The current model score was measured by replaying the archived actions in the revised environment. Dream-1 was called only for the original rollout. The replay had no access to the reference controller. `evidence/scores.json` records both results and the source hashes. Reproduce the action replay with Inspect Robots 1.0.0, inspect-robots-yam 0.38.0 and the Arena environment above:

```bash
python replay_actions.py --out results/dream-action-replay.json
```
