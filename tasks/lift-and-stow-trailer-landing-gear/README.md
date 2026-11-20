# Retract a grounded trailer landing leg and secure the crank

Raise the foot in low gear until it is clear of the ground. Push the same crank into high, wind the leg fully up to its travel stop, and stow the crank in its keeper. Withdraw both hands.

The trial starts after coupling, with 305 mm left to the fully retracted stop. Raising and lowering trailer landing gear is an O*NET task for heavy and tractor-trailer truck drivers. BLS reports the occupation's size. Frequency of this particular operation per driver was not measured. JOST specifies low gear until the feet leave the ground, followed by full retraction in high and crank storage. [O*NET tasks](https://www.onetonline.org/link/details/53-3032.00) · [BLS occupation](https://www.bls.gov/ooh/transportation-and-material-moving/heavy-and-tractor-trailer-truck-drivers.htm) · [A400 operating manual, steps 6–7](https://jostinternational.com/hubfs/Jost_International_April_2024/Pdf/LT-LG400-13-Inspection-Maintenance-Procedures.pdf?hsLang=en#page=2)

## Physical reference and reset

1. Source a JOST A400.T.19 outside-mount set (19 in available stroke), mating cross shaft, original folding crank and hanger. Specify the 14 in (355.6 mm) throw C10592.2 crank when ordering and confirm which accessories the dealer includes. TruckPro lists the leg set at US$1,891.17. Mounting, accessories and robot are extra. An existing correctly installed set of the same model can be used. The A400 ratios are 32 turns/in in low and 3.5 turns/in in high, giving 0.794 and 7.257 mm per revolution. [TruckPro JXA400.T.19](https://www.truckpro.com/p/jost-leg-set-A400.T.19/jxa400.t.19) · [JOST ratios](https://jostinternational.com/landing-gear-a400-series) · [Crank catalog](https://jostinternational.com/hubfs/Jost_International_April_2024/Pdf/Drop-Leg-Catalog.pdf?hsLang=en#page=18)
2. Use the aligned, cross-shaft-connected and braced set on an immobilized training trailer supported by its coupled tractor. Follow the manufacturer's installation, chocking and brake procedures. Keep the crank sweep clear. Both shoes must unload before shifting. For a shop reproduction, support the housing independently of the test pads throughout. [Installation and bracing guide](https://jostinternational.com/hubfs/Jost_International_April_2024/Pdf/LT-LG01-10-Landing-Gear-Installation-Guide.pdf?hsLang=en) · [Operator manual](https://jostinternational.com/hubfs/Jost_International_April_2024/Pdf/LT-LG400-13-Inspection-Maintenance-Procedures.pdf?hsLang=en)
3. With the mechanism fully up, mark the moving tube at the fixed housing edge and place a second witness mark 5 mm away. Set both legs 305 mm below this datum. Pull the selector into low and mark both seated selector positions. Under each shoe use a rigid level pad on an adjustable platform. Tare a low-profile platform scale before adding foot contact. Raise each platform until the shoe rests on it and the reading is approximately 3.4 kg (33 N, target 3–4 kg); lock the platform and leave the scale in place. The pads carry only this light preload. Record both reset marks, scale readings, and the clear distance to the travel stop. Retare and reset before each trial.
4. Record side views of the shoe and selector and an unobstructed view of the hanger, sleeve and hands on a common clock. Keep a ruler beside the shoe, a 3 mm reference at pad height and a 50 mm reference beside the grip. Establish the folded reference by seating the crank in the empty hanger. On hardware, use the operator's two-handed grip prescribed by JOST. The simulated right YAM gripper drives the loaded sleeve while the left stays withdrawn. Whether a physical YAM can supply the required torque on stock hardware remains untested.

## Score a trial

Read the shoe against the fixed pad and the retracted witness mark, the selector against its detents and the sleeve against the hanger. The endpoint tolerance is 5 mm on a nominal 305 mm stroke. A shaft backing up slightly during folding is acceptable while the shoe stays in the marked zone.

| Score | Observable result |
| --- | --- |
| 0 | The foot starts off the pad, the selector leaves low before the foot clears the pad by 3 mm, or high gear fails to finish retraction. |
| 0.5 | From a grounded start in low, lift the shoe clear, push the same selector inward, and crank in high until the shoe reaches the final 5 mm band. Keep the shoe in the band. |
| 1 | Also fold the crank into the hanger, withdraw both hands, and leave the sleeve seated and the shoe in the band for 3 continuous seconds before 20 minutes elapse. |

The software accepts a grounded reset within 2 mm of pad level. Selector marks are the outer and inner fifths of the 28 mm stroke (5.6 and 22.4 mm). High must be engaged by 295 mm or earlier and carry the foot into the band. The folded position is at least 80° from deployed, the sleeve reference is within 20 mm of its seated reference, and both gripper centers are at least 50 mm from the sleeve. Use the crease of the hanger, angle template, ruler and video time stamps to make these checks on a bench. A gap in consecutive samples restarts the three-second clock. The score uses positions and elapsed time. Internal contact impulses enter the simulated drivetrain dynamics only.

## Full-stroke Isaac Sim demonstration

At seed 0 the shoe cleared 3 mm at 76.40 s, the inward shift seated at 92.40 s, and the shaft made 41.52 high-gear turns to the stop at 855.35 s. The shoe moved 304.97 mm. The crank caught at 862.45 s, and both hands were clear from 865.00 through 868.00 s. The measured control loop took 53.3 minutes of wall time on Blackwell. The original full-length overview is 17,360 frames at 20 Hz. Replaying the measured states with the revised scorer gives **1.0 at 868.00 s**. Retraction first earned 0.5 at 844.35 s. The physics, action sequence and camera footage for this replay are unchanged.

Scorer controls on the recorded positions gave 0 for shifting before lift, starting airborne, capping lift at 295 mm or engaging high only after entering the band. Blocking the fold, keeping a hand at the sleeve or briefly interrupting the final release each gave 0.5. These are counterfactual scoring checks on a measured trajectory.

## Asset sources

The landing gear, shoe, ground plate, keeper, side bench and arm plinth are original task geometry. The task license covers the original material in this folder. The standard YAM embodiment and room are provided at run time by the pinned arena dependency. The YAM URDF and robot meshes originate with I2RT under the MIT license. The arena repository documents the robot asset attribution and lists no project-wide license of its own. Its source is downloaded during the optional image build. [Pinned arena README](https://github.com/kairosiann/yam-isaac-lab-arena/blob/0059d02a1a91bc6834c1d0846022891df0c78730/README.md#license) · [Robot asset license](https://github.com/kairosiann/yam-isaac-lab-arena/blob/0059d02a1a91bc6834c1d0846022891df0c78730/yam_arena/assets/yam/LICENSE.i2rt) · [I2RT robot source](https://github.com/i2rt-robotics/i2rt)

## Reproduce in Isaac Sim

Use seed 0, 240 Hz physics, 20 Hz control and a 24,000-action limit (1,200 simulated seconds). Build the pinned runtime with `python3 runtime/build_image.py` or use the existing image. Run from this task directory.

```sh
GPU=0
OUT="$(mktemp -d)"
CACHE="$(mktemp -d)"
mkdir -p "$CACHE/tmp"
docker run --rm --gpus "device=$GPU" --shm-size 8g --user "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v "$CACHE:/tmp/task-home" -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp -e NVIDIA_DRIVER_CAPABILITIES=all -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD:/task:ro" -v "$OUT:/outputs" -w /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none --kit_args '--portable-root /tmp/task-home/kit --/renderer/multiGpu/enabled=false'
```

For Dream-1 use the same image with `--policy dream --budget-path /dream-budget/account-reservation.json`. Mount your owner-readable Dream credential file read-only at `/run/secrets/dream.env`, set `DREAM_CREDENTIALS=/run/secrets/dream.env` and mount the writable directory of your approved account reservation at `/dream-budget`. Its native task ID is `a400-full-travel`. The runtime records each SDK request against that reservation. From this directory inside the image run `python runtime/rescore.py --task ./task.py path/to/trajectory.jsonl --expect path/to/result.json`.

Both policies receive the same instruction and environment. Dream uses robot camera RGB and measured joint state. The external overview provides the trial video.

## Dream-1 trial

Dream-1 scored 0.0 in the seed-0 Blackwell rerun. It completed after 24,000 control steps (1200.00 simulated seconds, step_limit). The model returned 500 action chunks. The Blackwell YAM demo scored 1.0 on replay with the revised scorer after 17,360 control steps (868.00 simulated seconds). The control loop ran for 71.3 minutes of wall time. The submitted model video shows the actual policy rollout. Its recorded states reproduced the terminal score on all 24,000 samples. The earlier trial stopped after 7,809 steps at a service error. The rerun retained the same seed, policy, physical scene, scorer, and action limit. The client can wait and repeat a retryable service request on the same observation before advancing the simulator.
