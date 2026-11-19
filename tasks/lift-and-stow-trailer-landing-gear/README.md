# Raise the landing foot, make a half turn in high, and park the crank

Lift the grounded foot in low. Shift the same crank into high, turn it half a revolution, then fold it into its keeper and let go.

Heavy and tractor-trailer truck drivers raise and lower landing gear as part of their core work. BLS reports about 2.2 million U.S. jobs in this occupation in 2025. [O*NET occupation 53-3032.00](https://www.onetonline.org/link/details/53-3032.00) · [BLS occupational outlook](https://www.bls.gov/ooh/transportation-and-material-moving/heavy-and-tractor-trailer-truck-drivers.htm)

In the JOST A450 coupling procedure, the operator raises the feet in low until ground contact ends, then selects high, completes retraction and secures the crank. The bench preserves both screw leads and shortens the working stroke. [Operating manual, p. 2](https://jostinternational.com/hubfs/LT-LG450-04%20-%2010%20YEAR%20MAGNUM%20OPERATING%20%26%20MAINTENANCE.pdf?hsLang=en) · [A450 gearing](https://jostinternational.com/landing-gear-a450-series)

## Setup

Two YAM arms with parallel grippers stand on a side bench and a separate plinth. A rigid gearbox carries a 355.6 mm-radius folding crank, a 28 mm sliding selector and a steel-like 230 × 210 × 26 mm landing shoe. The shoe starts on a fixed 380 × 380 × 20 mm plate, the selector is in low, and the crank starts deployed at 15°. The keeper sits at the opposite end of the 180° high-gear sweep. Dimensions and reset references are in the README and USD asset. Station coordinates are in metres. The gearbox is at (0.335048, -0.400598, 0.910000), yaw 95°. The plate center is at (0.388213, −1.008277, −0.0025), with its top at 0.0075. The keeper reference is (0.504776, -0.413856, 1.027583). The side bench measures 830 × 415 × 60 mm and the separate arm plinth 170 × 150 × 760 mm. The unloaded crank sweeps through the open aisle. Reset to 15° in low and settle for 30 control frames.

The gravity-compensated carriage has a 40 kg effective mass. Its screw preload is 28 µm with 1.2 MN/m stiffness and 4,800 N·s/m damping. A 16 ms filtered contact load supplies the modeled thrust friction. Rigid counterbalanced bench mechanism with 0.79375 and 7.25714 mm of screw travel per revolution in low and high. Measured foot support increases spindle resistance. Crank damping is 0.97 N·m·s/rad. The 3.72 mm nominal stroke represents a shortened test rig. The catch uses a detent force model. Full vehicle loads, road clearance and physical hardware performance remain unvalidated.

## Scoring

Four ordered milestones worth 0.25 each. Start with the shoe loaded on the plate. Grip with both hands and advance at least 15° in low from a grounded grip sample. Maintain at least 0.035 mm clearance with no ground contact for three observations. While gripping and clear, push inward to at least 22.4 mm. Leaving the low range at 5.6 mm before a qualified lift permanently scores zero. In high, accumulate at least 180° net rotation over consecutive gripped frame pairs and reach at least 3.6718 mm clearance (rounded). Fold the crank at least 85°, engage the keeper within 20 mm, remain fully retracted and hold with both hands clear for 3.00 s. See README for the grip sensor tolerance. The first scored observation requires plate contact, force above 0.005 N and a gap magnitude below 0.02 mm. A grip normally requires contacts on both finger pads of both grippers during the observation. An established grip may bridge one missing observation if both tool centers stay within 12 mm and all measured finger displacements are within 17 mm. Positive and negative gripped increments both count toward the net high-gear angle. While parked both tool centers must be over 47 mm clear of the grip with no monitored robot contact. The 61 parked samples span 3.00 s at 20 Hz.

For a physical bench trial, a dial indicator on the shoe with 0.01 mm or finer resolution and a contact indicator on the plate can distinguish takeoff from compression. Mark the spindle and the two selector positions and record the crank and hand release on video. To reproduce the exact rule on an instrumented bench, zero the shoe indicator at loaded contact, synchronize a contact switch and four pad-contact indicators to the video at 20 Hz, and use encoder marks at 15° and 180° and selector marks at 5.6 and 22.4 mm. Each pad may register anywhere within one 50 ms bin. Read the parked crank against an 85° template and keep a 20 mm keeper gauge and a 47 mm hand-clearance reference in view. The supported reset force is about 33 N in simulation. A physical bench trial has not been performed.

## Asset sources

The landing gear, shoe, ground plate, keeper, side bench and arm plinth are original task geometry. The task license covers the original material in this folder. The standard YAM embodiment and room are provided at run time by the pinned arena dependency. The YAM URDF and robot meshes originate with I2RT under the MIT license. The arena repository documents the robot asset attribution and lists no project-wide license of its own. Its source is downloaded during the optional image build. [Pinned arena README](https://github.com/kairosiann/yam-isaac-lab-arena/blob/0059d02a1a91bc6834c1d0846022891df0c78730/README.md#license) · [Robot asset license](https://github.com/kairosiann/yam-isaac-lab-arena/blob/0059d02a1a91bc6834c1d0846022891df0c78730/yam_arena/assets/yam/LICENSE.i2rt) · [I2RT robot source](https://github.com/i2rt-robotics/i2rt)

## Run and rescore

Seed 0, Isaac Sim 6.1.0, physics at 240 Hz and control at 20 Hz. Each trial allows 1,400 actions (70 simulated seconds). Build the pinned runtime with `python3 runtime/build_image.py` or use the existing image. Run from this task directory.

```sh
GPU=0
OUT="$(mktemp -d)"
CACHE="$(mktemp -d)"
mkdir -p "$CACHE/tmp"
docker run --rm --gpus "device=$GPU" --shm-size 8g --user "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v "$CACHE:/tmp/task-home" -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp -e NVIDIA_DRIVER_CAPABILITIES=all -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD:/task:ro" -v "$OUT:/outputs" -w /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none --kit_args '--portable-root /tmp/task-home/kit --/renderer/multiGpu/enabled=false'
```

For Dream-1 use the same image with `--policy dream --budget-path /dream-budget/account-reservation.json`. Mount your owner-readable Dream credential file read-only at `/run/secrets/dream.env`, set `DREAM_CREDENTIALS=/run/secrets/dream.env` and mount the writable directory of your approved account reservation at `/dream-budget`. Its native task ID is `half-turn-trailer-wrist`. The runtime records each SDK request against that reservation. From this directory inside the image run `python runtime/rescore.py --task ./task.py path/to/trajectory.jsonl --expect path/to/result.json`.

Both policies use seed 0, the same instruction, physics and scoring code. Dream receives top and wrist RGB and measured robot state. The external overview camera records both runs.

## Recorded trials

Dream-1 scored 0.00 after 1,400 actions (70.00 simulated seconds, step_limit). Scripted YAM scored 1.00 after 1,188 actions (59.40 simulated seconds). The timed control loops took 240.7 s and 239.5 s respectively, plus simulator startup.
