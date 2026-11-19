# Lift the grounded trailer foot, shift, and stow the crank

Use the instruction and reset layout in task.json. O*NET lists operating trailer landing gear as a core task for heavy and tractor-trailer truck drivers. BLS reports about 2.2 million U.S. jobs in this occupation in 2025. [BLS occupational outlook](https://www.bls.gov/ooh/transportation-and-material-moving/heavy-and-tractor-trailer-truck-drivers.htm)

The short-stroke test fixture retains the crank radius and two screw leads. It reproduces the order of load release, gear change and crank storage. [O*NET 53-3032.00](https://www.onetonline.org/link/details/53-3032.00)

## Run and score

Use seed 0 with the station cameras and the pinned YAM Isaac Sim 6.1.0 image. The CPU PhysX solver advances at 240 Hz and the control loop at 20 Hz. Episodes last up to 1,400 actions, or 70 simulated seconds. Reset and settle for 30 control frames. The successful scripted episode ended at 830 actions (41.50 simulated seconds). On the reference workstation, its measured loop time was about 150 seconds, plus application startup. From this task folder, build the image with `python3 runtime/build_image.py` if needed.

```sh
GPU=0
OUT="$(mktemp -d)"
CACHE="$(mktemp -d)"
mkdir -p "$CACHE/tmp"
docker run --rm --gpus "device=$GPU" --shm-size 8g --user "$(id -u):$(id -g)" --group-add 1234 -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v "$CACHE:/tmp/task-home" -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp -e NVIDIA_DRIVER_CAPABILITIES=all -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD:/task:ro" -v "$OUT:/outputs" -w /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none --kit_args '--portable-root /tmp/task-home/kit --/renderer/multiGpu/enabled=false'
```


The native runtime task ID is `load-coupled-trailer-landing-gear`. From the Docker command above, a Dream evaluation uses the same seed, cameras, physics and action limit. Replace `--policy scripted` with `--policy dream --budget-path /dream-budget/account-reservation.json`, mount an owner-readable credential file at `/run/secrets/dream.env:ro`, set `DREAM_CREDENTIALS=/run/secrets/dream.env`, and mount your writable, authorized episode reservation directory at `/dream-budget`. The credential file contains DREAM_API_KEY. The registry receipt in this runtime expects the native task ID, an account run ID, a USD 20 episode reservation and cumulative reserved and authorized amounts from the operator's existing account ledger. The runner journals each model request in that directory. An operator using another budget backend can provide its Ledger implementation to the Dream runner.

Each episode writes a result, measured trajectory and camera recordings. Inside the same image, run `python runtime/rescore.py --task ./task.py path/to/trajectory.jsonl --expect path/to/result.json` to check the recorded terminal metrics.

## Reset and measurements

All positions are in metres. The robot base midpoint is (0, 0, 0.750). Mount the gearbox at (0.335048, −0.400598, 0.910) with yaw 95°. Place the support plate at (0.388213, −1.008277, −0.0025), with its top at 0.0075. The keeper reference is (0.136371, −0.446087, 0.945598). Set the crank to 15° and the selector to zero before settling. The 40 kg effective carriage is gravity compensated. A 28 µm screw preload acts through 1.2 MN/m of stiffness and 4,800 N·s/m of damping. The first scored YAM observation recorded 33.55 N from the vertical foot–plate contact impulse. A 16 ms filtered normal force, friction coefficient 0.35 and effective radius of 55 mm produce a load-dependent torque on the spindle. A component test over the same nine angles gave mean resisting torques of 1.974 N·m under load and 1.564 N·m in air. These are simulator measurements.

## What the scorer checks

1. At reset the foot must have plate contact, a measured force above 0.005 N, and an absolute gap under 0.02 mm. While gripping in low, advance at least 15° from a grounded grip sample and hold a gap of at least 0.035 mm with zero ground contacts for three observations.
2. After the qualified lift, maintain grip and clearance through the axial shift. High engages at 22.4 mm. Leaving the low range at 5.6 mm before qualified lift, or while touching the plate, permanently invalidates the episode.
3. Accumulate at least 40° of net high-gear turn over consecutive gripped frame pairs, and reach the end of the short stroke within 0.045 mm (about 0.9503 mm clearance).
4. Fold the crank at least 85° into the keeper, engage the catch within 20 mm, keep the foot retracted and each tool center over 47 mm clear of the grip with no hand contact. Hold for 61 consecutive observations (3.00 seconds between first and last).

Both finger pads on each gripper normally contact the grip. An established grasp can bridge a single observation with missing contacts if both tool centers stay within 12 mm, the measured finger displacements stay within 17 mm, and the hands have not cleared the crank. The contact exception cannot establish the original grip. In the seed-0 control, shifting with 33.64 N still on the plate gave zero throughout. Masking one pad for one frame at 11.90 s preserved all 830 native YAM scoring observations.

The initial contact release is submillimetre. The station camera mainly shows arm motion; a side-view macro camera and displacement telemetry were used for the separate physical check. The model and the primary recorded YAM demo use the standard policy cameras.

## Recorded model trial

Dream-1 scored 0.0 after 1,400 actions (70.00 simulated seconds, step_limit). The measured control loop took 286.7 seconds, plus simulator startup. Seed 0 was used for both policies. Dream received the top and wrist RGB views, joint/gripper state and the task instruction. The overview RGB is the external recording camera.

During the model trial, no pad-to-grip or other monitored gripper-to-crank contact was recorded. The shaft nevertheless advanced to 23.76° of recorded low-gear travel and the foot reached 0.02454 mm clearance, below the 0.035 mm threshold. No grasp-qualified lift or axial shift was credited. The cause of this uncommanded fixture motion has not been isolated.
