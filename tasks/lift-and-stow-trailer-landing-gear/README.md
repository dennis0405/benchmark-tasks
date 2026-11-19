# Lift the grounded trailer foot, shift, and stow the crank

Use the instruction and reset layout in task.json. O*NET lists operating trailer landing gear as a core task for heavy and tractor-trailer truck drivers. BLS reports about 2.2 million U.S. jobs in this occupation in 2025. [BLS occupational outlook](https://www.bls.gov/ooh/transportation-and-material-moving/heavy-and-tractor-trailer-truck-drivers.htm)

The short-stroke test fixture retains the crank radius and two screw leads. It reproduces the order of load release, gear change and crank storage. [O*NET 53-3032.00](https://www.onetonline.org/link/details/53-3032.00)

## Run and score

Use seed 0 with the station cameras and the pinned YAM Isaac Sim 6.1.0 image. The CPU PhysX solver advances at 240 Hz and the control loop at 20 Hz. Episodes last up to 1,400 actions, or 70 simulated seconds. Reset and settle for 30 control frames. The successful scripted episode ended at 830 actions (41.50 simulated seconds). On the reference workstation, its measured loop time was about 145 seconds, plus application startup. From this task folder, build the image with `python3 runtime/build_image.py` if needed.

```sh
GPU=0
OUT="$(mktemp -d)"
CACHE="$(mktemp -d)"
mkdir -p "$CACHE/tmp"
docker run --rm --gpus "device=$GPU" --shm-size 8g --user "$(id -u):$(id -g)" --group-add 1234 -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v "$CACHE:/tmp/task-home" -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp -e NVIDIA_DRIVER_CAPABILITIES=all -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD:/task:ro" -v "$OUT:/outputs" -w /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none --kit_args '--portable-root /tmp/task-home/kit --/renderer/multiGpu/enabled=false'
```

To evaluate Dream-1, select `--policy dream` at seed 0 with the same camera rig, instruction and action limit. Mount authorized credentials at /run/secrets/dream.env, set DREAM_CREDENTIALS to that path and pass the operator's budget path with --budget-path. Use the scripted policy to reproduce the demonstration. The recording directory contains result.json, trajectory.jsonl and camera MP4 files. Replay a saved trajectory in the pinned runtime with `python runtime/rescore.py --task ./task.py episode-01/trajectory.jsonl --expect episode-01/result.json`.

## Measurement conventions

A loaded sample needs contact points, a contact-force reading above 0.005 N, calculated from the sum of contact impulse magnitudes divided by sample time, and an absolute gap below 0.02 mm. An airborne sample needs zero foot–plate contacts and at least 0.035 mm clearance. All four finger-pad contacts must be observed in a control interval to count as a two-hand hold. Selecting high before a qualified lift permanently invalidates the trial. The selector milestone tests four-pad contact at the observation where the high position qualifies. The high-turn milestone tests net high-gear rotation and foot clearance. The keeper uses a near-position and near-angle detent with restoring torque. The held crank must be within 20 mm of the keeper reference, with its catch engaged, and both tool centers farther than 47 mm from the grip with no reported hand contact. The last 61 consecutive observations span three seconds. The macro camera shown separately in the review material is diagnostic. Dream and the primary demo use the standard station policy cameras.

The underlying screw, detent and hinge balancing torques are reduced models. The 0.035 mm lift criterion relies on simulation geometry and contact telemetry. It would require a displacement indicator for physical replication.

The primary Dream episode ended at the step limit after 1,400 actions with score 0.0 (70 simulated seconds, about 280 seconds in the control loop). The early-shift control reached high with the shoe still bearing on the plate and scored 0.0.

## Fixture coordinates for reset

All coordinates are metres in the station world frame. The two robot base centers have their midpoint at (0, 0, 0.750). Mount the gearbox center at (0.335048, −0.400598, 0.910) with yaw 95°. The plate center is (0.388213, −1.008277, −0.0025) and its top surface is at 0.0075. The keeper reference point is (0.136371, −0.446087, 0.945598). Set the crank to 15° and the selector to zero before settling. The 0.17 kg shoe uses a 15 µm model preload offset with spring gain 4,800 N/m. At the first scored sample the native contact reading was 0.0687 N. The solver keeps the low-gear reaction and input-shaft drag separate. A physically built fixture would need an independent displacement indicator and support-force measurement to reproduce these thresholds.
