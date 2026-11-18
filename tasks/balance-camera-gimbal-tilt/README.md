# Balance a powered-off camera gimbal

Support the camera, squeeze the orange spring-brake tab toward its fixed handle, slide the camera fore or aft, and release the brake. Set the free tilt cradle near each green mark and withdraw both hands for three seconds at each mark. Both verifications use the same secured rail setting.

## Setup

The fixture stands on the standard bimanual YAM station. Its base is 220 × 210 × 16 mm. The base reference is (0.35, 0, 0.7475) m in station coordinates. The tilt bearing is 145 mm above that reference, at (0.35, 0, 0.8925) m. The horizontal tilt axis is local Y, with travel from -55° to +55°. Green marks indicate -30° and +30°.

The 0.62 kg camera/lens payload rides on a 160 × 28 × 10 mm plate with -48 to +40 mm rail travel. Its mass centre starts 19 mm forward of the payload frame. Its side handgrip is 38 × 20 × 46 mm. The cradle has a separate teal tilt handle. Squeezing the orange tab releases a spring-loaded shoe from the rail. Releasing it presses the shoe against the plate through native friction.

Camera housing and handles are rigid polymer-like compound solids. Rail, cradle, stand and shoe represent rigid metal components. Contact friction is 1.1 static and 0.85 dynamic, with zero restitution. The shoe spring is 600 N/m with a -30 mm rest position. Camera height and other axes are pre-adjusted. Dimensions, masses and inertia are in assets/tilt-gimbal/gimbal.xml and build_asset.py.

This is a single-axis prototype with a custom squeeze-release rail brake. Physical hardware has not been calibrated. Native joints, gravity, spring force and frictional contacts drive the motion.

## Score and reset

Move the rail at least 8 mm while the brake is open. Each hands-off test starts within 2° of its ±30° mark at no more than 5°/s. During three seconds, remain within 3° of the release angle and mark, with rail displacement within 0.5 mm. One direction earns 0.5 and both earn 1.0. Reopening the brake or moving the secured rail more than 0.5 mm invalidates prior verification.

Both TCPs must be over 145 mm from the tilt pivot. Robot-to-fixture contact impulse must be below 0.0001 N·s per control sample. The brake is closed below 0.8 mm displacement magnitude and open above 3 mm. Measurements come from native poses, velocities and contact reports. In a physical trial, use a protractor and video for the angle/time windows, a ruler or caliper for rail travel, and clear front/side views showing every robot link away from the fixture throughout each hold. The contact-impulse threshold is a simulator no-contact check; a physical trial uses visibly separated hands and links, without requiring an impulse sensor.

The limit is 2,400 commands at 20 Hz, or 120 simulated seconds. Use seed 0 and a fresh single-episode environment. The full-score Isaac Sim rollout took 1,203 commands, or 60.15 simulated seconds, including a repeated adjustment within the same episode.

## Run and reproduce

Use Isaac Sim 6.1.0 with pinned Isaac Lab/Arena, YAM and Dream dependencies in runtime/Dockerfile. Validation used yam-inspect-isaac:6.1.0. The PhysX solver runs on CPU and RTX cameras use the selected GPU.

If the image is unavailable, build its pinned dependencies with the following command. A rebuild on another machine has not been separately validated.

    python runtime/build_image.py --image yam-inspect-isaac:6.1.0

From this task directory, replace N with an available GPU index.

    mkdir -p outputs
    docker run --rm --gpus device=N --shm-size 8g -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD:/task:ro" -v "$PWD/outputs:/outputs" -w /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none --kit_args '--/renderer/multiGpu/enabled=false'

Outputs include four H.264 camera videos, initial/final frames, trajectory.jsonl, frozen task source and result.json. First use may take several minutes for shader compilation. Simulator wall time depends on hardware.

Replay with the exact frozen source in an environment containing yam_arena and NumPy.

    python runtime/rescore.py --task /path/to/frozen/task.py /path/to/episode-01/trajectory.jsonl --expect /path/to/episode-01/result.json

Dream-1 uses the same runtime with --policy dream, an authorized DREAM_API_KEY through a protected environment file, and an existing persistent budget supplied with --budget-path. The code package contains no credentials or account ledger. Dream receives standard top/wrist RGB views and measured robot joints/grippers. Native object state is used by scoring and the scripted demo. The runner may continue after a server session replacement without resetting the physical scene, observations or step count: at most eight replacements per trial and one consecutive replacement per action request, with every request reserved against the configured budget. Other errors stop and preserve the trial. The recording and result always report the actual completion and recovery count.

## Source and assets

task.py creates the scene, observer.py reads native bodies and contacts, scoring.py computes the score, and controller.py sends ordinary YAM joint/gripper targets. YAM's action term already supplies gravity compensation. An additional gravity-derived position offset would displace the grasp.

Fixture geometry consists of original compound primitives. See assets/tilt-gimbal/SOURCE.md. Optional MuJoCo component checks are gimbal.xml, probe.py and record_controls.py. The submitted demonstration and score use Isaac Sim.
