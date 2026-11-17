# DRG 80 thread and lock qualification experiments

These are component experiments for the 90-degree opening-stop benchmark, recorded on the production Brev L40S. The whole workcell still requires a physical YAM demonstration and end-to-end grading.

## Reproduce

Use the run's pinned IsaacLab-Arena Python (Kit 110.3). Fetch the two M8 OBJ files named in `input/nvidia-factory-reference/reference-manifest.json` into that directory. Verify their listed SHA-256 hashes. The original Factory acknowledgments and license accompany the references.

From this directory run each direction from a fresh process, for example:

```sh
python task/loaded1mm.py --device cuda:0 --direction positive --output measurements/my-positive.json --trace measurements/my-positive-trace.json
python task/loaded1mm.py --device cuda:0 --direction negative --output measurements/my-negative.json --trace measurements/my-negative-trace.json
python task/dual_lock.py --device cuda:0 --fixture task --velocity-iters 8 --cover elastic --hz 240 --output measurements/my-dual-lock.json
python task/check_score.py
```

`loaded36.py` uses the original 1.25 mm pitch and 2.5 mm axial offset. `loaded1mm.py` scales both contact geometries axially by 0.8 and starts at a 4 mm offset. Both use +36 degrees. The geometry cook can warn about the source mesh topology. Each result records actual native measurements. The four independent loaded runs used 4,080 steps each. Full per-contact traces remain on Mac and Brev. Their verified hashes and sizes are in `measurements/verified-thread-results.json`.

The two-thread rig uses a moving screw, frame thread, upper locknut and compliant seat. The refined layout removes both initial overlaps. The 240 Hz run kept at least 6.22 N on the seat after releasing the tools. It still exceeded the specified screw rotation and axial speed limits under a 2 N proof load. This rig applies instrumented torque and force directly for diagnosis. Its score is separate from any YAM task score. The Python scorer checks use synthetic input and exercise both rates independently.

At 480 Hz the same geometry held at least 7.36 N after tool release. With the subsequent 2 N proof load, screw axial peak-to-peak motion was 0.61 µm, but the maximum substep speed was 0.253 mm/s. At 240 Hz those numbers were 1.84 µm and 0.318 mm/s. Both exceed the 0.010 mm/s motion threshold. These are component proof loads. Reading a complete robot task would also require a measured piston effort and real jaw and tool contacts. The compliant cover in these comparisons is an experimental support spring. These measurements do not establish preload for a rigid-cover workcell.

`task/inspect_lock_trace.py` replays the original trace using adjacent observations. `task/check_contact_damage.py` checks split manifolds, the allowed one-step contact gap, latch persistence and both native rates using synthetic examples.

The `tendon_lock.py` variant attaches two elastic helices to native prismatic/revolute DOFs. It uses a rigid cover and one serial articulation for the screw and nut. The angular gearing is authored per degree and the torque coefficient per radian. See the [NVIDIA schema documentation](https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/population/PhysxTendonAxisAPI.html). At 240 Hz it held at least 7.397 N with external tool torque removed. Its proof-load peaks were 0.01037 rad/s and 0.07029 mm/s. Both exceed the frozen limits.

Putting the lower thread and four rigid seat pads on the same structural body reduced the screw's axial peak speed further. In `shared_frame_lock.py` the 240 Hz run meets 13 of 14 checks, including both angular-rate limits. Its maximum axial speed was 0.01541 mm/s against the 0.0100 mm/s requirement. An experimental simplified-core trial had a repeated trace-index error and is excluded from replay comparisons.

The piston trial uses the 30 g representative piston, its rod and head, a separate 5th native DOF, an actuator force read from the physics engine and a piston-to-screw contact sensor. The mesh-only trial hit the piston travel limit and failed. A convex end land (radius 3.2 mm, extension 0.60 mm) eliminated that false stop. The corrected 240 Hz run sustains at least 1.989 N of piston/screw contact with a 2 N input, far from the travel limit. After release the nut/cover force is at least 7.288 N. The powered axial-speed check still fails on nine native substeps during pressure rise. Run `task/cap_piston_lock.py` with the same arguments as `dual_lock.py`, selecting `--cover rigid`.

Offline YAM IK reaches the rack pregrasp, grasp and lift points for the left driver and right wrench. The best maximum positional errors are 0.0704 and 0.0499 mm respectively. These are individual kinematic calculations. They do not test physical grasps, simultaneous arm collisions or task success.
