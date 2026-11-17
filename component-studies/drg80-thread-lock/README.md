# DRG 80 thread and lock qualification experiments

These are component experiments for the 90-degree opening-stop benchmark, recorded on the production Brev L40S. The whole workcell still requires a physical YAM demonstration and end-to-end grading.

## Bidirectional relative-thread qualification

On the replacement Crusoe L40S, changing only the old fixed tendon's joint limit from four revolutions to two changed the spurious extension from 3.90 mm to 1.90 mm. Two independent implicit drives restored the expected displacement but transmitted only 3.12 N at 240 Hz. The corrected solver computes one shared thread load and applies conjugate axial force and relative torque. Native contacts determine both poses and transmit the measured load. Physical face separations enter the local implicit force predictor. No state, stop position or score is written.

Both drive directions passed at 240 and 480 Hz under 0.005 N·m with the nut rotating relative to a freely rotating screw. The controller inferred angular loading from native motion. Measured elastic extension was 0.314 mm and the physical plate reaction was 31.4 N. Halving the torque and increasing the plate gap to 0.25 mm gave 0.157 mm and 15.7 N. Each case also checked unloading, world-transform agreement and clearance from the rotary limit. These isolated tests omit thread friction to measure the normal constitutive law. The original uploaded fixed-tendon results below remain historical.

Run `python task/thread_coupled_probe.py --mechanism coupled-effort --observed-load --hz 240 --output measurements/verify240.json` from this directory with the pinned Isaac environment. Use 480 for the refinement or add `--magnitude .0025 --gap .00025` for the independent load case. [Native results and trace hashes](measurements/thread-balance-recovery.json).

## Approved powered-motion revision

The current contract measures the maximum minus minimum over each contiguous interval above 1 N of absolute native piston effort. Include the previous sample at entry. A span above 1 µm along the screw axis or above 0.01° about either the screw or nut axis latches damage until reset. The instantaneous-rate results documented below describe earlier experiments. The approved revision changes one damage rule and its matching negative control. The prior review receipt applies to its original digest.

Reproduce the native comparison in separate Isaac processes from this directory. Both cases use identical initial geometry, torque and pressure schedules. Only the fault case adds a 0.03 N·m driver torque for 0.25 s under pressure. All other checks are preserved.

```sh
for hz in 240 480; do
  for item in normal fault; do
    mode=normal
    if [ "$item" = fault ]; then mode=powered-adjustment; fi
    python task/approved_piston.py --device cuda:0 --fixture task --cover rigid --velocity-iters 8 --hz "$hz" --case "$mode" --output "measurements/dual-rev${item}${hz}.json"
  done
done
python task/check_energized_motion.py
python task/check_score.py
python task/verify_powered_series.py measurements
```

The public YAM task loader can exercise the actual raw-sample callback using `task/check_yam_grade_entry.py PATH/TO/task.py`. Its inputs are synthetic. Run `task/yam_rack_lift.py --workspace WORKSPACE --output measurements/yam-rack-lift.json` in the pinned Arena environment to replay the independent physical pickup using the two free USD tools, their rack supports and joint commands to both arms. The workspace must include `input/fixture-geometry-contract.json` and `task/assets/tools/`. The held lift is measured against the settled pose throughout the final 1.5 s. This stage earns no adjustment or locking credit.

### Fresh Brev results

| Case | Physics Hz | Screw axial span (µm) | Screw angle span (°) | Nut angle span (°) | Minimum seat force (N) | Powered hazard |
|---|---:|---:|---:|---:|---:|---|
| normal | 240 | 0.184 | 0.00267 | 0.00100 | 6.526 | clear, 17/17 checks |
| powered-adjustment | 240 | 228.740 | 83.26805 | 92.71708 | 0.000 | latched on step 3181 |
| normal | 480 | 0.172 | 0.00194 | 0.00054 | 6.604 | clear, 17/17 checks |
| powered-adjustment | 480 | 239.089 | 86.61138 | 138.77968 | 0.000 | latched on step 6361 |

The normal and injected traces at each rate were identical until the first fault torque. The damage flag latched on that first physical substep at both rates. The full per-substep traces are preserved on Mac and Brev. Their SHA-256 hashes are in [the replays](measurements/powered-motion-native-series.json).

[Recorded YAM acquisition](measurements/yam-rack-camera.mp4) contains the overhead and two wrist views from a 13 s native episode. Minimum lift throughout the 1.5 s hold was 48.1 mm for the driver and 48.9 mm for the wrench. A second episode with the fingers held open and otherwise identical arm commands left both tools on their racks. This recording ends after acquisition. No full-workcell success, Dream score or submission is recorded.

The separate relative-N73 load test still blocks full workcell qualification. Explicitly supplying both actor relationships to the rack-and-pinion joint produced the native error `not allowed to add a non-GPU compatible PxConstraint` in direct-GPU mode. Axial extension and load plate force remained zero. The alternate native fixed-tendon diagnostic produced axial travel in both directions, but hit the rotary limits and reached about 3.90 mm of calculated elastic extension against an expected static magnitude of 0.314 mm. The alternate gearing trial also exceeded the prescribed N73 force envelope. The locked-piston and pickup tests do not qualify this separate load law. Keep the complete physical tests before any full-score claim. The USD gearing and force coefficient units used in the primary tendon trial follow the [NVIDIA schema](https://nvidia-omniverse.github.io/PhysX/ovphysx/latest/population/PhysxTendonAxisAPI.html).

## Reproduce the earlier component experiments

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

Repeating the piston test at 480 Hz retained the stop face and clamp but exceeded both peak screw speed limits. Tightening to 0.020 N·m at 240 Hz maintained 14.818 N after tool removal, with 0.0671 µm peak-to-peak axial motion across the proof phase. Its 0.01075 mm/s instantaneous axial maximum still exceeds 0.01000 mm/s.

For a separate powered-adjustment negative control, the same 240 Hz assembly received an actual finite 0.030 N·m external driver torque for 0.25 s starting at 13.25 s. Screw travel rose from 0.202 µm to 228.76 µm and the seat force fell to zero. The official frozen score has not changed. The file `motion-resolution-replay.json` evaluates a possible alternative on existing traces only, with no certification or YAM result.

The final independent counterhold variant uses a native implicit, bounded rotational tool drive during tightening. Its native maximum effort was verified as zero after releasing the driver. It retained stop contact and 5.86 N on the cover, but the powered axial speed reached 0.01319 mm/s at 240 Hz. This also remains a component result.
