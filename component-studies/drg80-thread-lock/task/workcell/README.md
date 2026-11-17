# DRG 80 mounted workcell trials

The apparatus uses the fork's first-party gripper, piston, jaws, lever, finger tools and rack. Its screw and N73 have separate rotational and axial coordinates. Each thread produces one pair of conjugate internal forces. The jaw guides share a native PhysX articulation with the piston. The valve applies a finite force. The free tools are manipulated through normal bimanual YAM joint actions.

Run on the pinned Isaac Lab Arena environment with an RTX GPU. From the fork root:

```sh
study=component-studies/drg80-thread-lock
run=automation-results/20261114T031727-dc73f2bb
mkdir -p measurements
python "$study/task/workcell/check_mounted.py" --assets "$run/task-prototype/assets" --out measurements/fixture.json --early --drive-instrument --tightening-travel 2.5
python "$study/task/workcell/audit_workcell.py" measurements/fixture.trace.json --out measurements/fixture.audit.json
python "$study/task/workcell/yam_workcell.py" --workspace "$run" --output measurements/yam.json --video measurements/yam.mp4
```

The first command can apply capped native angular drives to commission the mechanism. It removes their stiffness and damping before release. The robot command spawns a separate workcell with no instrument drives. It records free-tool poses, actual screw rotation, physics ticks and any failed engagement. The MP4 can therefore end partway through the job. Inspect the JSON and raw trace before interpreting any video.

The audit reuses the task's powered-motion rule at the native sample rate. Its three endpoints start after the first open stop, with three intervening closes. A motion violation latches, even when an endpoint angle subsequently returns within tolerance. Additional full-task measurements include damage across the whole room, tool rack containment and robot clearances. These diagnostic commands do not issue a competition grade or submit a task.

For the welded-bearing trial, add `--continuous-seat --native-friction --rise-seconds .16` to either GPU command and `--tightening-travel 2.5` to the fixture command. Exterior bearing geometry stays fixed. A closed SDF boundary removes internal sector faces. Native static and dynamic rotational friction efforts follow the measured thread load. At 240 Hz this configuration maintained three near-identical 90.95° endpoints and at least 29.5 N on release. Absolute powered rotation stayed below 0.00029° on both parts. The maximum 36.77 µm axial change still fails. The subsequent reader additionally selects normal impulses on the actual planar bearing face.

`check_loaded_equilibrium.py` compares an ideal settled pose with the frozen stiffness and measured geometry. Gravity at 90° assists opening by 1.21 N. With the open port above 1 N, the ideal settled stop load exceeds 2.20 N. The original shaft/seat network can balance at most about 1.13 N of load change inside the approved pose spans. This is a static comparison for the current rigid-cover design. The measured transients are larger. Material sensitivity runs are explicitly excluded from benchmark qualification.

For the revised reversible-deflection criterion, run the same welded-bearing commands at 240 and 480 Hz (fixture option `--hz 480`). Audit each native-step trace with its matching `--hz`. The independent audits for mounted23b and mounted23a passed at their respective rates. The peak loaded axial excursions were 36.77 and 23.05 µm. The fault control still caught actual powered rotation. See the [measurement summary](../../measurements/mounted-elastic-recovery.json). The separate robot recording is a partial commissioning attempt.

### Unannounced axial loads and revised mounted fixture

The split predictor passed its declared-load checks but displaced the shaft 3.70 mm under an unannounced −5 N load. At the specified stiffness the expected displacement is −5 µm. This invalidates use of that mechanism for unknown robot contact, even though it passed its own powered-cycle checks.

The experimental alternative leaves both original axial stiffnesses and dampings in place. PhysX solves each axial spring during contact about the measured helical datum. The rotational effort uses the measured axial momentum balance, including the tool contact wrench, native bearing and stop reactions. It is a discrete coupling and has been checked on the loads listed below. Each fixture test uses a bounded commissioning instrument. The force-perturbation test adds its load after the predictor has finished.

Run from the fork root with the restored Arena interpreter. Fresh output names keep the physical reports separate.

```sh
study=component-studies/drg80-thread-lock/task/workcell
run=automation-results/20261114T031727-dc73f2bb
for hz in 240 480
do
  python "$study/check_unseen_axial.py" --assets "$run/task-prototype/assets" --out "measurements/unannounced-$hz.json" --hz "$hz" --solver pgs --implicit-axial --transmitted-reaction
  python "$study/check_mounted.py" --assets "$run/task-prototype/assets" --out "measurements/service-$hz.json" --hz "$hz" --solver pgs --continuous-seat --native-friction --implicit-axial --transmitted-reaction --rise-seconds .16 --early --drive-instrument --tightening-travel 1.25 --driver-correction-rad=-.60
done
```

The 0.60 rad clockwise correction and the 1.25 rad wrench travel past nominal first contact are tool inputs for calibrating the fixture. The task grader reads the measured mechanism state and does not use either parameter. At 240 Hz the three re-openings reached 90.558°. At 480 Hz they reached 89.624°. Both included three measured unloaded returns, retained contact force, admissible bearing clearance and a powered-motion check. [Physical measurements and raw trace hashes](../../measurements/robot-load-qualification.json).

For a real commissioning attempt use the same geometry, solver and driver correction in `yam_workcell.py`. The robot path only sends bimanual joint actions. A failed stroke, an incomplete observation stream or a run from the fallback layout cannot establish a service-task score.

```sh
python "$study/yam_workcell.py" --workspace "$run" --output measurements/actual-robot.json --solver pgs --continuous-seat --native-friction --implicit-axial --transmitted-reaction --rise-seconds .16 --driver-correction-rad=-.60
```

A measured control perturbation of 0.030 N·m while the joint was powered triggered the angular fault at native tick 1753. The equivalent PGS axial-load controls at both rates measured approximately ±5 µm for the screw and ±50 µm for the nut. Each recovered after unloading. The 4.60 N versus 9.04 N release preload at the two rates shows that quantitative preload predictions still depend on the integration rate.

In a free-tool trial with the alternative TGS integration, the actual driver reached 3.25 mm insertion and rotated the screw through 62.6° in the first stroke. At its end the measured helical extension was −0.74 µm. The next insertion stopped on the socket lip, with its centre displaced by 0.41 mm. The controller now uses the same measured alignment procedure for the first and subsequent insertions. The [rack-angle screen](../../measurements/driver-grasp-reach.json) uses the pinned YAM IK; it supplies planning data and does not prove grasp or contact success.

The frame and the working faces need separate checks for a free tool. In two 240 Hz PGS robot attempts, the reported tip depth reached 6.45 mm while the centres were 1.74 mm apart. The 6 mm deep original socket has only 0.10 mm nominal clearance on each flat. Both attempts stopped before commanding a turn. Their summary JSON and console logs survived; their optional full waypoint traces failed to serialize a NumPy scalar. The current runner handles native scalar types and atomically preserves the trace before recording its filename. The geometry check clips the 3.95 mm across-flats bit against the six socket planes and the bottom. The runner also checks mount displacement and jaw travel. These setup checks do not replace the full-room damage monitor required for a submission.

Short controlled torques of 0.0005 and 0.003 N·m for 12.5 ms at 240 Hz caused no measured powered rotation. Both passed the recovery checks. The 0.030 N·m fault produced real rotation and tripped the fault latch. The magnitude of the input itself does not decide the grade. Use `--powered-fault --fault-torque-nm 0.003 --fault-hold-s .0125` with the service fixture command to repeat the short test.

The development runner additionally records maximum observed normal force and penetration for contacts between the protected hardware and the tools/robot. A single reported manifold above 250 N or 2 mm exceeds the universal bound. The remaining room-wide pair classification and event-work rules are still required for an official task score.

In the 480 Hz PGS free-tool trial the flat-face guard withdrew after interference. At the final observation the driver centre was 0.773 mm off-axis and the shallowest tip corner stood 2.445 mm above the mouth. The trial correctly stopped before a torque stroke. Its full control trace is preserved on Mac and Brev. The current open problem is repeatable insertion and re-indexing of the unchamfered bit under robot contact. No Dream run or submission has been made.

### Driver lead-in and physical capture

The robot trial uses a 0.50 mm by 45° chamfer on the free driver. The width across flats at the nose is 2.95 mm and reaches 3.95 mm at 0.50 mm from the nose. The screw socket is 4.15 mm across flats. The full-depth clearance remains 0.10 mm per side. The [USD layer](../../input/chamfered-driver.usda) composes with the original first-party driver. The solver uses the same vertices and faces for the visible tip and convex collision hull. The robot command accepts `--square-bit` for an original-tip comparison.

In the fixed-socket contact fixture the lateral tether pulls toward the *initial offset*. It receives no socket-centre target. With the original handling settings, the chamfer seated from the centre at 240 and 480 Hz but stalled at both 0.40 mm offsets. With a 20 N/m lateral tether, stronger upright correction and limited axial feed, the chamfer seated from all three starts at 480 Hz with the original socket sectors. At 240 Hz only the negative offset passed. Two alternative socket collision representations were also tested. The [case table and raw checksums](../../measurements/driver-capture.json) retain every control and failure. Each offset has one trajectory per configuration.

```sh
python "$study/check_bit_capture.py" --assets "$run/task-prototype/assets" --hz 480 --compliant-feed --out measurements/driver-capture.json
```

### Capture with the threaded assembly and free robot tools

The diagnostic accepts `--mounted-offset-mm` to use the complete unpowered workcell, including the movable screw, nut, seat and axial/rotational coupling. The contact fixture supplies a finite feed to a free driver. No robot participates in that test. With the 0.50 mm lead-in, the 480 Hz mounted fixture passed at zero offset and stalled at −0.40 and +0.40 mm. The 1.00 mm, 45° lead-in passed all three positions with approximately 5.00 mm of full-width flank seated. Each result is a single trajectory. Both tips retain the original 3.95 mm driving width. The [longer-tip USD](../../input/long-lead-driver.usda) is a removable first-party layer.

```sh
python "$study/check_bit_capture.py" --assets "$run/task-prototype/assets" --hz 480 --compliant-feed --long-lead --mounted-offset-mm=0.4 --out measurements/mounted-driver.json
```

YAM can physically seat the 0.50 mm tip and turn the screw by holding wrist yaw during entry and responding to the measured contact wrench at 20 Hz. In yam17 the shallowest nose corner reached 2.13 mm below the mouth, the full-width flank reached 1.63 mm, and the centres were 0.058 mm apart. The screw turned 61.44° before the land depth decreased to 0.75 mm and twisting stopped. At that point the wall clearance remained positive. The `--supported-turn` option seats the driver deeper and pauses its twist to reseat a rising bit. The `--seated-regrasp` option opens the real fingers with the bit in the socket, returns the empty wrist around the handle, then pinches again. Neither option applies forces to the robot's tools or prescribes screw motion. See the [mounted trials, robot outcomes and raw trace checksums](../../measurements/mounted-driver-and-yam.json).

The rack observer transforms the active mesh, box and finite cylindrical collision geometry into world coordinates and checks the entire tool against its cradle. Replay with `check_tool_racks.py --assets "$run/task-prototype/assets" --trace measurements/yam15.trace.json --out measurements/yam15-racks.json`. The per-run tool USD wrappers must be next to the trace. That replay saw both tools racked at reset and each leave its own cradle during lift; two negative controls placed a tool origin inside while its handle protruded. The replay samples at the 20 Hz action rate. The instrumented robot runner also records containment transitions on native physics ticks.

The 0.50 mm tip completed two physical turns in yam18. YAM withdrew, re-indexed and reinserted between them. The third entry stopped with 0.55 mm of straight flank in the socket and 0.095 mm of geometric sidewall interference. [Measured depth and screw rotation](../../measurements/yam-driver-motion.png). In yam19 the 1.00 mm tip completed a 67.8° turn. When the fingers opened, the tool rested with a corner 0.102 mm past the nominal floor; the seated regrasp stopped before a second twist. These are diagnostic attempts.

For an incidence-angle comparison, add `--driver-phase-deg 15` or `30` to the 480 Hz, 1.00 mm mounted-offset command. Both offsets in angle blocked at the 0.40 mm lateral start when the feed held the original wrist yaw. With `--align-before-feed`, the instrument holds the driver clear for one second and physically rotates it toward the nearest measured flat. Both entered the socket.

At 480 Hz with TGS, a fixture instrument leaving 11.74 N of preload allowed the jaw to hit the stop, with 47.5 N of transient contact and 0.0193° of subsequent screw motion. The lock audit detected this. Raising instrument tightening travel from 1.25 to 2.00 rad gave 18.85 N after release. The three measured opening endpoints were 90.014°, 90.009° and 90.007°; the loaded motion and three unloaded recovery checks passed. This change adjusts the operator's tightening action. It does not alter mechanical stiffness, friction or scoring tolerances. Reproduce using the service fixture command above with `--solver tgs --hz 480 --tightening-travel 2.0`. The YAM expert can aim for that regime with `--wrench-preload-n 21`; the measured released force and the live cycle checks remain decisive.

`check_service_reach.py --out measurements/service-reach.json` checks stock YAM IK before the physical wrench and valve sequence. Of six nominal right-hand wrench orientations, 60° supported the complete screened approach and 65° turn. The original straight-down right-hand lever path missed by 10.8 mm. Approaching the lever from the left at an 85° tilt and following its hinge through both detents passed the nominal IK screen. The robot runner selects its actual path from measured robot and tool poses.

The native observer measures finger opening from each arm's actual joint7 and joint8, both nut bores' overlap with the contiguous screw guide, and the complete tools' signed separation from the finite adjustment cylinder. The tool query uses each convex mesh or box separately and retains the finite cylindrical shaft. A wrench whose reference point is outside but whose handle crosses the cylinder correctly fails the outside test. The [geometry checks](../../measurements/geometry-observations.json) include seven tool placements and the second run through the native reset. A preload-trace replay checks the 540 recorded pose snapshots in 6,480 physics ticks and four displaced-nut controls. With `--observe-reset` the running scene samples all of these fields on every native tick. These measurements cover only part of the full scoring contract.

With `--room-damage`, the extended runner reads the native GPU contact stream for all 30 dynamic bodies against all 42 registered actors, including the 12 static colliders. It combines points by unordered body pair, counts mirrored reports once, and uses contact-point velocities including angular motion. Only the two fixed arm-base/table interfaces are exempt. If both damage bands pass, the recorded contacts qualify under either classification. The intermediate band requires the specific touching surfaces to be identified before assigning a grade. At reset all four tool/cradle support contacts passed, and transformed collider bounds proved at least 100.46 mm of moving-arm/tool clearance from the fixture. A lower bound below 30 mm leaves the clearance undecided. [Native observations](../../measurements/room-native-observations.json).

In the 480 Hz PGS robot run (yam20), the full-width flanks reached 3.24, 3.50 and 3.56 mm on three separate insertions. The two completed strokes turned the screw 70.32° and 67.97°. The measured screw reached 197.53° clockwise on the third stroke when the wrist ran out of travel. In the separate recorded 480 Hz TGS run, the nose entered 0.96 mm and the full-width flanks did not seat. The [TGS recording](../../measurements/yam-480-tgs-entry.mp4) shows that specific run. [PGS and TGS trial data](../../measurements/mounted-driver-and-yam.json). Use `--adaptive-stroke` to shorten a turn if further wrist motion becomes infeasible and re-index by physically withdrawing the tool.

### Valve operation with the stock YAM arms

The expert can move the physical valve by pushing opposite sides of its handle with the closed fingertips (`--push-valve`). In the independent valve58 run the left arm selected the closing detent and released it. The right arm then selected the opening detent. The final lever angles were −32.93° and +32.93°, the measured pneumatic efforts were −5.0 and +5.0 N and the jaws reached 3.00° and 102.04°. The mechanism started unadjusted. Its 5,472 native samples gave a zero score upper bound throughout. [Valve and wrench measurements](../../measurements/yam-actuator-and-wrench-checks.json). [Three previously measured screwdriver insertions](../../measurements/yam-three-engagements.png).

Use `--check-valve-reversal --push-valve --room-damage` on the physical YAM command to exercise the two detents independently. Use `--check-wrench-stroke` to test a real wrench stroke on the unadjusted hardware. The square jaw mouths in wrench57 stopped short of the nut centre, the wrench slipped and its table contact exceeded the 2 mm penetration limit. The optional `--guided-wrench` variant opens each jaw mouth by 1.2 mm over its first 5 mm and waits for the measured nut to lie between the 13.30 mm flats before applying torque. The screw, nut and locking conditions match the original fixture.

A measured 14-dimensional action sequence can be repeated from reset using `--replay-actions measurements/name.trace.json` with a different output basename. The arms receive the recorded joint and finger commands through Arena; there are no object-pose writes in replay. Check the new native measurements before using any recording as evidence.

The room observer detects off-table tool loss and unsupported falls by the measured centres of mass, opposing finger impulses, rack containment and later impacts. It reads the table collider from USD. The pinned Arena station actually spans 1.10 × 1.30 m at z = 0.7475 m; the early fixture proposal listed 0.80 × 1.20 m at z = 0.750 m. The tools and mechanism in the valve tests remained inside both footprints. The production specification must reconcile this before submission. The general enclosed-tool extraction predicate and ambiguous contact-surface classifications are still open. The local grade bound assumes remaining unknowns are favourable. A positive bound cannot be published as a verified score.

The [22.8 s Arena top-camera recording](../../measurements/yam-valve-detents.mp4) replays all 456 stored bimanual actions of valve58 from a fresh reset. Native measurements independently confirmed both detents and a 102.04° unadjusted opening again. With the tapered wrench guide (wrench59), the nut reached 0.59 mm longitudinal and 0.052 mm lateral offset between the jaws, with 0.086 mm remaining geometric side clearance. The wrench alone drove the freely moving shaft approximately 41.6° during the stroke. The final relative nut turn was only 9.18°, below the diagnostic 20° criterion. The separate wrench trial did not use a left-hand counterhold or produce locking preload.

### Finite mount and tool release

The anchored articulation root resisted the full 200 N overload without any displacement. With the mount fixed to the world outside the internal articulation, the native joint broke at an applied 152.92 N and the fixture moved 11.42 mm. Use `--external-mount` for the finite support. On the new support, ±5 N gave ±5 µm at the screw and ±50 µm at the nut. Both recovered after unloading. An instrumented lock retained at least 5.07 N. The jaw reached 88.733° on each of three openings with three verified unloaded returns. The deliberately applied powered torque tripped the screw-angle guard at tick 3505. [Native checks and trace hashes](../../measurements/finite-mount-controls.json).

From the repository root, with the pinned Arena Python active:

```sh
w=component-studies/drg80-thread-lock/task/workcell
run=automation-results/20261114T031727-dc73f2bb
python "$w/check_mount_release.py" --assets "$run/task-prototype/assets" --out measurements/support.json --external-mount
python "$w/check_bit_capture.py" --assets "$run/task-prototype/assets" --out measurements/unheld.json --hz 480 --external-mount --compliant-feed --long-lead --depth-stop --mounted-offset-mm 0.4 --release
```

The depth stop is a convex 24-sided steel collar 6.50 mm across the vertices and 2.00 mm thick. Its lower face sits 4.75 mm above the bit tip. The driver mass and inertia include this extra material. In the native capture probe the instrument released all force and torque for 480 steps. The smallest floor clearance was 1.25 mm, the straight hex engagement stayed above 3.62 mm and the peak contact load was below 0.90 N. The original uncollared driver crossed the geometric floor by 0.102 mm. A preliminary thin-cylinder collar also failed. [All three physical controls](../../measurements/unheld-driver.json).

In yam23 the free driver turned the adjuster to −197.53°. The left arm reached its wrist limit before it could withdraw. The [camera recording](../../measurements/yam-three-driver-strokes.mp4) shows the entire partial attempt. The follow-up trial (yam24) reserved 12 mm of straight withdrawal and checked joint margin before the next torque command. The first withdrawal cleared the socket, but only the bevel entered on the second insertion. Independent YAM valve tests from the original setting recorded three complete 3.00° → 102.04° reversals. The [valve results](../../measurements/yam-actuator-and-wrench-checks.json) report native contacts and score bounds. The separate [30-position YAM reach survey](../../measurements/fixture-placement-screen.json) uses offline IK and leaves the physical scene unchanged.
