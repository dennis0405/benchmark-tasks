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
