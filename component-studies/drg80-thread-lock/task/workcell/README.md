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
