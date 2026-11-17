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
