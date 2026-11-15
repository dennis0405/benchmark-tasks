# LM rail datum seating — rapid feasibility prototype

This is an actual standard-YAM prototype, not a final benchmark submission. It uses only public
`TaskFile` boxes, the stock `yam_scripted` policy, public Cartesian `Waypoint` actions, and the
stock `task_score` interface. It never writes a pose or velocity after initialization and adds no
joint, grasp constraint, guard, observation, or scorer semantic.

The rail is 320 x 28 x 16 mm and 0.80 kg. The fixture is a fixed mounting base, one continuous side
datum, and two visible end stops. The demo makes a genuine bimanual attempt to share and carry the
same free rigid body. Whether the grippers establish and retain that contact is deliberately left to
the GPU prototype run.

Material/IK check (CPU only; never report this as native success):

```bash
/data/robocurve-automation/sources/IsaacLab-Arena/.venv/bin/python \
  task/lm-rail-datum-seat/validate_material.py \
  --report measurements/lm-rail-datum-seat/material_check.json
```

Native pinned GPU run on the trusted host:

```bash
source /data/robocurve-automation/sources/IsaacLab-Arena/yam.env
/data/robocurve-automation/sources/IsaacLab-Arena/.venv/bin/python \
  task/lm-rail-datum-seat/run_prototype.py \
  --report measurements/lm-rail-datum-seat/measurement.json \
  --max-steps 600 --control-hz 20
```

The runner always writes a fresh measurement JSON, including failures. A successful run additionally
contains raw extracted actions, task-hook body-state JSONL, the original Inspect Robots log, an H.264
top-camera video, and the first-frame setup image. Score 1.0 is prototype evidence only.

Industrial source: THK, *LM Guide Mounting Procedure*, especially Fig. 7 and the reference-surface
marking instructions: https://www.thk.com/in/en/products/lm_guide/maintenance/0002/
