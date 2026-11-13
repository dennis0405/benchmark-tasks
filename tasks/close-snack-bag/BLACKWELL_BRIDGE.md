# Blackwell shared-runner bridge

`task.py` adapts the existing rigid-panel `arena_scene.py` scene and scripted
`demo_robot.py` controller to the lab's shared `robocurve-skill` runner. It
restores the pre-folded joint state only at reset, uses the original physical
contact monitor and geometry measurements, and feeds those observations to
`bridge_scoring.py`. A full score requires 60 consecutive control frames of
contact, alignment, coverage, clearance and stability. Missing contact evidence
fails closed. A recorded violation permanently sets the episode score to zero.

This bridge has passed scoring unit tests and imported in the lab image, but it
has **not yet been validated by a full Isaac Sim run**. It does not turn the
rigid packet into a deformable bag, model air pressure or chip breakage, or
prove real-world material behavior.

After confirming GPU 4 is available, run on Blackwell:

```sh
python3 /mnt/ssd2/robocurve/.agents/skills/robocurve-skill/scripts/robocurve.py \
  run close-snack-bag --gpu 4 --policy scripted --episodes 1
```

Use the run directory printed by the launcher to inspect the result, logs,
trajectory and videos. A successful launch alone is not a successful task: check
`result.json`, the contact and hold evidence, and the recorded motion. The
runner snapshots the task and runtime into the run directory for reproducibility.

For CPU-side scoring checks inside the compatible environment:

```sh
python -m unittest -v test_scoring test_bridge_scoring
```
