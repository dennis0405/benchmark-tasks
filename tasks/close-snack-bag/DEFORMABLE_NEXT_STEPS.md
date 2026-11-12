# Deformable bag: references and next experiments

Status: research and an experiment plan, not a completed deformable task. No
RoboSimStudio or Blackwell run was performed for this note. Preserve the working
rigid-packet baseline and its 1.0 YAM rollout while testing alternatives.

## What the references actually show

### RoboVoLo: benchmark design

[VoLo / RoboVoLo](https://arxiv.org/html/2606.07723), section 3 and appendix D,
describes 126 long-horizon tasks in 15 categories, across common-sense, memory,
complex references and world knowledge. Its open-ended instructions and
failure-mode breakdown are useful for task ideation: for example, the snack
instruction implies preserving the food, and a good test should reject a fixed
action that ignores the scene. The paper's benchmark runs on Isaac Lab/PhysX;
it does **not** establish a deformable snack-film model. Compare the
[released task list](https://github.com/NVlabs/RoboVoLo) and RoboCurve's current
board before claiming a new task is distinct.

### RoboSimStudio: materials and interaction

The following references were inspected at RoboSimStudio commit
[`4e52e1a76437ef1296028afea744437f6e9469c0`](https://github.com/robosimstudio/robosimstudio/tree/4e52e1a76437ef1296028afea744437f6e9469c0).
Reported measurements below are upstream's results, not ours.

1. [`03_materials/06_bag.py`](https://github.com/robosimstudio/robosimstudio/blob/4e52e1a76437ef1296028afea744437f6e9469c0/examples/03_materials/06_bag.py)
   models a canvas bag with two sheets sharing vertices along three edges. It
   routes the bag to SuperDex and uses shell self-contact; the YAM grasps a lip
   to open the bag. `mouth_opening` and `layer_gap` provide useful measurements.
   The example needs the `superdex` extra and Python 3.12. It is an **opening**
   task, not evidence of a folded, clipped or airtight snack pouch. Its canvas
   preset has 1 mm thickness and a 4 mm self-contact radius; upstream reports a
   roughly 3.9 mm resting layer gap. Those values are not snack-film calibration.
2. [`03_materials/05_paper.py`](https://github.com/robosimstudio/robosimstudio/blob/4e52e1a76437ef1296028afea744437f6e9469c0/examples/03_materials/05_paper.py)
   uses a Newton VBD sheet with yield-based crease memory, two YAM hands and
   slide-force feedback. Its
   [robot test](https://github.com/robosimstudio/robosimstudio/blob/4e52e1a76437ef1296028afea744437f6e9469c0/tests/test_fold_paper_robot.py)
   checks fold retention, contact forces and arm separation. Paper is not
   snack film, and this example does not itself provide a closed bag.
3. [`06_tasks/06_bag_scored.py`](https://github.com/robosimstudio/robosimstudio/blob/4e52e1a76437ef1296028afea744437f6e9469c0/examples/06_tasks/06_bag_scored.py)
   shows a task with a measured mouth-opening threshold. It is a useful scoring
   example, but does not score clip engagement, leaks, food damage or the
   3-second retention criterion.
4. [`01_cloth/02_engines.py`](https://github.com/robosimstudio/robosimstudio/blob/4e52e1a76437ef1296028afea744437f6e9469c0/examples/01_cloth/02_engines.py)
   shows backend selection. The upstream cloth example uses one-way coupling:
   the cloth does not necessarily push back on the arm. Do not infer realistic
   bidirectional clip/film forces from a visually successful grasp alone.

At this revision RoboSimStudio specifies Newton >=1.7,<1.8; the earlier Arena
environment used Newton 1.5.2. The bag's SuperDex backend and the paper's VBD
crease model are different implementations. Combining the two, adding a spring
clip, and integrating with Isaac Lab-Arena/Dream-1 are research work, not
established compatibility. Use an isolated environment to avoid changing the
working Arena installation.

## Proposed sequence after Blackwell access

1. **Inspect the actual allocation.** Check login, GPU type and VRAM, driver,
   OS/container workflow, storage, network and the lab's scheduling rules.
   The earlier Brev instance was reported deleted. Most `/data/snack-runs/`
   intermediates were not pushed, while this branch retains the final baseline,
   compact diagnostics and two raw snapshots. See `RESTART.md`.
2. **Recover the baseline first.** Clone `feature/close-snack-bag`, create a
   fresh Arena environment, run `test_scoring` and `demo_robot.py`, and compare
   its video, score and source hashes with `evidence/flat-chip-demo/`. Fresh
   setup and Blackwell execution remain untested. Confirm credentials only when
   needed; do not commit them.
3. **Reproduce upstream examples separately.** In another environment pinned
   to the inspected commit, follow its
   [install guide](https://github.com/robosimstudio/robosimstudio/blob/4e52e1a76437ef1296028afea744437f6e9469c0/examples/install.md).
   Start with its bag and paper examples, their negative controls and tests;
   log exact package versions, solver choices, runtime, forces and visual
   observations. Its installation guide notes that a B200 without RT cores
   uses the tiled renderer rather than OVRTX; inspect the actual GPU instead of
   assuming the server's model or rendering capability.
4. **Build a minimal film pouch.** Compare the existing sewn-surface geometry
   with the upstream bag, then vary contact radius, mesh spacing, thickness,
   damping and self-collision independently. Use a measured real pouch if
   available. Check watertight seams except the opened mouth, plausible
   unopened-body shape, layer separation and triangle/rigid crossings. A
   plastic-memory rule needs independent validation; a canvas preset or a
   paper crease is only a starting hypothesis.
5. **Prove clip contact before robots.** Add the spring clip with controlled
   opening, insertion and release. Compare fixture and no-clip controls; record
   both jaw contact, force limits, relative slip, coverage, fold geometry,
   release, sampled and continuous collision checks where supported. Reject
   outcomes with unobserved or penetrating contacts. Validate a hold for at
   least 3 s with all fixtures clear. Do not attach or teleport the clip as a
   substitute for a grasp.
6. **Add YAM and scoring.** Only after clip/film contact is stable, test each
   arm, fingertip collisions and two-way force feedback, then a contact-only
   scripted demonstration and negative controls. Derive score inputs from
   validated observations, enforce continuous hold and test invalid/unknown
   readings. Air pressure, food containment and fracture are separate models:
   include them only after independent controls and calibration, or describe
   their omission explicitly.
7. **Check the submission path.** Verify that the selected solver, scene,
   reset, cameras, action interface and scorer are supported by the competition
   runner. Reproduce the baseline, run Dream-1 when credentials are available,
   record actual output, and assess the board's current duplicate and review
   rules before uploading or submitting.

The older probe logs are in `ITERATION_NOTES.md`: notably the v16 and v18
fixture trials had strict sampled film/support crossings. Treat them as failed
diagnostics, not as evidence that the physical bag is complete.
