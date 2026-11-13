# Snack pouch: GPU-free review, 2026-11-13

This is a design review and CPU-only geometry check, not a simulation result.
The rigid baseline and the shared-runner bridge stay separate from these probes.

## Evidence and implementation choice

- [RoboVoLo](https://arxiv.org/html/2606.07723) informs task language and
  evaluation design; it does not validate a snack-film model.
- [RoboSimStudio bag](https://github.com/robosimstudio/robosimstudio/blob/4e52e1a76437ef1296028afea744437f6e9469c0/examples/03_materials/06_bag.py)
  uses a sewn, self-contacting SuperDex canvas shell. It reports 3.9 mm resting
  layer separation, which is an upstream observation, not a snack-film value.
- [RoboSimStudio paper](https://github.com/robosimstudio/robosimstudio/blob/4e52e1a76437ef1296028afea744437f6e9469c0/examples/03_materials/05_paper.py)
  demonstrates VBD yield-based crease retention, not a plastic bag. Its
  one-way coupling with slide feedback must not be assumed to reproduce all
  two-way tool forces.
- [Isaac Lab deformables](https://isaac-sim.github.io/IsaacLab/main/source/concepts/deformables.html)
  lists surface deformables for PhysX and Newton VBD. The task's public
  runner's support for a given backend and its interaction with YAM, contacts
  and cameras still need a runtime test. Current online docs may describe
  newer versions than the installed image.
- The lab image `yam-inspect-isaac:6.1.0` exposes Python 3.12.13, Newton 1.5.2
  and Warp 1.16.0. The examined RoboSimStudio revision requests Newton 1.7.x;
  do not upgrade the shared image in place. Test it in a separate environment.

The preferred challenge-integration candidate is the existing Isaac Lab-Arena
plus Newton VBD surface path, starting from `pouch_geometry.py` and the earlier
probes. Reproduce the upstream bag and paper examples in an isolated environment
as reference controls when GPU access is available. The SuperDex bag and VBD
paper components cannot simply be combined without engineering and validation.

## What exists now

`open_pouch` builds two connected triangular surfaces: the two sides and bottom
share vertices, and the opposite mouth is open. In the 80 x 64 foldable-mouth
configuration with a 0.3 mm mouth half-gap, CPU inspection found 10,305
vertices and 20,480 triangles, no zero-area triangles under a 1e-12 m² cutoff,
and a 0.6 mm central opening. The specified geometry is 200 x 160 mm, with a
44 mm maximum initial body thickness. These are authored initial values, not
measured real packaging or evidence of stable simulation.

Ten CPU tests passed in the shared image: topology and opening, fold initializer,
triangle-box intersection and fixture clearance, and crease-state calculations.
They do not test dynamic nonpenetration, pressure, clip forces, or a YAM rollout.
V16 and V18 remain failed dynamic controls: sampled triangle/support crossings
were present even when sampled vertices were outside the support box. See
`ITERATION_NOTES.md` for the exact observations and limitations.

## Next controlled GPU experiments

1. Record image/package versions, solver settings and mesh hashes. Run an
   unmodified upstream bag control and a separate paper control if their
   dependencies can be isolated; compare observations with their reported
   results. No claim about snack material follows from reproducing either.
2. Begin with a coarse pouch and no clip or chips. Sweep mesh and contact
   settings one at a time. Track node positions, speeds, layer gaps and mouth
   area over time, and capture both triangle/rigid and layer/layer crossings.
   Include a self-contact-off control and a no-contact force baseline. Check
   convergence and contact-buffer capacity; do not accept static snapshots as
   proof of continuous collision freedom.
3. Add a press and retract it before adding the clip. Compare forced and
   unforced crease retention and monitor fixture contacts/penetration at a
   higher sampling rate. A paper yield rule is a hypothesis requiring
   calibration against a real snack pouch, not verified material physics.
4. Test clip opening, insertion and release with real joint motion, not a
   fixed attachment. Require contact on both jaws, no observed crossings,
   adequate mouth coverage and a 3-second fixture-free hold; run no-clip and
   misplaced-clip negative controls. Evaluate two-way force transfer separately.
5. Only after the fixture checks pass, integrate YAM and a scorer into the
   shared task runner. A dynamic mouth-area/gap metric is a sealing proxy,
   not a demonstrated airtight seal. Model chips and air in separate tests;
   do not claim breakage or pressure fidelity before calibration.

Newton 1.7's [VBD guide](https://newton-physics.github.io/newton/1.7.0/solvers/vbd.html)
marks the solver experimental. Tuning and collision behavior must be checked
against the version actually used in each experiment.
