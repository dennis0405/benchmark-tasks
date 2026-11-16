# Gripper-stop tool geometry seed

Status: CPU-authored geometry, **not native contact, YAM or task validation**.
This is operator-owned first-party input for the environment builder. It is not
a completed task or permission to bypass the novelty/physics/demo gates.

`geometry_seed.py` exports four portable rigid USD assets and `geometry.json`:

- A 3.95 mm AF male hex driver, shaft and graspable handle. Origin: tip end.
- An open wrench with a 13.3 mm jaw gap and thin handle. Origin: nut axis;
  the mouth opens toward +X and the handle extends toward -X.
- A 13 mm AF, 4 mm thick nut with an approximately 8.4 mm bore. Origin: center.
- A smooth 7.8 mm adjustment-screw proxy with a 4.15 mm AF, 6 mm deep socket.
  Origin: socket mouth; the threaded-proxy body extends toward -Z.

Units are metres. The ring/socket has 24 individual convex collider sectors,
preserving the empty center. Applying one convex hull to an entire ring would
seal the hole. The wrench uses separate boxes and keeps its mouth open.
Use `geometry.json` for component points, datum frames and estimated mass/inertia;
an environment may adapt the geometry while recording the resulting source hash.

All dimensions, masses, inertia and contact offsets are **benchmark assumptions**.
These are independently authored generic tools, not SCHUNK CAD or a verified
replica of its accessories. There is no screw thread, nut preload, friction law,
joint, actuator, scoring or robot controller in these assets. The environment
builder must supply and physically validate those, including counterholding.
The primitive collisions here are proposals: native timestep/contact checks can
require different offsets or shapes. A CPU clearance check does not prove
grasping, torque transfer, stable insertion or nonpenetration in simulation.
The steel/grip materials are visual looks only; assign and validate native
friction/restitution materials in the environment instead of inferring them
from appearance.

Regenerate with `python geometry_seed.py --output assets`.
Geometry tests run with `pytest seeds/drg80_stop/test_geometry_seed.py` from the
automation root. Original industrial-operation references remain in the run's
`input/operator-source-notes.json`; no vendor artwork or CAD is redistributed.
Run `uv run --no-project --with usd-core python validate_usd.py --assets assets`
in this folder for CPU USD parsing/composition/topology checks. Its output records
the actual OpenUSD version and asset hashes, and remains separate from native
physics evidence.

License: MIT for this first-party code and geometry; see `LICENSE`.

## Optional articulated fixture proposal

`fixture_seed.py` authors a separate `fixture/bench.usda`, four body-part USDs,
layout/geometry metadata and CPU composition/topology/anchor checks. It includes
a hollow frame, two jaws, a piston, the tool seed's adjustment screw and two nuts.
The internal mechanism uses proposed cylindrical D6 guides, jaw hinges, a gear
relation and a piston/rack relation. Referenced body roots and six joint anchors
resolve in CPU OpenUSD. PhysX-specific coupling frames and ratio sign require
native verification; no simulator has run this proposal.

The loose nut begins 1 mm above the cover. Under the proposed piston/jaw ratio,
setting the stop from its initial opening to 90 degrees advances the screw by
about 0.464 mm. Even if the loose nut co-rotates and follows that travel, about
0.536 mm remains before seating. A CPU geometry regression checks this margin;
native load, clearance and target-angle measurements are still required.

Generate with `uv run --no-project --with usd-core python fixture_seed.py`.
The valve, thread/helix dynamics, friction updates, scorer and robot controller
are intentionally missing. The scene is not a working environment. In
particular, its draft drive targets must not be mistaken for a completed
passive thread model. `native-component-plan.md` explains the component probes.

This is an independently authored representative cold-bench fixture. The
housing dimensions, exposed internal transmission, nut 74 placement and all
parameters are assumptions. It is not a manufacturer's CAD model. For Arena
integration use explicit body/root frames from `fixture-proposal.json` or a
verified custom scene factory; a single ordinary library `Asset` does not
automatically expose all seven native rigid bodies to task state/recording.

## Analytic thread reference, not a solver result

`thread_reference.py` supplies an optional energy-consistent internal force and
torque pair for the coaxial screw/nut. Its tests check the energy gradient,
reciprocal reaction, passive dissipation and multiple complete thread turns.
It does not integrate motion, implement native contacts, claim static locking
or pass a component gate. The coefficients used in tests are illustrative.
The native builder must supply measured unwrapped relative angles and the
correct COM/frame transforms. It must check integration stability with the
actual masses, inertias, physics timestep and chosen solver. An implicit joint
implementation may replace this reference.

Version-specific schema caution: the actual pinned Isaac runtime's
`generatedSchema.usda` SHA256
`4a7638c3890522bbe7c25600c0090e6cb7f403550ccb8be7641bf84034213b83`
defines the coupling relationships as `physics:hinge0`, `physics:hinge1`,
`physics:hinge`, `physics:prismatic` and the ratio attributes as
`physics:gearRatio`/`physics:ratio`. The current fixture proposal uses those
names. Newer ovphysx docs show other namespaces; do not substitute those
without checking the runtime actually used. This source check still does not
prove native parsing or coupling behavior.
