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
