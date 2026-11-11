# Open snack pouch redesign — experimental

This is an isolated geometry and material research prototype, not the Arena
scene, scorer, clip, or YAM demonstration. The previous video showed a directly
initialized fold in an elastic surface; it did not demonstrate a physical
folding action or a calibrated snack packet.

## Geometry

`pouch_geometry.py` generates two connected film surfaces with shared vertices
on the two sides and bottom. Only the torn top is a boundary. The nominal
20 x 16 cm body has a 44 mm maximum thickness, tapering to a 4 mm open mouth.
These are provisional design dimensions, not measurements of a real packet.
The shape is merely a bulged stress-free rest geometry: it does not simulate
sealed air or loose chips. The top edge is a straight cut and the seals have no
finite width. It is not a textured or submission-ready asset.

`test_pouch_geometry.py` checks the boundary, manifold edges, triangle area,
dimensions and the two-surface connectivity. It does not check dynamic contact.

## Preliminary free-body control

Using the Newton VBD environment, `pouch_body_probe.py` instantiates this mesh
on a floor with no clip, contents, air, robot, or direct fold initialization.
It uses the prior experimental triangle and bending coefficients, not measured
material parameters. Its source is an adapted copy of the prior Newton probe;
only the default flat geometry without clip is supported here.

A 0.5 s run with 80 x 64 cells per face remained finite. Its total height went
from 44 mm at the first step to 11 mm at 0.5 s as the unsupported body collapsed.
The saved intersection checker found zero strict crossings at 0.004, 0.25, and
0.5 s. Sampling is sparse and cannot prove continuous collision freedom, and
this is not a folding, grip, sealing or retention test.

```sh
source /data/IsaacLab-Arena/yam.env
cd /data/benchmark-tasks/tasks/close-snack-bag
python -m unittest -v test_pouch_geometry
python pouch_body_probe.py /data/snack-runs/pouch-body-new --duration 0.5
python check_surface_intersections.py /data/snack-runs/pouch-body-new/surface-snapshots.npz
```

## Design and validation still required

- Model a partially filled bag with movable content or a justified compliant
  surrogate and account for escaped air; a fixed box is not equivalent.
- Model the multilayer film's stretch, bending, damping and irreversible crease
  response. Current elastic rest-shape bending cannot establish real crease
  behavior. Determine supported solver capabilities before choosing a model.
- Measure a real example: dimensions and film thickness, mass, fold/release
  angle versus time, force versus displacement, and clip force/opening. Fit
  model parameters to those observations and keep independent validation trials.
- Demonstrate finger-driven fold, clip contact, release for 3 seconds, and check
  all film/film and film/clip intersections, retention and leakage proxies.
  Validate scoring against those observations before integrating YAM.

The snack-film description is informed by [market-available snack packaging
research](https://doi.org/10.1002/cite.202300110). Newton's surface bending and
contact parameters are described in its [VBD tuning documentation](https://newton-physics.github.io/newton/stable/concepts/simulation_tuning_vbd.html);
those parameter names alone do not validate any material calibration.
