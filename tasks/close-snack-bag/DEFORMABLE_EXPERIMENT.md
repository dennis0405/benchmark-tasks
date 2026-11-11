# Deformable packet experiments — not submission-ready

These are isolated PhysX tests, not YAM actions. They must not be reported as a
successful benchmark demonstration. `arena_scene.py` remains the older rigid
panel task; its scorer and demo do not evaluate the deformable packet.

## Model

`bag_surface_mesh.py` generates a single connected, oriented triangle mesh: two
200 x 160 mm skins, joined at both sides and the sealed end, with an open
mouth. `surface_press_probe.py` and `surface_clip_probe.py` spawn it as a
PhysX surface deformable on GPU. The body contains a fixed rigid cuboid as a
contents fixture; it is not a model of individual chips or trapped air. The
press probe uses a kinematic plate, while the clip probe places the clip and,
optionally, the folded nodal positions directly at reset. Neither test uses a
robot.

In the optional `--hold-fixture` probe, a kinematic plate presses an outer edge
and the clip hinge actuator is held open for one second before being released.
The plate withdraws by 2.5 s; the probe then continues until 6 s. This is also a
test fixture, not a robot action.

The authored rest mesh is flat at the mouth. The `--folded` test writes a
geometric two-turn initial state to the nodes **once at reset** while retaining
that flat rest mesh; it is not a physically demonstrated folding action. Mesh
nodes are matched to the original positions with a one-to-one nearest-neighbor
check. No node or clip pose is written during the subsequent rollout. The
surface models are approximate, and their material is not calibrated to a real
packet. Simulated thicknesses of 1 mm and 0.1 mm were explored; neither is a
measurement of the intended packet. Air, multilayer film plasticity, tearing,
crimp seams, food geometry, and actual force-displacement behavior are not
modeled. Numerical finiteness alone is not evidence of physical realism.

## Observations

- With the 1 mm surface and fixed contents fixture, the press region's median
  height went from 17.9 mm after settling to 1.8 mm under the plate and 2.4 mm
  after release. The no-press control remained around 17.6 mm. Lowering the
  dynamic friction from 0.4 to 0.05 produced approximately 4.4 mm after
  release. The remaining depression could involve snagging, friction, and/or
  self-contact; it has not been calibrated to a real material.
- A flat mouth placed inside the clip retained nodes within 1 mm of both jaw
  boxes for 4 s and a hinge angle near -0.08 rad. The empty clip reached about
  +0.008 rad. This is indirect geometric and kinematic evidence of interaction,
  not a measured contact force or a demonstration of clipping folded material.
- For the twice-folded 1 mm surface, one deep clip initialization had about
  2.5 mm of node penetration into the lower jaw. A shallower, initially
  nonpenetrating initialization allowed the clip to slip away. Neither is
  valid retention evidence.
- A 0.1 mm surface run remained finite for 4 s. In a direct-placement trial,
  some mouth nodes remained near both jaws, but the vertical mouth span rose
  from about 13 mm to 24 mm. The neighboring nodes and the clip angle alone do
  not establish that all folded layers were retained. A matching unclipped
  trial is also saved. A second run showed variations in jaw position and
  penetration; repeated robustness and intersection checks are still needed.
- With the edge-hold fixture and open-jaw command, the folded mouth's vertical
  extent increased from approximately 13 mm at the start to 23 mm at 6 s. The
  fixture had been withdrawn for 3.5 s by then. This result does not establish
  the required closed-fold retention, even though some nodes remained close to
  both jaws. `fold-clip-fixture.json` records the sampled result.

The jaw proximity metric is signed distance from sampled mesh nodes to each
oriented jaw box. A negative value indicates a node inside the box; triangle
intersections may occur even when no node is inside. A near node is not a
contact-force reading. PhysX's rigid contact reporter used by the old scorer
cannot simply be assumed to report deformable contacts: this requires a
separate implementation and backend validation. NVIDIA documents contact
reporting limitations for deformables in an earlier Omni Physics version; the
exact behavior of this installed backend still needs testing. See [NVIDIA's
known limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/dev_guide/guides/current_limitations.html).

Numerical results for the individual experiments are in `evidence/deformable/`.
The tests were developed iteratively, so earlier result files correspond to
earlier versions of the probe scripts. The current flags reproduce their
intended setups, not necessarily bit-identical trajectories. The environment
used Isaac Sim 6.1, Isaac Lab source commit `91a0906f9`, and an NVIDIA L40S.

## Run

Use a **new output directory for each run**, so existing files are preserved:

```bash
source /data/IsaacLab-Arena/yam.env
cd /data/benchmark-tasks/tasks/close-snack-bag
python -m unittest -v test_surface_mesh test_scoring
python surface_press_probe.py /data/snack-runs/new-press
python surface_press_probe.py /data/snack-runs/new-no-press --press-z 0.09
python surface_clip_probe.py /data/snack-runs/new-flat-clip
python surface_clip_probe.py /data/snack-runs/new-empty-clip --empty
python surface_clip_probe.py /data/snack-runs/new-fold-clip --folded \
  --clip-x 0.194 --clip-z 0.0145 --fold-radius 0.0022 \
  --thickness 0.0001 --contact-offset 0.0002
python surface_clip_probe.py /data/snack-runs/new-fold-no-clip --folded --no-clip \
  --fold-radius 0.0022 --thickness 0.0001 --contact-offset 0.0002
python surface_clip_probe.py /data/snack-runs/new-fold-fixture --folded --hold-fixture \
  --clip-x 0.194 --clip-z 0.0145 --fold-radius 0.0022 \
  --thickness 0.0001 --contact-offset 0.0002
```

Before integrating the surface into the task, demonstrate nonpenetrating
layer retention through release, inspect all folded layers and the 96 mm
coverage criterion, validate self-collision and force/geometry measurements,
then implement deformable-node-based scoring and run a new full YAM demo.
Material measurements and comparison against a real packet are still needed.
