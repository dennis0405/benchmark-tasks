# Thin-film implementation and validation — 2026-11-13

Status: **pouch self-contact controls passed at sampled states; clip retention
failed. No completed deformable task, YAM demo, Isaac Sim run or submission is
claimed.** The existing rigid task and its scores have not been replaced.

## Material implementation

`film_material.py` defines an uncalibrated isotropic equivalent: thickness
80 µm, density 1000 kg/m³, Young's modulus 100 MPa, Poisson ratio 0.35. These
are assumptions, not measurements of a particular package. Thickness changes
areal mass linearly, membrane coefficients linearly and bending rigidity
cubically. At a 2.5 mm representative hinge width:

- areal density: 0.08 kg/m²;
- plane-stress shear coefficient: 2962.96 N/m;
- plane-stress first Lamé coefficient: 3190.88 N/m;
- bending rigidity: 4.8623e-6 N·m;
- approximate Newton edge coefficient: 0.0019449;
- numerical yield curvature: 100 m⁻¹, giving a 0.25 rad hinge yield;
- crease recovery: 0.1 s⁻¹.

The edge conversion approximates each hinge by a uniform-width strip. It needs
resolution/convergence checks and real-package bending calibration; it is not
an exact laminate constitutive model. Newton's bending implementation uses
`k = stiffness * edge_rest_length`, motivating the `D / width` starting point.
Source: [Newton 1.7 bending kernels](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/vbd/particle_vbd_kernels.py).

Crease memory uses the yield-based rest-angle kernel from the pinned
[RoboSimStudio paper example](https://github.com/robosimstudio/robosimstudio/blob/4e52e1a76437ef1296028afea744437f6e9469c0/robosimstudio_materials/paper.py).
It changes the elastic rest angle after measured bending; it does not pin a
successful final pose. The native pouch restricts plastic hinges to its mouth.
The pouch starts pre-folded in the folded trials, explicitly as a reset condition.
No robot folding action is demonstrated by those trials.

## Measurements

| Experiment | Result | Interpretation |
|---|---|---|
| Material scaling tests | 3 passed | Thickness scaling and resolution-dependent yield are checked; not material calibration |
| Upstream crease-kernel tests | 7 passed on CPU | Yield, recovery and angle convention checked |
| CPU strip, numerical stiffness profile | 23/23 saved states without strict crossings in each condition; plastic strip retained 1.994 rad after 3 s, elastic strip reopened | Qualitative memory control; this strip uses edge coefficient 3, **not** the derived pouch coefficient above |
| SuperDex 80 µm rest probes, 5 mm cells | 2 mm contact radius: 0/13 saved states crossed; 1 mm and 0.1 mm: 13/13 crossed | Reducing physical thickness/contact radius alone does not solve discretization |
| Native VBD derived pouch, 80×64, self-contact on | 180 frames / 3 s; all finite; 0/14 saved states crossed | Basic self-contact control passed under this configuration |
| Same derived pouch, self-contact off | all finite; 13/14 saved states crossed | Expected failure control; a smaller mouth opening alone is not sealing evidence |
| Native VBD clip/film trial 05 | 240 frames / 4 s; 17/18 saved states intruded into clip boxes; film still moving at late samples | **Failed retention diagnostic** despite finite numbers |
| Bare clip, VBD | final measured angle +0.506 rad, outside authored [-0.55, 0.02] | This import/configuration's rigid-joint path is unsuitable without correction |
| Same bare clip, MuJoCo CPU | final angle +0.0083 rad, inside limits | Rigid backend control works; it does not establish mixed rigid/film stability |
| MuJoCo–VBD coupled trial 05 | implausibly large rigid position on frame 1; nonfinite/invalid rigid state on frame 2 | **Failed coupling diagnostic**, not a demo |

The strip's retained-angle measurement must not be presented as validation of
the physically derived full pouch. Force telemetry in failed contact trials is
not calibrated or converged. Earlier trials with contact-buffer overflow were
also rejected; one expensive retuning trial was interrupted and is incomplete.

Crossing checks cover sampled strict transverse triangle intersections and
triangle/box intrusion with a 0.1 µm boundary tolerance. They do not certify
continuous collision freedom or detect every coplanar overlap. All trials here
are individual runs, not repeatability statistics. Air pressure, food,
breakage, insertion, hand release and the final scorer remain unvalidated.

## Environment and reproducibility

The Brev instance `robocurve-snack-film` (`c3udbl8py`) was created with L40S
48 GB, 8 CPUs, 147 GB RAM and a 256 GiB data disk. The displayed quote was
$1.77/hour including storage; stopped storage was quoted at $0.03/hour.
The instance initially exposed driver 565.57.01. An isolated Python 3.12
environment runs Newton 1.7.0, Warp 1.18.0+cu12 and SuperDex 1.0.0. The official
CUDA-12 Warp wheel was used without changing the host driver. Exact resolved
versions are in the selected evidence's `checkpoint-meta/requirements.txt`.

- Sources: `/data/robocurve/tasks/close-snack-bag/experiments/`.
- External pinned upstream clone: `/data/robocurve/robosimstudio/`.
- Environment: `/data/robocurve/envs/rss/`.
- Runs and raw logs: `/data/robocurve/runs/`.
- Compact local evidence: `../evidence/film-material-20261113/`.
- No paid model inference, media upload or competition PR was performed.

Run from the pinned upstream checkout with `PYTHONPATH=.` and the installed
environment; `ROBOSIMSTUDIO_RENDERER=tiled` avoids needing OVRTX for these
diagnostics. Keep the provided task source and asset folder structure intact.

```sh
python /data/robocurve/tasks/close-snack-bag/experiments/native_film_pouch.py \
  /fresh/output --device cuda:0 --nx 80 --ny 64 --seconds 3 --folded
python /data/robocurve/tasks/close-snack-bag/experiments/check_rss_crossings.py /fresh/output
```

`--self-contact 0` selects the negative control. `--clip` is an experimental
closing-only diagnostic: the free clip starts open around the fold. It is
currently **known to fail** and must not be used as a successful demo.
`coupled_film_clip.py` is likewise a failed research probe. Current code adds
more diagnostic guards than some older recordings; use each run's frozen
source where present and its recorded settings when comparing results.

CPU supports the SuperDex bag and small standalone VBD diagnostics, but current
Isaac Sim camera/YAM integration needs a compatible RTX GPU and runtime.
This Brev environment has verified standalone Newton GPU execution, **not**
Isaac Sim 6.1 compatibility. Its driver differs from the tested driver in the
[Isaac Sim 6.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html).

Next required gate: obtain a finite, converged mixed rigid/film solve with valid
joint limits, plausible forces and zero observed intrusion. Check state/reset
initialization, rigid/soft coupling and contact scales on a smaller scene before
another full-pouch trial. Retention, YAM and scoring follow that gate.
