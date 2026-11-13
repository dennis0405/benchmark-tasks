# RoboSimStudio bag reproduction

Measured 2026-11-13 on Blackwell's CPU, with GPU visibility disabled. All eight
GPUs were busy; no GPU simulation or Isaac Sim run is claimed here.

Upstream source: [RoboSimStudio](https://github.com/robosimstudio/robosimstudio/tree/4e52e1a76437ef1296028afea744437f6e9469c0),
commit `4e52e1a76437ef1296028afea744437f6e9469c0`.
`tests/test_open_a_bag.py` drives the shipped `03_materials/06_bag.py` scene
through its existing robot controls. The instrumentation calls that drive with
self-contact enabled and disabled, recording the signed layer-gap metric every
control frame and node positions every 15 frames. It does not change forces,
robot commands or bag material. Instrumentation time is included in timing.

## Measured result

| Metric | Self-contact on | Self-contact off |
|---|---:|---:|
| Recorded control frames | 460 | 460 |
| Resting signed layer gap | 3.884 mm | -0.242 mm |
| Minimum recorded signed layer gap | 3.157 mm | -0.731 mm |
| Final observation-window mouth heights | 75.01–76.84 mm | 74.91–76.81 mm |
| Sampled mesh states with strict transverse crossings | 0 / 31 | 26 / 31 |
| All recorded node arrays finite | yes | yes |

The upstream bag unit tests also passed: 21 tests. These include pure geometry,
declarations and mocked engine tests, not 21 independent physical rollouts.
The two drives establish a self-contact differential for this upstream canvas
bag in this CPU configuration. They do not validate snack-film thickness,
crease retention, clip insertion, air sealing or the shared task.py runtime.
Only one drive per condition has been run.

The independent crossing checker uses segment/triangle tests with a spatial
broad phase, skipping incident triangles. Its synthetic controls detect a strict
crossing and reject separated triangles. It detects strict transverse crossings
at saved states; it excludes coplanar contact and does not certify that nothing
crossed between samples. The signed layer-gap metric is also not a general
intersection certificate for arbitrarily folded geometry.

## Reproduce in an isolated environment

Keep the upstream clone and virtual environment outside the submitted task.
The local files here are only instrumentation, not a vendored upstream checkout.
From the pinned upstream checkout, with the scripts accessible one folder above:

```sh
uv sync --python python3.12 --extra cpu --no-group rtx
export CUDA_VISIBLE_DEVICES=
export ROBOSIMSTUDIO_DEVICE=cpu ROBOSIMSTUDIO_RENDERER=tiled
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH=.
.venv/bin/python -m unittest -v tests.test_bag
.venv/bin/python ../reproduce_rss_bag.py /fresh/output/folder
.venv/bin/python ../check_rss_crossings.py /fresh/output/folder
```

Server experiment sources are under
`/mnt/ssd2/robocurve/tasks/close-snack-bag/experiments/`.
The separate environment and results are under
`/mnt/ssd2/robocurve/runs/close-snack-bag/rss-repro/`.
Compact local evidence (about 0.8 MB) is in
`../evidence/rss-cpu-repro-20261113/`, including package versions, resolved
dependencies, unit-test log, per-frame gap traces, node snapshots and crossing
reports. The existing shared Isaac image was not modified.

Next: reproduce this control with the GPU scene path when capacity allows,
then vary film thickness/contact radius at a fixed mesh resolution. Require
the same differential and crossing checks before adding crease memory or a clip.
