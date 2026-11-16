# DRG80 opening-stop lock: environment/component stage

This directory contains the public `task.py` YAM declaration, full-scale first-party rigid geometry, and native component diagnostics. It does not claim a robot demo or successful native measurements.

Run with the pinned IsaacLab-Arena Python from this directory. The 480 Hz run must precede the final 240 Hz aggregate so the latter can compare independent traces:

```sh
python run_api_preflight.py --headless --device cuda:0
python run_component_diagnostics.py --headless --device cuda:0 --refinement --trace ../measurements/physics-480-raw.json --output ../measurements/refinement-preflight.json
python run_component_diagnostics.py --headless --device cuda:0 --trace ../measurements/component-trace.json --output ../measurements/physics.json
```

`physics.json` is the only aggregate contract report. It contains all 28 frozen physics checks and all 14 required control outcomes. The earlier files are API/timestep diagnostics and never replace an aggregate check. All reports contain native numerical readings; the runner raises on missing 480 Hz evidence or unavailable native data.

The geometry is independently authored under the included MIT license. Dimensions and material/thread assumptions are in `assets/fixture-geometry-contract.json`. SCHUNK manuals informed the representative operation but no proprietary CAD or manual text is distributed.
