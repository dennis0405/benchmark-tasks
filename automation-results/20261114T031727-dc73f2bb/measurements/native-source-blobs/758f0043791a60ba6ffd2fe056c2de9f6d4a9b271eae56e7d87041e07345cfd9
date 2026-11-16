# DRG80 opening-stop lock: environment/component stage

This directory contains the public `task.py` YAM declaration, full-scale first-party rigid geometry, and native component diagnostics. It does not claim a robot demo or successful native measurements.

Run with the pinned IsaacLab-Arena Python from this directory. The 480 Hz run must precede the final 240 Hz aggregate so the latter can compare independent traces:

The current pinned `AppLauncher` selects headless execution when no visualizer is
requested; it does not expose a `--headless` CLI option.  Run the bounded native
API preflight first:

```sh
python run_native_api_preflight.py --device cuda:0
```

The component/aggregate commands below are retained for the later qualification
stage and must not be run until this preflight has produced a passing native
report:

```sh
python run_component_diagnostics.py --device cuda:0 --refinement --trace ../measurements/physics-480-raw.json --output ../measurements/refinement-preflight.json
python run_component_diagnostics.py --device cuda:0 --trace ../measurements/component-trace.json --output ../measurements/physics.json
```

`physics.json` is the only aggregate contract report. It contains all 28 frozen physics checks and all 14 required control outcomes. The earlier files are API/timestep diagnostics and never replace an aggregate check. All reports contain native numerical readings; the runner raises on missing 480 Hz evidence or unavailable native data.

The geometry is independently authored under the included MIT license. Dimensions and material/thread assumptions are in `assets/fixture-geometry-contract.json`. SCHUNK manuals informed the representative operation but no proprietary CAD or manual text is distributed.
