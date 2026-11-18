# Hydrostatic leak test on a water-filled pipe

Pressurize the circuit with the right-hand pump lever, isolate the pipe, and compare the TEST gauge readings 60 seconds apart. Set the result selector, reopen isolation, open the return, and release both hands. Either arm may operate the fixture. The benchmark instruction, thresholds, reset and initial layout are in task.json. A new single-episode process gives each specimen an independent reset.

## Reporting a diagnosis

Once the timed observation is complete, move the result selector from neutral to the chosen mark while isolation is closed and the return is shut. That first choice commits the diagnosis. Opening either valve before committing, a wrong first choice, or reaching the opposite mark later in the same active trial permanently caps the diagnostic trial at 0.50. Overpressure still scores zero. For full credit keep the correct mark through the final three-second hands-clear interval.

## Recorded results

The videos are the original Isaac Sim recordings. The scores below and the Dream-1 score were verified by replaying the recorded observations with the submitted scorer. The scene, actions and instruction are the ones used in the recordings.

| Specimen | Seed | Pressure during isolation | Decision | Scripted score | Duration |
| --- | --- | --- | --- | --- | --- |
| Capillary leak | 0 | 2.946 to 1.434 bar | FAIL | 1.0 | 125.60 s |
| Plugged | 1 | 3.100 to 3.100 bar | PASS | 1.0 | 123.35 s |

The passive lever stayed still with both hands clear for the last 56.1 s of both isolation windows. Opening the return let the handle settle back onto its lower stop. See MODEL.md for the hydraulic approximation. Numerical pressure traces come from this model. The scripted controller reads modeled pressure. Dream-1 receives only the instruction, top and wrist RGB images, and measured arm and gripper state.

## Run a trial

The verified environment uses Isaac Sim 6.1.0, CPU PhysX at 240 Hz, a 20 Hz controller, and an RTX GPU for camera rendering. Use the pinned image or build it with `python runtime/build_image.py --image yam-inspect-isaac:6.1.0`. The shared Blackwell launcher supplies the simulator permissions and writable cache. The standalone example runs as the host user with the simulator access group (1234 in the pinned image) and a private writable cache. From this task directory, set GPU to an available index. Store outputs outside the task folder so the frozen source copy excludes previous runs.

```sh
GPU=0
OUT="$(mktemp -d)"
CACHE="$(mktemp -d)"
mkdir -p "$CACHE/tmp"
docker run --rm --gpus "device=$GPU" --shm-size 8g --user "$(id -u):$(id -g)" --group-add 1234 -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v "$CACHE:/tmp/task-home" -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp -e NVIDIA_DRIVER_CAPABILITIES=all -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -v "$PWD:/task:ro" -v "$OUT:/outputs" -w /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 /task/run.py --policy scripted --episodes 1 --seed 0 --device cuda:0 --livestream 0 --viz none --kit_args '--portable-root /tmp/task-home/kit --/renderer/multiGpu/enabled=false'
```

Use seed 1 for the sound control. Each episode is limited to 4,000 actions or 200 simulated seconds. The runner writes H.264 videos, initial and final frames, result.json and trajectory.jsonl. For Dream-1 use the same command with `--policy dream`. Mount an owner-only credentials file at /run/secrets/dream.env and pass `-e DREAM_CREDENTIALS=/run/secrets/dream.env`. Mount the operator's existing authorized budget directory at /dream-budget:rw. The portable runner defaults to /dream-budget/ledger.sqlite3. In the recorded trial the same transport guard charged request reservations against the team's previously authorized one-episode reservation. Session replacement can continue at the current simulator state (at most eight recoveries, at most one consecutive replacement per request).

In the pinned environment, run `python runtime/rescore.py --task ./task.py /path/to/episode/trajectory.jsonl --expect /path/to/episode/result.json` to check a score. For hydraulic and scoring unit tests, run `python -m unittest -v test_model` from this directory.

## Occupational context and asset provenance

Hydrostatic inspection and pressure-gauge leak detection are core tasks of US plumbers, pipefitters and steamfitters. BLS reports 510,600 jobs in this occupation in 2025. [O*NET task list](https://www.onetonline.org/link/details/47-2152.00) · [BLS employment](https://www.bls.gov/ooh/construction-and-extraction/plumbers-pipefitters-and-steamfitters.htm)

The original pump geometry, controls and dial artwork are supplied under the included MIT license. The pump capacity follows the [RP30 product specification](https://ro.rothenberger.com/ro-ro/produse/testarea-presiunii-apei/pompa-manuala-rp30-61130/). The sequence follows the [manufacturer’s RP50-S manual](https://egeplast.ie/wp-content/uploads/2022/09/ROTHENBERGER-RP-50-S-6.0200-05.02.09.pdf). The 60 s observation is the benchmark’s bench procedure. Hardware calibration is pending.

## Physical fixture and reproducibility


Use a pressure-rated water-only bench pump, brass quarter-turn valves with
90 × 19 × 12 mm handles, a steel selector on a marked plate, rated hose, and
the copper specimen specified in task.json. Reference all heights to the table.
Use rated fittings and an independent 5 bar relief. The return and the specimen
cartridge discharge to the reservoir. Use a steel capillary, nominal 0.20 mm
ID and 0.65 m long, coiled inside the cartridge. The cartridge is a proposed
fixture. It has not been calibrated on hardware. At 20 °C the laminar-flow
estimate gives 0.018 mL/s at 3 bar. Before mounting, apply a steady regulated
3 bar water differential, collect its outlet for 60 s on a 0.01 g scale and
accept 1.08 ± 0.10 g. Use a solid plug for the sound lot and check collection
is under 0.02 g. Bleed air and verify that a plugged circuit loses at most
0.05 bar over 60 s. Keep the cartridge identity hidden from the operator.

Use two calibrated 0–6 bar logging gauges (resolution 0.01 bar), one per side
of isolation, capturing peaks during pumping. Record the TEST trace and a video
of the lever, valve witness marks and selector at 20 Hz. Time the 60 s interval
from closure in band. A scale alongside each handle gives valve angle. Infer
positive displaced volume from calibrated stroke travel (16 mL over 50 degrees,
sinusoidal linkage). A marked 50 mm clearance around the moving controls and
video of both grippers establish the final 3 s released interval.

## Dream-1 episode

Seed 0 used the same capillary leak and scoring code as the full-score demo. Dream-1 scored 0.0 in 4,000 steps (200 simulated seconds). The recorded termination was `step_limit`. The run made 85 guarded network attempts and used 2 in-episode session recoveries. TEST reached a maximum of 0.041 bar.
