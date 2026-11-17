# DRG 80 thread and lock qualification experiments

These are component experiments for the 90-degree opening-stop benchmark, recorded on the production Brev L40S. The whole workcell still requires a physical YAM demonstration and end-to-end grading.

## Reproduce

Use the run's pinned IsaacLab-Arena Python (Kit 110.3). Fetch the two M8 OBJ files named in `input/nvidia-factory-reference/reference-manifest.json` into that directory. Verify their listed SHA-256 hashes. The original Factory acknowledgments and license accompany the references.

From this directory run each direction from a fresh process, for example:

```sh
python task/loaded1mm.py --device cuda:0 --direction positive --output measurements/my-positive.json --trace measurements/my-positive-trace.json
python task/loaded1mm.py --device cuda:0 --direction negative --output measurements/my-negative.json --trace measurements/my-negative-trace.json
python task/dual_lock.py --device cuda:0 --fixture task --velocity-iters 8 --cover elastic --hz 240 --output measurements/my-dual-lock.json
python task/check_score.py
```

`loaded36.py` uses the original 1.25 mm pitch and 2.5 mm axial offset. `loaded1mm.py` scales both contact geometries axially by 0.8 and starts at a 4 mm offset. Both use +36 degrees. The geometry cook can warn about the source mesh topology. Each result records actual native measurements. The four independent loaded runs used 4,080 steps each. Full per-contact traces remain on Mac and Brev. Their verified hashes and sizes are in `measurements/verified-thread-results.json`.

The two-thread rig uses a moving screw, frame thread, upper locknut and compliant seat. The refined layout removes both initial overlaps. The 240 Hz run kept at least 6.22 N on the seat after releasing the tools. It still exceeded the specified screw rotation and axial speed limits under a 2 N proof load. This rig applies instrumented torque and force directly for diagnosis. Its score is separate from any YAM task score. The Python scorer checks use synthetic input and exercise both rates independently.
