# Clip / film closing diagnostic

This is a standalone Newton 1.7 diagnostic. It starts with a pre-folded pouch
and an open, free clip already around the fold. The film has no attachment,
pinning or successful-pose correction. The table supports the pouch. It is
not an insertion, lifting, YAM, Isaac Sim or airtightness demonstration.

Current checkpoint: long holding runs under the expanded geometry gate are
being revalidated. Earlier clip/film-only passes do **not** pass the current
gate: subsequent checks found clip/table penetration and an invalid opening
where the handles overlapped.

## Corrected root cause

The older native probe wrote `-.55` into the builder's joint default while
leaving the authored clip body poses closed. VBD's cached rest-angle reference
then became inconsistent. The corrected probe preserves closed model defaults
and opens only the separate episode State through FK. On CPU, Warp's NumPy
array can alias model memory, so the initial coordinate array uses `.copy()`.

The one-second bare-hinge A/B measured:

| Model defaults | Final hinge angle | Limits respected |
|---|---:|---|
| Consistent closed rest; separately opened episode | +0.00825 rad | yes |
| Older inconsistent defaults | +0.50562 rad | no |

This corrects the earlier attribution to VBD's rigid-joint path itself.
`hinge_rest_repro.py --inconsistent-rest` preserves the failing control.

## Declared validation gate

`verify_native_hold.py` requires all of the following, separately from the
competition scorer:

- Every physics state, including initial closure, checked for triangle/box
  interior intrusion at 0.1 micrometre tolerance and strict transverse film
  self-crossing; also cross-body clip OBB intersections and film/rigid intrusion
  into the z=0 plane. No detected event over the entire episode.
- A continuous three-second window (181 samples at 60 Hz) with active fold
  contacts on both jaw boxes, each spanning at least 80 mm laterally.
- Fold centroid motion relative to the clip at most 2 mm, root drift at most
  2 mm, hinge within authored limits plus 0.01 rad tolerance and variation at
  most 0.02 rad.
- Finite film and rigid measurements, with one completed GPU check stamp for
  every physics state. Missing or duplicate stamps reject a run.

These are diagnostic thresholds, not material calibration. Coplanar self-overlap
and motion between physics steps are not certified. No air, food, breakage,
external loading or lift capacity is validated. Contact-force telemetry is
uncalibrated and is used only as a contact witness.

## Closure impact and contact skin

After fixing the joint reference, the original 1/600 s trial still had two
intruding states in 3,001 checks. Halving the step to 1/1200 s left two in
6,001 checks. Increasing hinge damping from 0.003 to 0.01 produced a central
placement with 0/6,001 intruding/crossing states and a three-second holding
window (0.729 mm relative centroid motion). A +3 mm lateral perturbation still
failed; its longer probe was interrupted after the failure was established.

The first-failure recorder and independent CPU SAT locate a lower-jaw tip
crossing: no film vertex is strictly inside the box, but triangle interiors
cross its corner. A quoted 0.06 mm value was the vertex offset along a face
normal, **not** a measured vertex-inside or triangle penetration depth.
`inspect_native_failure.py` reports that distinction explicitly.

Contact stiffness 1e5 alone also failed at a corner. The follow-up configuration
uses a 0.2 mm conservative **numerical clip contact margin**, 1e5 contact
stiffness, 0.01 hinge damping and 1/1200 s steps. Its short 0.2-second controls
passed in all three lateral placements (center, +3 mm, -3 mm), with 241 physics
states checked per run. Long holding validation is reported in the evidence.
The nominal box geometry and intrusion tolerance are unchanged. The skin
starts contact response before the film reaches the nominal solid surface;
it is a numerical approximation, not calibrated coating thickness or evidence
of exact 80 micrometre contact fidelity.

The earlier 100 mm jaw trial did not prevent the exposed sides of the 160 mm pouch mouth
from reopening. In the earlier passing central-retention run, mouth height
reached about 51 mm. **Whole-mouth sealing is not achieved by this diagnostic.**
Do not substitute its holding result for a successful sealing-task score.

## Expanded contact and reset corrections

- The earlier central trial's lower jaw entered the table by about 1.5 mm.
  The asset's authored rigid contact stiffness was 2,500 N/m. Finite rigid
  stiffness is overridden separately from the film contact coefficient.
- At -0.55 rad the two handle boxes overlap. A regression test detects this
  and rejects the old aperture. The revised reset and lower limit use -0.27
  rad (roughly 12 mm tip opening, replacing the earlier 20 mm target).
- The clip begins at a table-resting height instead of falling about 14 mm
  during closure. The pouch's rest neck descends 13.7 mm over x=120..140 mm;
  its double-fold reset follows the same descent. The closed bottom and side
  seams remain joined. This is an authored initial geometry, not a robot fold.
- Clip contact skin is 0.1 mm and plane skin 0.5 mm in the current candidate.
  These are conservative numerical envelopes. Intrusion tests still use the
  nominal boxes and z=0 table, at the original 0.1 micrometre tolerance.
- A solver-owned collision pipeline uses `contact_matching='latest'`.
  Rigid-soft DAT jointly truncates rigid and film updates against reported
  contact division planes. Rigid and soft-self detection schedules must match.
  DAT does not itself certify global collision freedom; independent checks
  remain mandatory.
- CUDA graph replay captures an even number of substeps, with geometry checks
  inside every substep. Per-frame rings are concatenated including reset;
  per-state execution stamps verify cadence. No pose is pinned or rewritten
  after reset. Non-graph `--stop-on-geometry-error` freezes a first failure.

Short controls with the corrected reset passed all four geometry checks.
The long center run uses 100 iterations with detection every 10 iterations;
the lateral +3 mm trial uses 50 iterations and before/after-initialization
detection. These are individual diagnostics, not repeatability statistics or
a calibrated force/convergence study.

## Run

Use the isolated environment and pinned upstream checkout described in
`FILM_MATERIAL_STATUS.md`. Do not silently re-sync the CUDA-12 Warp wheel back
to the CUDA-13 PyPI build on this host's driver.

```sh
cd /data/robocurve/robosimstudio
export PYTHONPATH=. JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/data/robocurve/envs/rss/bin/python \
  /data/robocurve/tasks/close-snack-bag/experiments/native_film_pouch.py \
  /fresh/output --device cuda:0 --nx 80 --ny 64 --seconds 4.2 \
  --folded --clip --verify-every-step --record-every 1 --substeps 20 \
  --iterations 100 --collision-refresh 10 --rigid-soft-dat \
  --spring-damping .01 --contact-stiffness 1e5 --rigid-contact-stiffness 1e6 \
  --clip-contact-margin .0001 --ground-contact-margin .0005 \
  --clip-offset 0 0 -.0137 --mouth-drop .0137 \
  --initial-open-angle -.27 --open-limit -.27 --cuda-graph
/data/robocurve/envs/rss/bin/python \
  /data/robocurve/tasks/close-snack-bag/experiments/verify_native_hold.py /fresh/output
```

For first-failure diagnosis, replace `--cuda-graph` with
`--stop-on-geometry-error`. For the lateral trial also use `--iterations 50
--collision-refresh -1 --clip-offset 0 .003 -.0137`.

Optional independent rendering needs `pyrender==0.1.45` and `opencv-python-headless`
in addition to existing trimesh/imageio-ffmpeg. `render_native_clip.py` uses EGL
and recorded mesh/body coordinates. It does not interpolate or replace the
simulated motion; its replay runs at half speed. The orange film material is
for visibility, not a calibrated packaging appearance.

The renderer's tested PyOpenGL is 3.1.10. Install it with `uv pip install
--python /path/to/python --no-deps PyOpenGL==3.1.10` after pyrender; this deliberately
overrides pyrender's legacy 3.1.0 pin, whose EGL texture allocation failed on
this Python/NumPy environment. A renderer-only `np.infty = np.inf` compatibility
alias supports pyrender's old shadow-bounds code on NumPy 2. Neither change
affects the simulation process or its solver.

```sh
/data/robocurve/envs/rss/bin/python \
  /data/robocurve/tasks/close-snack-bag/experiments/render_native_clip.py /fresh/output
```

Full mesh recordings stay in `/data/robocurve/runs/` on Brev. Compact reports,
selected states and final video are preserved in Git. Keep the Brev instance
running, as requested by the user.
