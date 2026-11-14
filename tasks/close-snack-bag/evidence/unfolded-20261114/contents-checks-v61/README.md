# Contents component checks; no completed sealing episode

- `brittle-tracked-rest-v45`: 2,400 observed physics steps, all twelve cracker seams intact, no observed solid/film crossings or contact-capacity overflows. Contents-only assessment passes initial containment and observation consistency. This is a resting scene, not folding or retention.
- `brittle-gap-weak-v51`: 0.5 N pull, maximum anchor separation 51.37 micrometres; seam remains enabled.
- `brittle-gap-strong-v52`: 5 N pull, separation crosses the uncalibrated 0.4 mm threshold at physical step 2. Seam remains disabled after unloading. These are numerical controls, not measured cracker strength.
- `thin-film-seal-v48`: fails the upright folding handoff at 148 control steps. All 17,760 physics steps have zero observed geometric hits. It does not reach clip placement or release.
- `focused-suite-v50.log`: 32 focused tests passed.
- `body-anchor-plan-v50.json`: offline IK only, not dynamic contact validation.
- `host-ik-comparison-v60.json`: small pose probes agree. First device solve includes compilation; cached solves are already about 1 ms in either scope. This does not establish a host-FK speedup. Upstream may select Pyroki in both scopes.

Only initial/final states are in `selected-states.npz`; full raw recordings remain on Brev under `/data/robocurve/runs`. Do not run the full trajectory assessor against these selected states. Full traces, results and source archives preserve the measured scope. The source archives were checked against each isolated execution manifest. The two tensile fixtures were run directly, without isolated source archives.

The current evaluator sources are preserved separately in `assessment-sources`, with SHA-256 hashes. The evaluator is experimental and does not emit a public Arena submission score.
