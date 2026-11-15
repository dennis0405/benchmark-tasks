# Native industrial pilot: recovery checkpoint

2026-11-15 KST. Run `20261114T031727-dc73f2bb`, result branch
`automation/20261114T031727-dc73f2bb`.

The current candidate is removal and recovery of four installation tangs from
M2.5 HELICOIL inserts in an ODRI-informed actuator case. Connector work is excluded.
This is an **unfinished prototype**, not a submission-ready task.

Later on 2026-11-15 the user approved a replacement L40S and asked to broaden
the search, prioritizing actual native prototypes and a completed task over
repeated review loops. The HELICOIL files remain archived; continuing that
candidate is optional. Mechanical inspection is preferred, with broader robot
assembly and data-center service also in scope. Connector work stays excluded.

| Area | Evidence and limit |
| --- | --- |
| Industrial sources | Pinned ODRI preparation procedure, photographs, and Böllhoff manual inspected. Geometry and fracture calibration still need validation. |
| Novelty | Historical selection comparison passed with documented inactive uncertainty. The actual built task still needs the final fresh all-category/team/six-suite comparison. |
| Contract | Independently reviewed outcome scoring; 35 physics checks and 15 controls. Digest `acc74d5535d48dea995f5b2948e92d8e0aabfe0936053ee9e30571626f1b96da`. |
| Native contact probe | 240 Hz GPU probe recorded 152 tensor contact rows, zero legacy event rows, zero callback errors. This proves contact observation only. |
| Contact force decoding | The probe's first force was a one-element scalar `[30.411045...]`, not a world vector. Scalar-times-normal decoding was repaired in source; the repaired complete task has not passed a native rerun. Actor sign and compound-shape indexing also need verification. |
| Native mechanics | Prior raw states showed up to 22.094 mm case-relative insert displacement and 75.329 degrees rotation. End-state return is not stability. No completed physics gate. |
| Public YAM integration | Primitive-only TaskFile declarations do not yet instantiate the shared native mechanism. A task-local lazy factory adapter and actual READY/reset/camera smoke remain necessary. |
| Full YAM / Dream / submission | No full-score task demo, final Dream episode, or external contest submission. |
| Compute | Previous `th9ih02z4` remains intact after START FAILED. Replacement `robocurve-automation-r2` / `ffss1b4z8` is RUNNING with verified L40S/data disk, restored pinned packages and ChatGPT login. The native stock `stack_cubes` example scored1.0; this is environment validation, not a new industrial task. |

## Lessons now implemented in the controller

- Auxiliary native API probes have their own numerical checks. The **final**
  diagnostic must still satisfy every unchanged frozen physics check. A passing
  preflight cannot replace a missing or weakened aggregate.
- Dream action chunks contain at most 48 controls in the installed SDK. The old
  arbitrary 80-request limit could truncate a 7,200-step episode. Requests within
  the single final episode are now uncapped by default; the episode guard,
  control-step limit, cumulative ledger and prohibition on reevaluation remain.
- Brev starts in detached CLI mode so the controller can see terminal provider
  failures promptly. A stop requested during startup prevents deployment and
  worker launch. A blank interrupt message retains its exception type.
- Before a certified release exists for the current revision, checkpoint the
  working source under `automation-results/<run>/task-prototype/`, with an explicit
  unvalidated prototype manifest. Earlier checkpoints preserved measurements and
  reviews but omitted this pre-export source; they were not complete code backups.
  Successful verified pushes clear the stale preservation-error flag.

## Recovery inputs

- The Mac retains the complete private run directory, including original source
  evidence and agent history. Do not publish whole manuals or credentials.
- The fork checkpoint contains prototype source, contract/reviews, measurements,
  and logs. These files are evidence of progress and failures, not certification.
- Continue the same run and result branch using
  `requests/2026-11-15-industrial-task-native-runtime-v24.txt`. Its exact text must
  be passed to `rcauto resume --request TEXT --from-stage environment --detach`.
- If the user chooses a replacement VM, first register its verified name/ID and
  restore the pinned runtime on that VM. Keep the failed instance intact. Do not
  operate the separate snack-bag VM or checkout.
- Resume native work only after GPU/SSH readiness. Finish the shared scene,
  physics, two real YAM demos, all controls/replay/media/portability, then the
  single final Dream episode and fresh actual-built novelty review. Actual
  submission still requires approval of the specific completed bundle.
