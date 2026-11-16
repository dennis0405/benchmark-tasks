# Task-local native YAM scene adapter

The pinned primitive TaskFile loader has no arbitrary USD or pre-build scene hook.
Its Box/Cylinder/Sphere/library Asset declarations cannot represent the complete
articulated DRG fixture. Installed packages must remain unchanged and scoring
must not secretly construct or actuate the scene.

For tasks needing a native scene, keep public `task.py` declarations/scoring and
add `native-runtime.json` alongside them:

```json
{
  "module": "native_runtime.py",
  "class_name": "NativeYam",
  "sha256": "the SHA-256 of the actual module bytes"
}
```

`NativeYam` subclasses the pinned
`yam_arena.inspect_embodiment.YamArenaEmbodiment`. It overrides `_ensure_sim`
to build the actual scene with the same YAM rig and may manage scene cleanup.
It must inherit `__init__`, `reset`, `step`, `_observation` and `_task_step`
without overriding them. The native scene and callback bridge still need real
physics, reset, demo, scoring and portability qualification. A descriptor or
class definition proves none of those.

The controller checks the descriptor and module digest before a Dream budget
reservation. The trusted entrypoint checks it against the validated job again,
then uses Inspect Robots' public embodiment registration decorator under
`rcauto_yam_native`. It verifies parent class and method identities. The fixed
20 Hz control, camera settings, task path, Dream policy and single-episode guard
remain in the trusted entrypoint. Tasks without a descriptor use the original
`yam_arena` path.

Imports disable bytecode writes in the frozen release. Task-local helper modules
remain importable from the task directory. Include every helper and asset in
the release manifest and evaluation identity; the descriptor's own hash only
covers its named module. This interface guard is not a sandbox or a substitute
for source review.

Standalone reproduction must register the same class through Inspect Robots'
public decorator (a task-local launcher or documented package entrypoint), not
patch the installed SDK. A native dry run must use that same class and prove
the inherited action/observation/scoring path before the one paid episode.

Implementation validation: ten CPU regression cases cover unchanged primitive
tasks, exact module/job identity, task-relative paths, no bytecode writes and
rejection of overrides to all five preserved methods. GPU/native integration
is pending task construction and must not be reported as validated by these
tests.
