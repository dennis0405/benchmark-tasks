---
name: robocurve-workflow
description: Run the local RoboCurve production CLI to generate a task from a user prompt, inspect progress, resume revisions, or submit an explicitly approved result. Use for operating the workflow, not for implementing the automation code or continuing separate snack-bag experiments.
---

# RoboCurve workflow

The execution authority is `rcauto`, implemented at
`/Users/dennis0405/Workspace/SNU-PI/robocurve/automation`.
Use `rcauto` if installed; otherwise use `uv run --project` with that absolute path.
Read the project's README when invocation or an exception requires more detail.

## Start and monitor

A user request to create a new industrial task is the production prompt. Preserve its
constraints verbatim in a local prompt file and call `rcauto start --prompt-file PATH
--detach`. Use a structured subprocess argv, never interpolate prompt text into a shell.
Show the returned run ID and supervisor log. Follow `rcauto status RUN` and the log until
review-ready, blocked, paused or a genuine handoff. Do not trigger new runs merely to
answer status questions, view examples, or implement the pipeline itself.

The CLI controls the registered Brev instance, sequential Codex roles, diagnostics,
Dream final evaluation, fork checkpoints and stop. Do not duplicate those decisions in
this skill or operate unrelated instances. No scheduler is involved.

## Review and revision

Open the run's `report.html`, actual videos, measurements and `release-manifest.json`.
Distinguish real production from `selftest`, native-runtime validation from Docker GPU
access, and a created PR from official receipt or acceptance. Surface missing industrial
sources, novelty coverage, physical validity or model results instead of filling them in.

For a revision, call `rcauto resume RUN --request TEXT` (or an explicitly justified
`--from-stage`). Preserve the same run/branch. A Dream attempt, including an unknown or
failed attempt, must not be reset or automatically repeated. Presentation edits may reuse
the original result only when the evaluation input digest is unchanged; physics/input
edits block finalization. Use `reconcile` for recorded
external effects; inspect evidence before choosing a recovery.

## Human approval and submission

Submission approval must refer to the specific reviewed result and its digest. A generic
positive reply, a start request, or an SDK action-guard `approval` event is insufficient.
After explicit approval, call `rcauto approve RUN --digest DIGEST --actor USER
--statement STATEMENT`, then `rcauto submit RUN --approval APPROVAL_ID` if the user
approved actual submission. The CLI rechecks current novelty without starting the GPU.
Read `rcauto receipt RUN`; report the actual PR/bot/board state, never infer acceptance.

## Credentials and exceptions

Use `rcauto doctor` for readiness. Credentials are entered by the user in their terminal
through `rcauto auth board` and `rcauto auth dream`; do not ask
them to paste passwords or tokens into chat. Agent roles never receive team/Dream keys.
Do not run these interactive auth commands through a tool that would capture secrets.
Dream reserves against the cumulative $20 ledger before any SDK initialization. Until
provider billing is reconciled, the conservative reservation remains charged. Do not
invent a per-call price or claim the local ledger is a provider billing limit.

At an exception, show the recorded cause, preservation state and actual compute state.
`stop-requested`, `STOPPING`, `STOPPED` and `stop-unknown` are different states.
The Mac supervisor must remain online; detached execution prevents terminal closure
from stopping it, and macOS caffeinate prevents idle sleep while it runs.
