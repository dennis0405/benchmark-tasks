"""Local supervisor: remote production, local preservation/review, then compute stop."""

import html
import json
import logging
import shlex
import shutil
import time
from pathlib import Path

from .agents import Agent
from .catalog import GitHub, get_source
from .engine import Production, compare_catalog, read
from .gates import GateError, evaluation_identity, manifest, validate_media, verify_manifest
from .publish import Publisher
from .runtime import Brev
from .store import StateError, now, write_json


class Coordinator:
    def __init__(self, store, config, source_root):
        self.store, self.config, self.source_root = store, config, Path(source_root)
        self.brev = Brev(config)

    def sync(self):
        api = GitHub(self.store.root / "catalog-cache")
        snapshot = api.sync(
            self.config.get("upstream", "robocurve/benchmark-tasks"),
            own_authors=self.config.get("own_authors", []),
        )
        path = self.store.root / "catalog.json"
        write_json(self.store.root / "catalog-attempt.json", snapshot)
        if not snapshot["complete"]:
            raise GateError(
                f"catalog incomplete: {len(snapshot['missing'])} records; see catalog-attempt.json"
            )
        write_json(path, snapshot)
        return snapshot

    def fresh_novelty(self, rid):
        root = self.store.run_dir(rid)
        catalog = self.sync()
        folders = list((root / "release" / "tasks").iterdir())
        final_spec = read(folders[0] / "task.json")
        candidate = {
            "task_spec": final_spec,
            "contract": read(root / "workspace/artifacts/contract.json"),
        }
        agent = Agent(model=self.config.get("model"), deny_paths=[self.store.root / "private"])
        result = compare_catalog(agent, root / "workspace", candidate, catalog)
        write_json(root / "workspace/input/final-spec.json", candidate)
        Production(self.store, rid, agent, self.config).review(
            "final-novelty-critic",
            "Read input/final-spec.json and the actual final task under ../release/tasks/. "
            "Read artifacts/novelty.json and input/benchmark-sources.json. Verify final objective, "
            "physical skills and scoring against the latest PR catalog and all required benchmark "
            "originals. Recheck original URLs if changed/stale. Block overlaps, superficial variants "
            "or missing coverage. This is a CPU-only final check, not permission to run simulation.",
        )
        return result

    def prepare_input(self, rid, catalog):
        root = self.store.run_dir(rid)
        workspace = root / "workspace"
        (workspace / "input").mkdir(parents=True, exist_ok=True)
        write_json(workspace / "input/request.json", self.store.get(rid))
        write_json(workspace / "input/catalog.json", catalog)
        write_json(workspace / "input/runtime.json", self.config["runtime"])
        if self.store.get(rid).get("mode") == "selftest":
            return
        for name in ["review-prompt"]:
            try:
                source = get_source(
                    "https://tasks.robocurve.org/" + name, workspace / "input/official-sources"
                )
                (workspace / "input" / (name + ".txt")).write_text(source["text"])
            except (StateError, OSError, ValueError) as e:
                raise GateError("official review prompt unavailable") from e
        # Track exact current contest rules without changing the user's feature checkout.
        import base64

        rules = GitHub(self.store.root / "catalog-cache").get(
            "repos/robocurve/benchmark-tasks/contents/AGENTS.md"
        )
        (workspace / "input/repository-AGENTS.md").write_bytes(base64.b64decode(rules["content"]))

    def download_run(self, rid, remote_run):
        import tempfile

        root = self.store.run_dir(rid)
        staging = Path(tempfile.mkdtemp(prefix=".worker-download-", dir=root))
        try:
            self.brev.rsync(remote_run + "/", str(staging) + "/", download=True)
            if any(p.is_symlink() for p in staging.rglob("*")):
                raise GateError(
                    "worker output contains a symlink; refusing to follow it on the Mac"
                )
            for name in [
                "workspace",
                "release",
                "agent-runs",
                "logs",
                "snapshot",
                "release-manifest.json",
                "production-receipt.json",
                "worker-status.json",
            ]:
                source, target = staging / name, root / name
                if not source.exists():
                    continue
                if source.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
        finally:
            shutil.rmtree(staging)

    def produce(self, rid, from_stage=None):
        root = self.store.run_dir(rid)
        remote_run = self.config["remote_root"] + "/worker-state/runs/" + rid
        b = self.brev
        with self.store.lock("instance-" + self.config["instance_id"]):
            started = False
            transferred = False
            launched = False
            try:
                self.store.update(rid, status="syncing-catalog", error=None, approval=None)
                catalog = (
                    {"complete": True, "digest": "fixture", "records": [{"id": "fixture-1"}]}
                    if self.store.get(rid).get("mode") == "selftest"
                    else self.sync()
                )
                self.prepare_input(rid, catalog)
                self.store.update(rid, status="starting")
                started = True
                b.start()
                b.ssh("mkdir -p " + shlex.quote(self.config["remote_root"] + "/worker-state"))
                # A dedicated instance still may have an old worker. Never stop/redeploy over it.
                busy = b.ssh(
                    "flock -n "
                    + shlex.quote(self.config["remote_root"] + "/worker-state/gpu-reservation.lock")
                    + " true || echo BUSY"
                ).strip()
                attach = False
                if "BUSY" in busy:
                    raw_owner = b.ssh(
                        "cat "
                        + shlex.quote(self.config["remote_root"] + "/worker-owner.json")
                        + " 2>/dev/null || true"
                    )
                    try:
                        attach = json.loads(raw_owner).get("run") == rid
                    except ValueError:
                        attach = False
                    if not attach:
                        started = False
                        raise GateError(
                            "another remote GPU job owns this instance; leaving it untouched"
                        )
                    launched = True
                else:
                    if self.store.get(rid).get("remote_run"):
                        self.download_run(rid, remote_run)
                        self.prepare_input(rid, catalog)
                    b.deploy(self.source_root)
                    b.ssh("mkdir -p " + shlex.quote(remote_run))
                    b.rsync(str(root / "workspace") + "/", remote_run + "/workspace/")
                    if (root / "revision-request.txt").exists():
                        b.rsync(root / "revision-request.txt", remote_run + "/workspace/input/")
                    job = {
                        "run": self.store.get(rid),
                        "config": self.config,
                        "from_stage": from_stage,
                    }
                    write_json(root / "worker-job.json", job)
                    b.rsync(root / "worker-job.json", remote_run + "/worker-job.json")
                    b.ssh(
                        "rm -f "
                        + shlex.quote(remote_run + "/cancel.json")
                        + " "
                        + shlex.quote(remote_run + "/worker-status.json")
                    )
                    executable = self.config["remote_root"] + "/pipeline/.venv/bin/python"
                    launch = (
                        "nohup flock -n "
                        + shlex.quote(
                            self.config["remote_root"] + "/worker-state/gpu-reservation.lock"
                        )
                        + " "
                        + shlex.quote(executable)
                        + " -m rcauto.worker --job "
                        + shlex.quote(remote_run + "/worker-job.json")
                        + " >"
                        + shlex.quote(remote_run + "/worker-launch.log")
                        + " 2>&1 </dev/null &"
                    )
                    launched = True
                    b.ssh(launch)
                self.store.update(rid, status="working", remote_run=remote_run)
                deadline = time.monotonic() + self.config.get("worker_timeout", 28800)
                missing_since = time.monotonic()
                while time.monotonic() < deadline:
                    if (root / "cancel-requested.json").exists():
                        b.ssh("touch " + shlex.quote(remote_run + "/cancel.json"))
                    raw = b.ssh(
                        "cat "
                        + shlex.quote(remote_run + "/worker-status.json")
                        + " 2>/dev/null || true"
                    )
                    if raw.strip():
                        status = json.loads(raw)
                        self.store.update(rid, remote_status=status, stage=status.get("stage"))
                        if status["status"] in {"blocked", "paused", "pre-evaluation-ready"}:
                            break
                    elif time.monotonic() - missing_since > 120:
                        raise GateError("worker did not start; inspect remote worker-launch.log")
                    time.sleep(self.config.get("poll_seconds", 15))
                else:
                    b.ssh("touch " + shlex.quote(remote_run + "/cancel.json"))
                    raise GateError(
                        "worker exceeded supervision deadline; requested cooperative pause"
                    )
                # Kernel lock is released only after worker finally writes its consistent SQLite snapshot.
                wait_until = time.monotonic() + 120
                while time.monotonic() < wait_until:
                    owned = b.ssh(
                        "flock -n "
                        + shlex.quote(
                            self.config["remote_root"] + "/worker-state/gpu-reservation.lock"
                        )
                        + " true || echo BUSY"
                    )
                    if "BUSY" not in owned:
                        break
                    time.sleep(2)
                else:
                    raise GateError("worker still owns GPU; no unsafe stop")
                self.download_run(rid, remote_run)
                transferred = True
                if status["status"] != "pre-evaluation-ready":
                    raise GateError(status.get("error", "worker paused at stage boundary"))
                verify_manifest(root / "release", read(root / "release-manifest.json"))
                receipt = read(root / "production-receipt.json")
                if receipt["revision"] != self.store.get(rid).get("revision", 0):
                    raise GateError(
                        "worker completed an older revision; new request remains pending"
                    )
                self.store.update(rid, resume_from_stage=None)
                self.store.update(rid, status="pre-evaluation-ready")
                if self.store.get(rid).get("mode") == "selftest":
                    self.report(rid, "selftest-complete")
                    self.store.update(rid, status="selftest-complete")
                    return
                # The final one-episode evaluation is a distinct trusted process, never an agent tool.
                if self.store.effect(rid, "dream"):
                    self.reuse_evaluation(rid)
                else:
                    self.evaluate(rid)
                self.report(rid, "final-novelty-pending")
                Publisher(self.store, self.config).checkpoint(rid)
                self.store.backup(root / "controller-snapshot")
                b.stop()
                started = False
                self.store.update(rid, compute="stopped")
                self.finalize(rid)
            except (Exception, KeyboardInterrupt) as e:
                logging.getLogger(__name__).exception("production interrupted; preserving state")
                self.store.update(rid, status="blocked", error=str(e))
                self.report(rid, "blocked", str(e))
                if transferred and self.store.get(rid).get("mode") != "selftest":
                    try:
                        Publisher(self.store, self.config).checkpoint(rid)
                    except Exception as backup_error:
                        logging.getLogger(__name__).exception("checkpoint preservation failed")
                        self.store.update(rid, preservation_error=str(backup_error))
            finally:
                if started:
                    try:
                        if not launched:
                            self.store.backup(root / "controller-snapshot")
                            b.stop()
                            self.store.update(rid, compute="stopped")
                            continue_cleanup = False
                        else:
                            continue_cleanup = True
                        # Never stop while our worker may still be executing. A disconnected Mac can resume supervision.
                        if continue_cleanup:
                            owned = b.ssh(
                                "flock -n "
                                + shlex.quote(
                                    self.config["remote_root"]
                                    + "/worker-state/gpu-reservation.lock"
                                )
                                + " true || echo BUSY"
                            )
                            if "BUSY" in owned:
                                self.store.update(
                                    rid, compute="running-worker-needs-reconciliation"
                                )
                            else:
                                if not transferred:
                                    self.download_run(rid, remote_run)
                                self.store.backup(root / "controller-snapshot")
                                b.stop()
                                self.store.update(rid, compute="stopped")
                    except Exception as e:
                        logging.getLogger(__name__).exception("compute stop not confirmed")
                        self.store.update(rid, compute="stop-unknown", stop_error=str(e))

    def evaluate(self, rid, credentials=None):
        if self.store.get(rid).get("mode") == "selftest":
            raise GateError("selftest can never call Dream")
        from .secrets import credentials as load_credentials

        credentials = credentials or load_credentials(self.store.root)
        if not credentials.get("DREAM_API_KEY"):
            raise GateError(
                "production complete; Dream key missing: run rcauto auth dream then rcauto evaluate RUN"
            )
        root = self.store.run_dir(rid)
        verify_manifest(root / "release", read(root / "release-manifest.json"))
        task = next((root / "release/tasks").iterdir())
        spec = read(task / "task.json")
        contract = read(root / "workspace/artifacts/contract.json")
        remote = self.config["remote_root"] + "/worker-state/runs/" + rid
        job = {
            "task": remote + "/release/tasks/" + task.name + "/task.py",
            "output": remote + "/dream-result",
            "instruction": spec["instruction"],
            "max_steps": contract["run_steps"],
            "max_requests": self.config.get("dream_max_requests", 80),
            "run": rid,
            "owner_path": self.config["remote_root"] + "/worker-owner.json",
        }
        if not (task / "task.py").is_file():
            raise GateError("Dream requires the validated task.py adapter")
        write_json(root / "evaluation-job.json", job)
        self.brev.rsync(root / "evaluation-job.json", remote + "/evaluation-job.json")
        with self.store.lock("dream-budget"):
            self.store.reserve_effect(
                rid,
                "dream",
                {
                    "cap_usd": self.config["dream_episode_reserve_usd"],
                    "release_digest": read(root / "release-manifest.json")["digest"],
                    "input_digest": evaluation_identity(root / "release"),
                },
                budget=self.config["dream_cumulative_usd"],
            )
        cmd = (
            "flock -n "
            + shlex.quote(self.config["remote_root"] + "/worker-state/gpu-reservation.lock")
            + " env ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y PYTHONPATH="
            + shlex.quote(self.config["remote_root"] + "/pipeline/src")
            + " "
            + shlex.quote(self.config["runtime"]["python"])
            + " -m rcauto.evaluation --job "
            + shlex.quote(remote + "/evaluation-job.json")
            + " >"
            + shlex.quote(remote + "/dream-result.log")
            + " 2>&1"
        )
        dream_secrets = {
            k: v
            for k, v in credentials.items()
            if k in {"DREAM_API_KEY", "DREAM_GATEWAY_URL", "DREAM_API_URL", "DREAM_DATA_URL"}
        }
        try:
            self.brev.ssh(
                cmd, timeout=self.config.get("sim_timeout", 1800), stdin=json.dumps(dream_secrets)
            )
        finally:
            self.brev.rsync(
                remote + "/dream-result/", str(root / "dream-result") + "/", download=True
            )
        logs = [p for p in (root / "dream-result").glob("*.json") if p.name != "attempt.json"]
        if len(logs) != 1:
            raise GateError("Dream result missing/ambiguous; attempt retained and will not repeat")
        log = read(logs[0])
        result = log.get("results", {})
        if result.get("total_trials") != 1 or log.get("status") != "success":
            raise GateError("Dream evaluation invalid/errored; no automatic reevaluation")
        score = result["metrics"]["task_score"]
        import math

        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise GateError("invalid Dream task score")
        # Export video from the one existing log. This runs no policy and performs no new inference.
        exporter = str(Path(self.config["runtime"]["python"]).parent / "inspect-robots")
        self.brev.ssh(
            shlex.quote(exporter) + " video " + shlex.quote(job["output"] + "/" + logs[0].name),
            timeout=300,
        )
        self.brev.rsync(remote + "/dream-result/", str(root / "dream-result") + "/", download=True)
        videos = sorted((root / "dream-result").rglob("*.mp4"))
        preferred = [p for p in videos if "top" in p.name.lower()]
        if not videos:
            raise GateError("Dream video missing; do not rerun policy")
        destination = root / "release/evidence/model_video.mp4"
        shutil.copy2((preferred or videos)[0], destination)
        validate_media(destination, "video")
        mapping = read(root / "release/release-map.json")
        mapping["model_video"] = "evidence/model_video.mp4"
        write_json(root / "release/release-map.json", mapping)
        spec["scores"] = f"Dream-1 run: {score}; scripted demo: 1.0"
        write_json(task / "task.json", spec)
        write_json(root / "release-manifest.json", manifest(root / "release"))
        self.store.finish_effect(
            rid,
            "dream",
            {
                "score": score,
                "episode_count": 1,
                "billing": "reservation retained until provider usage reconciled",
            },
        )

    def reuse_evaluation(self, rid):
        row = self.store.effect(rid, "dream")
        data = json.loads(row["data"])
        root = self.store.run_dir(rid)
        if row["status"] != "finished" or "score" not in data:
            raise GateError(
                "Dream attempt unknown/failed; revision preserved but evaluation is never repeated"
            )
        if evaluation_identity(root / "release") != data.get("input_digest"):
            raise GateError(
                "revision changed evaluation inputs; old Dream result is invalid, no automatic reevaluation"
            )
        videos = sorted((root / "dream-result").rglob("*.mp4"))
        preferred = [p for p in videos if "top" in p.name.lower()]
        if not videos:
            raise GateError("original model video unavailable; restore it, never rerun inference")
        shutil.copy2((preferred or videos)[0], root / "release/evidence/model_video.mp4")
        mapping = read(root / "release/release-map.json")
        mapping["model_video"] = "evidence/model_video.mp4"
        write_json(root / "release/release-map.json", mapping)
        task = next((root / "release/tasks").iterdir())
        spec = read(task / "task.json")
        spec["scores"] = f"Dream-1 run: {data['score']}; scripted demo: 1.0"
        write_json(task / "task.json", spec)
        write_json(root / "release-manifest.json", manifest(root / "release"))

    def finalize(self, rid):
        """CPU-only recovery after a valid final episode; never repeats paid inference."""
        row = self.store.effect(rid, "dream")
        if not row or row["status"] != "finished" or "score" not in json.loads(row["data"]):
            raise GateError("valid finished Dream result required before finalization")
        root = self.store.run_dir(rid)
        receipt = read(root / "production-receipt.json")
        if receipt["revision"] != self.store.get(rid).get("revision", 0):
            raise GateError(
                "requested revision has not completed; previous release cannot be finalized"
            )
        if evaluation_identity(root / "release") != json.loads(row["data"]).get("input_digest"):
            raise GateError("final task does not match the recorded Dream inputs")
        verify_manifest(root / "release", read(root / "release-manifest.json"))
        validate_media(root / "release/evidence/model_video.mp4", "video")
        self.fresh_novelty(rid)
        self.report(rid, "awaiting-review")
        Publisher(self.store, self.config).checkpoint(rid)
        self.store.update(rid, status="awaiting-review")
        return self.store.get(rid)

    def report(self, rid, status, error=None):
        root = self.store.run_dir(rid)
        run = self.store.get(rid)
        payload = {
            "run": rid,
            "status": status,
            "prompt": run["prompt"],
            "at": now(),
            "error": error,
            "mode": run.get("mode", "production"),
            "dream": self.store.effect(rid, "dream"),
            "release": read(root / "release-manifest.json")
            if (root / "release-manifest.json").exists()
            else None,
            "review_path": "workspace/artifacts",
            "note": "local report; not a contest submission",
        }
        artifact_root = root / "workspace/artifacts"
        for key in [
            "selected",
            "contract",
            "novelty",
            "physics-reviewer",
            "demo-scoring-critic",
            "release-critic",
        ]:
            p = artifact_root / (key + ".json")
            if p.exists():
                payload[key] = read(p)
        evidence = root / "workspace/input/sources.json"
        if evidence.exists():
            payload["sources"] = [
                {k: s[k] for k in ("id", "title", "publisher", "url", "sha256") if k in s}
                for s in read(evidence).values()
            ]
        task_root = root / "release/tasks"
        if task_root.exists():
            tasks = list(task_root.iterdir())
            if len(tasks) == 1:
                payload["task"] = read(tasks[0] / "task.json")
        write_json(root / "report.json", payload)
        (root / "report.md").write_text(
            f"# {rid}\n\nStatus: **{status}**\n\n{run['prompt']}\n\n"
            f"{error or 'Review code, actual videos, measurements and industrial evidence before approving.'}\n\n"
            "- Release: `release/`\n- Evidence: `workspace/artifacts/`\n"
            "- This report is not submission approval.\n"
        )
        escape = lambda s: html.escape(str(s), quote=True)
        videos, image = "", ""
        mapping_path = root / "release/release-map.json"
        if mapping_path.exists():
            mapping = read(mapping_path)
            for label in ["demo_video", "model_video"]:
                if label in mapping:
                    videos += f'<h2>{escape(label)}</h2><video controls preload="metadata" src="release/{escape(mapping[label])}"></video>'
            if mapping.get("screenshot"):
                image = f'<h2>Initial setup</h2><img src="release/{escape(mapping["screenshot"])}" alt="setup">'
        tables = "".join(
            f"<h2>{escape(key)}</h2><pre>{escape(json.dumps(payload[key], indent=2, ensure_ascii=False))}</pre>"
            for key in [
                "task",
                "selected",
                "contract",
                "sources",
                "physics-reviewer",
                "demo-scoring-critic",
                "release-critic",
            ]
            if key in payload
        )
        page = (
            '<!doctype html><meta charset="utf-8"><title>RoboCurve review</title>'
            "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; img-src 'self' file:; media-src 'self' file:; style-src 'unsafe-inline'\">"
            "<style>body{max-width:1000px;margin:32px auto;font:16px system-ui;line-height:1.6;padding:0 20px}"
            "pre{white-space:pre-wrap;background:#f1f4f6;padding:16px}video,img{max-width:100%}</style>"
            f"<h1>{escape(rid)}</h1><p>Status: <strong>{escape(status)}</strong></p><p>{escape(run['prompt'])}</p>"
            f"<p>{escape(error or 'Review actual evidence before approving this exact release digest.')}</p>"
            f"<p>Release digest: {escape((payload.get('release') or {}).get('digest', 'none'))}</p>"
            + image
            + videos
            + tables
            + "<p>Approval and submission require separate CLI commands.</p>"
        )
        (root / "report.html").write_text(page)
        return payload
