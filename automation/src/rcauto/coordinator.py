"""Local supervisor: remote production, local preservation/review, then compute stop."""
import json
import logging
import shlex
import shutil
import time
from pathlib import Path

from .agents import Agent
from .catalog import GitHub, get_source
from .engine import compare_catalog, read
from .gates import GateError, manifest, verify_manifest
from .publish import Publisher
from .runtime import Brev
from .store import StateError, now, write_json


class Coordinator:
    def __init__(self, store, config, source_root):
        self.store, self.config, self.source_root = store, config, Path(source_root)
        self.brev = Brev(config)

    def sync(self):
        api = GitHub(self.store.root / "catalog-cache")
        snapshot = api.sync(self.config.get("upstream", "robocurve/benchmark-tasks"),
                            own_authors=self.config.get("own_authors", []))
        path = self.store.root / "catalog.json"
        write_json(self.store.root / "catalog-attempt.json", snapshot)
        if not snapshot["complete"]:
            raise GateError(f"catalog incomplete: {len(snapshot['missing'])} records; see catalog-attempt.json")
        write_json(path, snapshot)
        return snapshot

    def fresh_novelty(self, rid):
        root = self.store.run_dir(rid)
        catalog = self.sync()
        folders = list((root / "release" / "tasks").iterdir())
        final_spec = read(folders[0] / "task.json")
        candidate = {"task_spec": final_spec, "contract": read(root / "workspace/artifacts/contract.json")}
        return compare_catalog(Agent(model=self.config.get("model")), root / "workspace", candidate, catalog)

    def prepare_input(self, rid, catalog):
        root = self.store.run_dir(rid)
        workspace = root / "workspace"
        (workspace / "input").mkdir(parents=True, exist_ok=True)
        write_json(workspace / "input/request.json", self.store.get(rid))
        write_json(workspace / "input/catalog.json", catalog)
        write_json(workspace / "input/runtime.json", self.config["runtime"])
        for name in ["review-prompt"]:
            try:
                source = get_source("https://tasks.robocurve.org/" + name,
                                    workspace / "input/official-sources")
                (workspace / "input" / (name + ".txt")).write_text(source["text"])
            except (StateError, OSError, ValueError) as e:
                raise GateError("official review prompt unavailable") from e
        # Track exact current contest rules without changing the user's feature checkout.
        import base64
        rules = GitHub(self.store.root / "catalog-cache").get("repos/robocurve/benchmark-tasks/contents/AGENTS.md")
        (workspace / "input/repository-AGENTS.md").write_bytes(base64.b64decode(rules["content"]))

    def produce(self, rid, from_stage=None):
        root = self.store.run_dir(rid)
        remote_run = self.config["remote_root"] + "/worker-state/runs/" + rid
        b = self.brev
        with self.store.lock("instance-" + self.config["instance_id"]):
            started = False
            transferred = False
            try:
                self.store.update(rid, status="syncing-catalog", error=None, approval=None)
                catalog = self.sync()
                self.prepare_input(rid, catalog)
                self.store.update(rid, status="starting")
                b.start()
                started = True
                # A dedicated instance still may have an old worker. Never stop/redeploy over it.
                busy = b.ssh("flock -n " + shlex.quote(self.config["remote_root"] + "/worker-state/gpu-reservation.lock") +
                             " true || echo BUSY").strip()
                if "BUSY" in busy:
                    started = False
                    raise GateError("another remote GPU job owns this instance; leaving it untouched")
                b.deploy(self.source_root)
                b.ssh("mkdir -p " + shlex.quote(remote_run))
                b.rsync(str(root / "workspace") + "/", remote_run + "/workspace/")
                if (root / "revision-request.txt").exists():
                    b.rsync(root / "revision-request.txt", remote_run + "/workspace/input/")
                job = {"run": self.store.get(rid), "config": self.config, "from_stage": from_stage}
                write_json(root / "worker-job.json", job)
                b.rsync(root / "worker-job.json", remote_run + "/worker-job.json")
                b.ssh("rm -f " + shlex.quote(remote_run + "/cancel.json") + " " +
                      shlex.quote(remote_run + "/worker-status.json"))
                executable = self.config["remote_root"] + "/pipeline/.venv/bin/python"
                launch = ("nohup flock -n " + shlex.quote(self.config["remote_root"] + "/worker-state/gpu-reservation.lock") +
                          " " + shlex.quote(executable) + " -m rcauto.worker --job " +
                          shlex.quote(remote_run + "/worker-job.json") + " >" +
                          shlex.quote(remote_run + "/worker-launch.log") + " 2>&1 </dev/null &")
                b.ssh(launch)
                self.store.update(rid, status="working", remote_run=remote_run)
                deadline = time.monotonic() + self.config.get("worker_timeout", 28800)
                missing_since = time.monotonic()
                while time.monotonic() < deadline:
                    if (root / "cancel-requested.json").exists():
                        b.ssh("touch " + shlex.quote(remote_run + "/cancel.json"))
                    raw = b.ssh("cat " + shlex.quote(remote_run + "/worker-status.json") + " 2>/dev/null || true")
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
                    raise GateError("worker exceeded supervision deadline; requested cooperative pause")
                # Kernel lock is released only after worker finally writes its consistent SQLite snapshot.
                wait_until = time.monotonic() + 120
                while time.monotonic() < wait_until:
                    owned = b.ssh("flock -n " + shlex.quote(self.config["remote_root"] + "/worker-state/gpu-reservation.lock") +
                                  " true || echo BUSY")
                    if "BUSY" not in owned:
                        break
                    time.sleep(2)
                else:
                    raise GateError("worker still owns GPU; no unsafe stop")
                b.rsync(remote_run + "/", str(root) + "/", download=True)
                transferred = True
                if status["status"] != "pre-evaluation-ready":
                    raise GateError(status.get("error", "worker paused at stage boundary"))
                verify_manifest(root / "release", read(root / "release-manifest.json"))
                self.store.update(rid, status="pre-evaluation-ready")
                # The final one-episode evaluation is a distinct trusted process, never an agent tool.
                self.evaluate(rid)
                self.fresh_novelty(rid)
                self.report(rid, "awaiting-review")
                Publisher(self.store, self.config).checkpoint(rid)
                self.store.update(rid, status="awaiting-review")
            except (Exception, KeyboardInterrupt) as e:
                logging.getLogger(__name__).exception("production interrupted; preserving state")
                self.store.update(rid, status="blocked", error=str(e))
                self.report(rid, "blocked", str(e))
                if transferred:
                    try:
                        Publisher(self.store, self.config).checkpoint(rid)
                    except Exception as backup_error:
                        logging.getLogger(__name__).exception("checkpoint preservation failed")
                        self.store.update(rid, preservation_error=str(backup_error))
            finally:
                if started:
                    try:
                        # Never stop while our worker may still be executing. A disconnected Mac can resume supervision.
                        owned = b.ssh("flock -n " + shlex.quote(self.config["remote_root"] + "/worker-state/gpu-reservation.lock") +
                                      " true || echo BUSY")
                        if "BUSY" in owned:
                            self.store.update(rid, compute="running-worker-needs-reconciliation")
                        else:
                            if not transferred:
                                b.rsync(remote_run + "/", str(root) + "/", download=True)
                            self.store.backup(root / "controller-snapshot")
                            b.stop()
                            self.store.update(rid, compute="stopped")
                    except Exception as e:
                        logging.getLogger(__name__).exception("compute stop not confirmed")
                        self.store.update(rid, compute="stop-unknown", stop_error=str(e))

    def evaluate(self, rid, credentials=None):
        from .secrets import credentials as load_credentials
        credentials = credentials or load_credentials(self.store.root)
        if not credentials.get("DREAM_API_KEY"):
            raise GateError("production complete; Dream key missing: run rcauto auth dream then rcauto evaluate RUN")
        if not credentials.get("DREAM_PROVIDER_CAP_CONFIRMED"):
            raise GateError("Dream provider-side cumulative $20 cap must be confirmed in local auth before paid execution")
        root = self.store.run_dir(rid)
        verify_manifest(root / "release", read(root / "release-manifest.json"))
        task = next((root / "release/tasks").iterdir())
        spec = read(task / "task.json")
        contract = read(root / "workspace/artifacts/contract.json")
        remote = self.config["remote_root"] + "/worker-state/runs/" + rid
        job = {"task": remote + "/release/tasks/" + task.name + "/task.py", "output": remote + "/dream-result",
               "instruction": spec["instruction"], "max_steps": contract["run_steps"],
               "max_requests": self.config.get("dream_max_requests", 80)}
        if not (task / "task.py").is_file():
            raise GateError("Dream requires the validated task.py adapter")
        write_json(root / "evaluation-job.json", job)
        self.brev.rsync(root / "evaluation-job.json", remote + "/evaluation-job.json")
        with self.store.lock("dream-budget"):
            self.store.reserve_effect(rid, "dream", {"cap_usd": self.config["dream_episode_reserve_usd"],
                                                      "release_digest": read(root / "release-manifest.json")["digest"]},
                                      budget=self.config["dream_cumulative_usd"])
        cmd = ("ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y PYTHONPATH=" +
               shlex.quote(self.config["remote_root"] + "/pipeline/src") + " " +
               shlex.quote(self.config["runtime"]["python"]) + " -m rcauto.evaluation --job " +
               shlex.quote(remote + "/evaluation-job.json") + " >" + shlex.quote(remote + "/dream-result.log") + " 2>&1")
        self.brev.ssh(cmd, timeout=self.config.get("sim_timeout", 1800), stdin=json.dumps(credentials))
        self.brev.rsync(remote + "/dream-result/", str(root / "dream-result") + "/", download=True)
        logs = [p for p in (root / "dream-result").glob("*.json") if p.name != "attempt.json"]
        if len(logs) != 1:
            raise GateError("Dream result missing/ambiguous; attempt retained and will not repeat")
        log = read(logs[0])
        result = log.get("results", {})
        if result.get("total_trials") != 1 or log.get("status") != "success":
            raise GateError("Dream evaluation invalid/errored; no automatic reevaluation")
        score = result["metrics"]["task_score"]
        # Export video from the one existing log. This runs no policy and performs no new inference.
        exporter = str(Path(self.config["runtime"]["python"]).parent / "inspect-robots")
        self.brev.ssh(shlex.quote(exporter) + " video " + shlex.quote(job["output"] + "/" + logs[0].name), timeout=300)
        self.brev.rsync(remote + "/dream-result/", str(root / "dream-result") + "/", download=True)
        videos = sorted((root / "dream-result").rglob("*.mp4"))
        preferred = [p for p in videos if "top" in p.name.lower()]
        if not videos:
            raise GateError("Dream video missing; do not rerun policy")
        destination = root / "release/evidence/model_video.mp4"
        shutil.copy2((preferred or videos)[0], destination)
        mapping = read(root / "release/release-map.json")
        mapping["model_video"] = "evidence/model_video.mp4"
        write_json(root / "release/release-map.json", mapping)
        spec["scores"] = f"Dream-1 run: {score}; scripted demo: 1.0"
        write_json(task / "task.json", spec)
        write_json(root / "release-manifest.json", manifest(root / "release"))
        self.store.finish_effect(rid, "dream", {"score": score, "episode_count": 1,
                                                "billing": "reservation retained until provider usage reconciled"})

    def report(self, rid, status, error=None):
        root = self.store.run_dir(rid)
        run = self.store.get(rid)
        payload = {"run": rid, "status": status, "prompt": run["prompt"], "at": now(), "error": error,
                   "mode": run.get("mode", "production"), "dream": self.store.effect(rid, "dream"),
                   "release": read(root / "release-manifest.json") if (root / "release-manifest.json").exists() else None,
                   "review_path": "workspace/artifacts", "note": "local report; not a contest submission"}
        write_json(root / "report.json", payload)
        (root / "report.md").write_text(f"# {rid}\n\nStatus: **{status}**\n\n{run['prompt']}\n\n"
                                       f"{error or 'Review code, actual videos, measurements and industrial evidence before approving.'}\n\n"
                                       "- Release: `release/`\n- Evidence: `workspace/artifacts/`\n"
                                       "- This report is not submission approval.\n")
        return payload
