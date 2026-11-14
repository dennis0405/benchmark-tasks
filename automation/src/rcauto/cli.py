"""CLI execution boundary; skills only route user intent to these commands."""

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .coordinator import Coordinator
from .engine import Production, approve, read
from .publish import Publisher
from .runtime import Brev
from .secrets import credentials, save
from .store import StateError, Store, now, write_json

ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(
        description="Prompt-driven RoboCurve production; submission requires human approval."
    )
    p.add_argument("--config", default=str(ROOT / "config/production.json"))
    p.add_argument(
        "--state", default=os.environ.get("RCAUTO_STATE", str(Path.home() / ".local/state/rcauto"))
    )
    sub = p.add_subparsers(dest="command", required=True)
    start = sub.add_parser(
        "start", help="create a run and start production on the registered Brev instance"
    )
    group = start.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompt-file")
    start.add_argument("--detach", action="store_true")
    status = sub.add_parser("status", help="local status; does not start GPU")
    status.add_argument("run", nargs="?")
    resume = sub.add_parser("resume", help="resume the same run/branch or apply a revision")
    resume.add_argument("run")
    resume.add_argument("--request")
    resume.add_argument("--from-stage", choices=Production.STAGES)
    resume.add_argument("--detach", action="store_true")
    stop = sub.add_parser("stop", help="checkpoint at a safe stage boundary, then stop GPU")
    stop.add_argument("run")
    approval = sub.add_parser(
        "approve", help="record the user's explicit approval of the reviewed bundle"
    )
    approval.add_argument("run")
    approval.add_argument("--actor", required=True)
    approval.add_argument("--statement", required=True)
    approval.add_argument("--digest", required=True)
    submit = sub.add_parser(
        "submit", help="refresh novelty, upload media and open one approved PR; no GPU start"
    )
    submit.add_argument("run")
    submit.add_argument("--approval", required=True)
    evaluate = sub.add_parser(
        "evaluate", help="perform the reserved final Dream episode after production is complete"
    )
    evaluate.add_argument("run")
    finalize = sub.add_parser(
        "finalize",
        help="recover final novelty/report/fork preservation without GPU or reevaluation",
    )
    finalize.add_argument("run")
    auth = sub.add_parser("auth", help="enter credentials locally; never paste them into chat")
    auth.add_argument("service", choices=["board", "dream"])
    doctor = sub.add_parser(
        "doctor", help="check tools/auth; --remote tests the dedicated instance and stops it"
    )
    doctor.add_argument("--remote", action="store_true")
    doctor.add_argument("--agent", action="store_true")
    doctor.add_argument(
        "--sim",
        action="store_true",
        help="run upstream scripted YAM baseline, not a new task or Dream",
    )
    sub.add_parser(
        "catalog-sync", help="sync all PR metadata and active task revisions without GPU"
    )
    selftest = sub.add_parser("selftest", help="fixture run; cannot evaluate or submit")
    selftest.add_argument(
        "--remote",
        action="store_true",
        help="exercise real Brev/SSH/detached worker/stop with synthetic fixtures",
    )
    receipt = sub.add_parser(
        "receipt", help="read actual PR comments; PR creation is not contest acceptance"
    )
    receipt.add_argument("run")
    reconcile = sub.add_parser(
        "reconcile", help="inspect unknown external effects without retrying them"
    )
    reconcile.add_argument("run")
    reconcile.add_argument("--dream-actual-usd", type=float)
    reconcile.add_argument("--billing-reference")
    supervisor = sub.add_parser("_supervise", help=argparse.SUPPRESS)
    supervisor.add_argument("run")
    supervisor.add_argument("--from-stage")
    return p


def dispatch(args, store, config):
    coordinator = Coordinator(store, config, ROOT)
    cmd = args.command
    if cmd == "status":
        return store.get(args.run) if args.run else store.list()
    if cmd == "start":
        prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
        run = store.create(prompt, config["instance"], mode="production", config=config)
        return execute_or_detach(args, store, coordinator, run["id"])
    if cmd == "_supervise":
        coordinator.produce(args.run, args.from_stage)
        return store.get(args.run)
    if cmd == "resume":
        run = store.get(args.run)
        if run.get("mode") == "selftest":
            raise StateError("selftest cannot resume as production")
        if (
            store.effect(args.run, "dream")
            and not args.request
            and not args.from_stage
            and not run.get("resume_from_stage")
        ):
            return coordinator.finalize(args.run)
        root = store.run_dir(args.run)
        if args.request or args.from_stage:
            store.update(
                args.run,
                revision=run.get("revision", 0) + 1,
                revision_request=(
                    args.request if args.request is not None else run.get("revision_request")
                ),
                approval=None,
            )
            store.update(
                args.run,
                resume_from_stage=args.from_stage
                or ("environment" if store.effect(args.run, "dream") else "research"),
            )
        if args.request:
            (root / "revision-request.txt").write_text(args.request)
        if (root / "cancel-requested.json").exists():
            (root / "cancel-requested.json").unlink()
        return execute_or_detach(
            args,
            store,
            coordinator,
            args.run,
            args.from_stage or store.get(args.run).get("resume_from_stage"),
        )
    if cmd == "stop":
        run = store.get(args.run)
        write_json(
            store.run_dir(args.run) / "cancel-requested.json", {"at": now(), "requested": "human"}
        )
        if run.get("status") not in {"working", "starting", "syncing-catalog"}:
            return {
                "run": args.run,
                "status": "pause-requested",
                "note": "no live supervisor; resume reconciliation required before stop",
            }
        return {
            "run": args.run,
            "status": "pause-requested",
            "note": "supervisor stops after safe stage boundary and preservation",
        }
    if cmd == "approve":
        m = read(store.run_dir(args.run) / "release-manifest.json")
        if args.digest != m["digest"]:
            raise StateError("reviewed digest mismatch")
        return approve(store, args.run, args.actor, args.statement)
    if cmd == "submit":
        with store.lock("publish-" + args.run):
            return Publisher(store, config).submit(
                args.run,
                args.approval,
                fresh_novelty=lambda: coordinator.fresh_novelty(args.run),
                credentials=credentials(store.root),
            )
    if cmd == "evaluate":
        root = store.run_dir(args.run)
        if store.effect(args.run, "dream"):
            raise StateError("Dream already attempted; use reconcile")
        if (
            store.get(args.run).get("mode") == "selftest"
            or not (root / "release-manifest.json").exists()
        ):
            raise StateError("validated production bundle required")
        with store.lock("instance-" + config["instance_id"]):
            coordinator.brev.start()
            try:
                coordinator.brev.deploy(ROOT)
                coordinator.evaluate(args.run)
                coordinator.report(args.run, "final-novelty-pending")
                Publisher(store, config).checkpoint(args.run)
            finally:
                import shlex

                busy = coordinator.brev.ssh(
                    "flock -n "
                    + shlex.quote(config["remote_root"] + "/worker-state/gpu-reservation.lock")
                    + " true || echo BUSY"
                )
                if "BUSY" in busy:
                    store.update(args.run, compute="running-evaluation-needs-reconciliation")
                else:
                    coordinator.brev.stop()
                    store.update(args.run, compute="stopped")
        return coordinator.finalize(args.run)
    if cmd == "finalize":
        with store.lock("publish-" + args.run):
            return coordinator.finalize(args.run)
    if cmd == "auth":
        if args.service == "board":
            team = input("Team code: ").strip()
            token = getpass.getpass("Team access token (hidden): ")
            import httpx

            with httpx.Client(headers={"User-Agent": "robocurve-automation/0.1"}) as client:
                result = client.post(
                    "https://tasks.robocurve.org/api/login", json={"code": team, "token": token}
                )
                result.raise_for_status()
            save(store.root, {"ROBOCURVE_TEAM": team, "ROBOCURVE_TOKEN": token})
        else:
            key = getpass.getpass("DREAM_API_KEY (hidden, no inference sent): ")
            if not key.strip():
                raise StateError("empty Dream key")
            save(store.root, {"DREAM_API_KEY": key.strip()})
        return {"service": args.service, "configured": True, "paid_requests": 0}
    if cmd == "catalog-sync":
        result = coordinator.sync()
        return {k: result[k] for k in ["scope", "complete", "digest", "completed_at"]} | {
            "records": len(result["records"])
        }
    if cmd == "doctor":
        if (args.agent or args.sim) and not args.remote:
            raise StateError("--agent/--sim require --remote")
        result = {
            "tools": {
                x: shutil.which(x)
                for x in ["brev", "git", "gh", "uv", "codex", "ffmpeg", "ffprobe"]
            },
            "credentials": {k: bool(v) for k, v in credentials(store.root).items()},
            "pipeline": "implemented; industrial pilot only starts with user prompt",
        }
        if args.remote:
            b = Brev(config)
            b.start()
            try:
                result["deploy"] = b.deploy(ROOT)[-1000:]
                result["remote"] = b.ssh(
                    "nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; "
                    "/home/ubuntu/.local/bin/codex login status"
                )
                if args.agent:
                    remote_python = config["remote_root"] + "/pipeline/.venv/bin/python"
                    script = (
                        "from pathlib import Path;from rcauto.agents import Agent;import json;"
                        "p=Path('/data/robocurve-automation/connection-smoke/workspace');p.mkdir(parents=True,exist_ok=True);"
                        "(p/'input').mkdir(exist_ok=True);(p/'artifacts').mkdir(exist_ok=True);"
                        "probe=p/'probe.txt';probe.unlink(missing_ok=True);"
                        "r=Agent(timeout=120).run('connection-smoke','Write probe.txt with exactly rcauto-connection-ok. "
                        "Verify its contents using a shell tool, then return ok=true.',p,"
                        "{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False},writable=True);"
                        "assert probe.read_text().strip()=='rcauto-connection-ok';"
                        "print(json.dumps({'response':r,'tool_write_verified':True}))"
                    )
                    import shlex

                    result["agent"] = b.ssh(
                        shlex.quote(remote_python) + " -c " + shlex.quote(script), timeout=180
                    )
                remote_python = config["remote_root"] + "/pipeline/.venv/bin/python"
                import shlex

                result["worker_selftest"] = json.loads(
                    b.ssh(
                        shlex.quote(remote_python)
                        + " -m rcauto.cli --state "
                        + shlex.quote(config["remote_root"] + "/pipeline-checks/selftest")
                        + " selftest",
                        timeout=180,
                    )
                )
                if args.sim:
                    from .validation import validate_native_baseline

                    result["simulation"] = validate_native_baseline(b, config)
            finally:
                result["compute"] = b.stop()["status"]
        write_json(store.root / "doctor.json", result)
        return result
    if cmd == "selftest":
        if args.remote:
            import copy

            config = copy.deepcopy(config)
            config["runtime"]["python"] = config["remote_root"] + "/pipeline/.venv/bin/python"
            config["repair_attempts"] = 0
            r = store.create(
                "REMOTE CONTROL-FLOW FIXTURE ONLY", instance=config["instance"], mode="selftest"
            )
            Coordinator(store, config, ROOT).produce(r["id"])
            return store.get(r["id"])
        from .selftest import run

        return run(store)
    if cmd == "receipt":
        run = store.get(args.run)
        if not run.get("pr"):
            raise StateError("no recorded PR")
        from .receipt import poll

        return poll(store, args.run, config.get("receipt_authors", []))
    if cmd == "reconcile":
        if args.dream_actual_usd is not None:
            if args.dream_actual_usd < 0 or not args.billing_reference:
                raise StateError("nonnegative actual billing and provider reference required")
            store.finish_effect(
                args.run,
                "dream",
                {
                    "actual_usd": args.dream_actual_usd,
                    "billing_reference": args.billing_reference,
                    "reconciled_by": "human",
                },
            )
        return {
            "run": store.get(args.run),
            "effects": {k: store.effect(args.run, k) for k in ["dream", "upload", "pr"]},
        }
    raise StateError("unknown command")


def execute_or_detach(args, store, coordinator, rid, from_stage=None):
    if getattr(args, "detach", False):
        log = store.run_dir(rid) / "supervisor.log"
        argv = [
            sys.executable,
            "-m",
            "rcauto.cli",
            "--state",
            str(store.root),
            "--config",
            args.config,
            "_supervise",
            rid,
        ]
        if from_stage:
            argv.extend(["--from-stage", from_stage])
        with log.open("a") as f:
            child = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=f, stderr=f, start_new_session=True
            )
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            subprocess.Popen(
                ["caffeinate", "-i", "-w", str(child.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        store.update(rid, supervisor_pid=child.pid, supervisor_log=str(log))
        return store.get(rid)
    coordinator.produce(rid, from_stage)
    return store.get(rid)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = read(args.config)
        result = dispatch(args, Store(args.state), config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (StateError, OSError, ValueError) as e:
        print(
            json.dumps({"error": str(e), "type": type(e).__name__}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
