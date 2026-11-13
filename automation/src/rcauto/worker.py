"""Detached GPU worker. No publication or paid-evaluation credentials are accepted here."""
import argparse
import json
import logging
import os
from pathlib import Path

from .agents import Agent
from .engine import Production
from .store import Store, now, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    args = p.parse_args()
    job = json.loads(Path(args.job).read_text())
    config, run = job["config"], job["run"]
    store = Store(config["remote_root"] + "/worker-state")
    root = store.run_dir(run["id"])
    root.mkdir(parents=True, exist_ok=True)
    status = root / "worker-status.json"
    try:
        with store.lock("gpu-worker"):
            store.register(run)
            write_json(status, {"status": "working", "pid": os.getpid(), "at": now()})
            pipeline = Production(store, run["id"], Agent(model=config.get("model"),
                                  timeout=config.get("agent_timeout", 1800)), config)
            pipeline.run(from_stage=job.get("from_stage"))
    except BaseException as e:
        logging.getLogger(__name__).exception("worker/evaluation terminated")
        write_json(status, {"status": "blocked", "error": str(e), "type": type(e).__name__, "at": now()})
        if store.db.execute("SELECT 1 FROM runs WHERE id=?", (run["id"],)).fetchone():
            store.update(run["id"], status="blocked", error=str(e))
    finally:
        store.backup(root / "snapshot")


if __name__ == "__main__":
    main()
