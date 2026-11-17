"""Score the archived 14-D trajectory in the public task, without a policy call."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from inspect_robots import Scene
from inspect_robots.types import Action

from inspect_adapter import PizzaEmbodiment

TASK_DIR = Path(__file__).resolve().parent
TRACE_SHA256 = "3fcebf1b3aa4a0a0bedb5fb2195089ae8803ef6d80e396c56f6e32ffb88aee84"
SEED = 1415158438


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = gzip.decompress((TASK_DIR / "evidence/dream-actions.jsonl.gz").read_bytes())
    if hashlib.sha256(data).hexdigest() != TRACE_SHA256:
        raise ValueError("Recorded actions do not match the original episode.")
    header, *steps = map(json.loads, data.splitlines())
    if len(steps) != 5000 or header["action_dim"] != 14:
        raise ValueError("Expected the complete 5,000-step bimanual trajectory.")
    runner = PizzaEmbodiment(
        task=str(TASK_DIR / "task.py"),
        control_hz=20,
        cam_height=176,
        cam_width=320,
        object_jitter=0.0,
    )
    try:
        runner.reset(
            Scene(id=header["scene_id"], instruction=runner.task.module.INSTRUCTION),
            seed=SEED,
        )
        peak = 0.0
        for i, row in enumerate(steps):
            if row["t"] != i:
                raise ValueError("Missing or reordered action.")
            result = runner.step(Action(data=np.array(row["action"], dtype=np.float32)))
            peak = max(peak, float(result.info["score"]))
        receipt = {
            "source_run": header["run_id"],
            "action_file_sha256": TRACE_SHA256,
            "steps": len(steps),
            "seed": SEED,
            "max_score": peak,
            "grade": runner._env.grade.result(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
