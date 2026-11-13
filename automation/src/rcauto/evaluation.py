"""Trusted single-episode entrypoint; credentials enter through stdin, never argv/files."""

import argparse
import http.client
import json
import logging
import os
import sys
from pathlib import Path

from .store import now, write_json


class EpisodeGuard:
    def __init__(self, path, max_requests):
        self.path, self.max_requests = Path(path), max_requests
        self.episodes, self.requests = 0, 0

    def event(self, kind, **data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(dict(kind=kind, at=now(), **data)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def start(self):
        if self.episodes:
            raise RuntimeError("second Dream episode prohibited")
        self.episodes += 1
        self.event("episode-start", episode=self.episodes)

    def request(self):
        if self.requests >= self.max_requests:
            raise RuntimeError("Dream request guard reached; no automatic retry")
        self.requests += 1
        self.event("request-intent", request=self.requests)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    secret = json.load(sys.stdin)
    os.environ["DREAM_API_KEY"] = secret["DREAM_API_KEY"]
    os.environ["DREAM_CREDENTIALS"] = "/dev/null"
    for name in ["DREAM_GATEWAY_URL", "DREAM_API_URL", "DREAM_DATA_URL"]:
        if secret.get(name):
            os.environ[name] = secret[name]
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    guard = EpisodeGuard(output / "request-journal.jsonl", job["max_requests"])
    from inspect_robots_dream import policy

    old_reset = policy.DreamPolicy.reset

    def reset(self, scene):
        guard.start()
        return old_reset(self, scene)

    policy.DreamPolicy.reset = reset
    old_request = http.client.HTTPConnection.request

    def request(self, method, url, *a, **kw):
        if method == "POST" and url.split("?", 1)[0].endswith("/v1/get_action"):
            guard.request()
        return old_request(self, method, url, *a, **kw)

    http.client.HTTPConnection.request = request
    from inspect_robots.cli import main as inspect_main

    sys.argv = [
        "inspect-robots",
        "run",
        "--embodiment",
        "yam_arena",
        "--policy",
        "dream",
        "-E",
        "task=" + job["task"],
        "-E",
        "object_jitter=0",
        "-E",
        "control_hz=20",
        "-E",
        "cam_width=320",
        "-E",
        "cam_height=176",
        "--instruction",
        job["instruction"],
        "--scorer",
        "task_score",
        "--max-steps",
        str(job["max_steps"]),
        "--epochs",
        "1",
        "--grader",
        "none",
        "--no-prompt",
        "--store-frames",
        "--log-dir",
        str(output),
    ]
    code = 0
    try:
        inspect_main()
    except SystemExit as e:
        code = e.code or 0
    except BaseException as e:
        logging.getLogger(__name__).exception("worker/evaluation terminated")
        code = 1
        guard.event("error", error_type=type(e).__name__)
    finally:
        write_json(
            output / "attempt.json",
            {
                "episodes": guard.episodes,
                "requests": guard.requests,
                "exit_code": code,
                "billing": "unverified-reservation-retained",
            },
        )
        os.environ.pop("DREAM_API_KEY", None)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
