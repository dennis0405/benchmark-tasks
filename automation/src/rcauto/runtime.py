"""Brev lifecycle restricted to one provisioned instance, plus SSH file transport."""

import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from .store import StateError


class RuntimeErrorState(StateError):
    pass


def capture(argv, timeout=120, stdin=None):
    p = subprocess.run(
        argv, input=stdin, capture_output=True, text=True, timeout=timeout, check=False
    )
    if p.returncode:
        raise RuntimeErrorState(f"command failed: {argv[0]}: {p.stderr[-800:]}")
    return p.stdout


class Brev:
    def __init__(self, config):
        self.config = config
        self.name = config["instance"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}", self.name):
            raise RuntimeErrorState("invalid registered instance name")

    def inventory(self):
        return json.loads(capture(["brev", "ls", "--json"]))["workspaces"]

    def status(self):
        matches = [x for x in self.inventory() if x["name"] == self.name]
        if len(matches) != 1:
            raise RuntimeErrorState("registered instance missing/ambiguous; no automatic creation")
        if matches[0]["id"] != self.config["instance_id"]:
            raise RuntimeErrorState("instance identity mismatch")
        return matches[0]

    def wait(self, expected, timeout=900):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.status()
            if result["status"] == expected:
                return result
            time.sleep(self.config.get("poll_seconds", 10))
        raise RuntimeErrorState(f"instance state unknown: did not reach {expected}")

    def start(self):
        current = self.status()
        if current["status"] == "STOPPED":
            capture(["brev", "start", self.name], timeout=900)
        elif current["status"] not in {"RUNNING", "STARTING"}:
            raise RuntimeErrorState("instance busy/changing state: " + current["status"])
        self.wait("RUNNING")
        capture(["brev", "refresh"], timeout=120)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                self.ssh("true")
                return
            except RuntimeErrorState:
                time.sleep(5)
        raise RuntimeErrorState("instance running but SSH not ready")

    def stop(self):
        current = self.status()
        if current["status"] == "STOPPED":
            return current
        if current["status"] == "RUNNING":
            capture(["brev", "stop", self.name], timeout=180)
        elif current["status"] != "STOPPING":
            raise RuntimeErrorState("refusing stop in unknown state")
        return self.wait("STOPPED")

    def ssh_argv(self, script):
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.name, script]

    def ssh(self, script, timeout=120, stdin=None):
        return capture(self.ssh_argv(script), timeout=timeout, stdin=stdin)

    def rsync(self, source, destination, download=False):
        if "\n" in str(source) + str(destination):
            raise RuntimeErrorState("invalid transfer path")
        remote = str(source if download else destination)
        if not re.fullmatch(r"/[a-zA-Z0-9_./-]+", remote):
            raise RuntimeErrorState("remote transfer path must be an absolute simple path")
        src = self.name + ":" + str(source) if download else str(source)
        dst = str(destination) if download else self.name + ":" + str(destination)
        return capture(
            [
                "rsync",
                "-a",
                "--exclude=__pycache__",
                "--exclude=.venv",
                "--exclude=.git",
                "--exclude=.pytest_cache",
                "-e",
                "ssh -o BatchMode=yes -o ConnectTimeout=20",
                src,
                dst,
            ],
            timeout=600,
        )

    def deploy(self, source):
        base = self.config["remote_root"]
        # A JSON-only model reply does not prove that its file/shell tools are available.
        self.ssh(
            "test -x /home/ubuntu/.local/bin/codex-code-mode-host && "
            "/home/ubuntu/.local/bin/codex sandbox -- /usr/bin/true"
        )
        self.ssh("mkdir -p " + shlex.quote(base + "/pipeline"))
        for relative in ["src", "config", "pyproject.toml", "uv.lock"]:
            self.rsync(Path(source) / relative, base + "/pipeline/")
        return self.ssh(
            "cd "
            + shlex.quote(base + "/pipeline")
            + " && /home/ubuntu/.local/bin/uv sync --frozen --no-dev",
            timeout=300,
        )
