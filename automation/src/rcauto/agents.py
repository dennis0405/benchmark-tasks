"""Sequential Codex role invocations and bounded process-group execution."""

import json
import os
import signal
import subprocess
from pathlib import Path

from jsonschema import ValidationError, validate

from .gates import digest
from .store import StateError, now, write_json


def clean_environment(extra=None):
    allowed = {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "DISPLAY",
        "XAUTHORITY",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "ACCEPT_EULA",
        "PRIVACY_CONSENT",
        "OMNI_KIT_ACCEPT_EULA",
    }
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env["PATH"] = str(Path.home() / ".local/bin") + os.pathsep + env.get("PATH", "/usr/bin:/bin")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


def command(argv, cwd, log, timeout=1800, env=None, stdin=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        p = subprocess.Popen(
            [str(a) for a in argv],
            cwd=cwd,
            env=env or clean_environment(),
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            p.communicate(stdin, timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as e:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            raise StateError(f"command timeout/interrupted; inspect {log}") from e
    if p.returncode:
        raise StateError(f"command exited {p.returncode}; inspect {log}")
    return {"argv": [str(a) for a in argv], "exit_code": 0, "log": str(log), "finished_at": now()}


class Agent:
    def __init__(self, executable="codex", model=None, timeout=1800, deny_paths=()):
        self.executable, self.model, self.timeout = executable, model, timeout
        self.deny_paths = deny_paths

    def run(self, role, prompt, workspace, schema, writable=False):
        workspace = Path(workspace)
        directory = workspace.parent / "agent-runs" / (role + "-" + now().replace(":", "-"))
        directory.mkdir(parents=True)
        schema_path, result_path = directory / "schema.json", directory / "result.json"
        write_json(schema_path, schema)
        (directory / "prompt.txt").write_text(prompt)
        denied = [
            str(Path.home() / p)
            for p in [
                ".ssh",
                ".aws",
                ".codex",
                ".dreamlabs",
                ".config/gh",
                ".local/state/rcauto/private",
            ]
        ] + [str(p) for p in self.deny_paths]
        filesystem = {str(workspace / "input"): "read", str(workspace / "artifacts"): "read"}
        filesystem.update({p: "deny" for p in denied})
        table = ", ".join(json.dumps(k) + "=" + json.dumps(v) for k, v in filesystem.items())
        profile = (
            "{extends="
            + json.dumps(":workspace" if writable else ":read-only")
            + ",filesystem={"
            + table
            + "},network={enabled=true}}"
        )
        argv = [
            self.executable,
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--json",
            "--skip-git-repo-check",
            "-c",
            'default_permissions="rcauto"',
            "-c",
            "permissions.rcauto=" + profile,
            "-c",
            'web_search="live"',
            "-c",
            "features.multi_agent=false",
            "-C",
            str(workspace),
            "--output-schema",
            str(schema_path),
            "-o",
            str(result_path),
            "-",
        ]
        if self.model:
            argv[2:2] = ["--model", self.model]
        receipt = command(argv, workspace, directory / "events.jsonl", self.timeout, stdin=prompt)
        try:
            result = json.loads(result_path.read_text())
            validate(result, schema)
        except (OSError, ValueError, ValidationError) as e:
            raise StateError(f"invalid agent JSON/schema: {directory}") from e
        write_json(
            directory / "receipt.json",
            dict(
                receipt,
                role=role,
                model=self.model,
                prompt_hash=digest(prompt),
                output_hash=digest(result),
            ),
        )
        return result
