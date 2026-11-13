"""Local-only, mode-0600 credentials. Never copied to workers or public checkpoints."""

import json
import os
from pathlib import Path

from .store import StateError

NAMES = {
    "ROBOCURVE_TEAM",
    "ROBOCURVE_TOKEN",
    "DREAM_API_KEY",
    "DREAM_GATEWAY_URL",
    "DREAM_API_URL",
    "DREAM_DATA_URL",
}


def credentials(root):
    p = Path(root) / "private" / "credentials.json"
    value = {}
    if p.exists():
        if p.stat().st_mode & 0o077:
            raise StateError("credential file must have mode 0600")
        value = json.loads(p.read_text())
    value.update({k: v for k, v in os.environ.items() if k in NAMES and v})
    return value


def save(root, values):
    current = credentials(root)
    if set(values) - NAMES:
        raise StateError("unsupported credential")
    current.update(values)
    directory = Path(root) / "private"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "credentials.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(current, f)
        f.flush()
        os.fsync(f.fileno())
