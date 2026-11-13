#!/usr/bin/env python3
"""Install the complete pinned Linux x86_64 Codex tool runtime, not credentials."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

VERSION = "0.147.0"
ASSETS = [
    (
        "codex-x86_64-unknown-linux-musl.tar.gz",
        "0246e2e773834e07f0fb5249ed6ebad12e4591e608f8c7bb97dd6a9690544c36",
        "codex",
    ),
    (
        "codex-code-mode-host-x86_64-unknown-linux-musl.tar.gz",
        "0146adfaac8363ec9fcdb5895f7624db5b2e8617a283887938b7fb97a1dd4356",
        "codex-code-mode-host",
    ),
    (
        "bwrap-x86_64-unknown-linux-musl.tar.gz",
        "e73dc46e2ec7176499cb14e26c7b80b9d8e24a39cd51fe8fa0d45ddd8f6fb87c",
        "codex-resources/bwrap",
    ),
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin-dir", type=Path, default=Path.home() / ".local/bin")
    p.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/rcauto/codex-tools")
    args = p.parse_args()
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        p.error("this pinned installer is for Linux x86_64 only")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    installed = []
    for asset, expected, relative in ASSETS:
        cached = args.cache_dir / asset
        url = f"https://github.com/openai/codex/releases/download/rust-v{VERSION}/{asset}"
        if not cached.exists() or sha(cached) != expected:
            with tempfile.NamedTemporaryFile(dir=args.cache_dir, delete=False) as temporary:
                name = Path(temporary.name)
                try:
                    with urllib.request.urlopen(url, timeout=120) as response:
                        shutil.copyfileobj(response, temporary)
                    temporary.flush()
                    if sha(name) != expected:
                        raise RuntimeError("official release checksum mismatch")
                    os.replace(name, cached)
                finally:
                    name.unlink(missing_ok=True)
        target = args.bin_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(cached) as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            if len(members) != 1:
                raise RuntimeError(
                    "expected exactly one executable in the official component archive"
                )
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                name = Path(temporary.name)
                try:
                    with archive.extractfile(members[0]) as source:
                        shutil.copyfileobj(source, temporary)
                    temporary.flush()
                    name.chmod(0o755)
                    os.replace(name, target)
                finally:
                    name.unlink(missing_ok=True)
        installed.append({"path": str(target), "archive_sha256": expected})
    print(json.dumps({"version": VERSION, "installed": installed}, indent=2))


if __name__ == "__main__":
    main()
