"""Run a research episode from an immutable, hashed task-tree copy.

This prevents a later lazy import from reading concurrently edited sources.
Experiment evidence and bytecode caches are excluded, not task assets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot",type=Path)
    parser.add_argument("runner",choices=("fold_pouch_robot.py","clip_rigid_probe.py"))
    args,remaining=parser.parse_known_args()
    source=Path(__file__).resolve().parents[1]
    destination=args.snapshot.resolve()
    if destination==source or source in destination.parents:
        parser.error("snapshot must be outside the source task tree")
    shutil.copytree(source,destination,ignore=shutil.ignore_patterns(
        "evidence","__pycache__",".pytest_cache","*.pyc"))
    hashes={str(path.relative_to(destination)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(destination.rglob("*")) if path.is_file()}
    (destination/"snapshot-manifest.json").write_text(json.dumps(dict(
        source=str(source),files=hashes,runner=args.runner,arguments=remaining,
        note="Private execution copy; no running episode imports the mutable source task tree"),indent=2))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1")
    child=subprocess.Popen([sys.executable,str(destination/"experiments"/args.runner),*remaining],env=env)
    def relay(signum,frame):
        del frame
        if child.poll() is None:child.send_signal(signum)
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,relay)
    returncode=child.wait()
    changed=[name for name,digest in hashes.items()
             if not (destination/name).is_file() or
             hashlib.sha256((destination/name).read_bytes()).hexdigest()!=digest]
    (destination/"post-run-source-check.json").write_text(json.dumps(dict(
        unchanged=not changed,changed_files=changed,child_exit_code=returncode),indent=2))
    if changed:raise SystemExit("Execution source snapshot changed during the episode")
    raise SystemExit(returncode)


if __name__=="__main__":main()
