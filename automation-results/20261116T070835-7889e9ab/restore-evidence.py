"""Run in a downloaded checkpoint to restore exact original evidence paths."""
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

base = Path(__file__).resolve().parent

def locate(relative):
    path = (base / relative).resolve()
    if base not in path.parents:
        raise ValueError("Evidence path escapes checkpoint")
    return path

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

index = json.loads((base / "evidence-archive.json").read_text())
for entry in index["files"]:
    packed, target = locate(entry["compressed"]), locate(entry["original"])
    if digest(packed) != entry["compressed_sha256"]:
        raise ValueError("Compressed evidence digest mismatch: " + entry["compressed"])
    if target.exists():
        if digest(target) != entry["sha256"]:
            raise ValueError("Refusing to overwrite changed evidence: " + entry["original"])
        continue
    temporary = target.with_name(target.name + ".restore-tmp")
    if temporary.exists():
        raise ValueError("Restore temporary path already exists")
    try:
        with gzip.open(packed, "rb") as source, temporary.open("xb") as output:
            shutil.copyfileobj(source, output)
        if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
            raise ValueError("Restored evidence digest mismatch: " + entry["original"])
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
print("Restored and verified", len(index["files"]), "evidence files")
