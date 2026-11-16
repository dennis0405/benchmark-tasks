"""Losslessly compress copied checkpoint evidence; never alter the live workspace."""

import gzip
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from .gates import GateError, file_hash, inside
from .store import write_json

RESTORE_SOURCE = '''"""Run in a downloaded checkpoint to restore exact original evidence paths."""
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
'''


def archive_evidence(root, *, min_bytes=1024 * 1024):
    """Archive large raw text in the public copy before its unchanged size gate."""
    entries = []
    for directory in (root / "measurements", root / "logs"):
        if not directory.is_dir():
            continue
        for source in sorted(directory.rglob("*")):
            if (
                not source.is_file() or source.is_symlink()
                or source.suffix not in {".json", ".jsonl", ".log", ".txt"}
                or source.stat().st_size < min_bytes
            ):
                continue
            packed = source.with_name(source.name + ".gz")
            temporary = packed.with_name(packed.name + ".tmp")
            if packed.exists():
                raise GateError("compressed evidence path already exists")
            if temporary.exists():
                raise GateError("evidence compression temporary path already exists")
            original_hash, original_bytes = file_hash(source), source.stat().st_size
            try:
                with (
                    source.open("rb") as src,
                    temporary.open("xb") as output,
                    gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as zipped,
                ):
                    shutil.copyfileobj(src, zipped)
                # Verify a complete round trip before removing the copied original.
                with gzip.open(temporary, "rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != original_hash:
                        raise GateError("evidence compression verification failed")
                if temporary.stat().st_size >= original_bytes:
                    continue
                os.replace(temporary, packed)
                entries.append({
                    "original": str(source.relative_to(root)),
                    "compressed": str(packed.relative_to(root)),
                    "sha256": original_hash, "bytes": original_bytes,
                    "compressed_sha256": file_hash(packed),
                    "compressed_bytes": packed.stat().st_size,
                })
                source.unlink()
            finally:
                temporary.unlink(missing_ok=True)
    if entries:
        write_json(root / "evidence-archive.json", {
            "codec": "gzip", "files": entries,
            "restore": "python3 restore-evidence.py",
            "scope": "Public checkpoint copy only; live workspace evidence is unchanged.",
        })
        (root / "restore-evidence.py").write_text(RESTORE_SOURCE)
    return entries


def snapshot_native_sources(workspace, source_manifest):
    """Keep exact code/assets before launch, including attempts that later fail."""
    blob_root = workspace / "measurements/native-source-blobs"
    blob_root.mkdir(parents=True, exist_ok=True)
    blobs = {}
    for relative, info in source_manifest["files"].items():
        source = inside(workspace / "task", relative)
        blob = blob_root / info["sha256"]
        if not blob.exists():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=blob_root, delete=False) as output:
                    temporary = Path(output.name)
                    with source.open("rb") as stream:
                        shutil.copyfileobj(stream, output)
                if file_hash(temporary) != info["sha256"]:
                    raise GateError("native source changed before snapshot")
                os.replace(temporary, blob)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if blob.stat().st_size != info["bytes"] or file_hash(blob) != info["sha256"]:
            raise GateError("native source snapshot is corrupt")
        blobs[relative] = str(blob.relative_to(workspace))
    target = workspace / "measurements/native-source-snapshots" / (source_manifest["digest"] + ".json")
    write_json(target, {
        "qualification": False, "scope": "Exact source bytes; not an execution or physics pass",
        "source_manifest": source_manifest, "blobs": blobs,
    })
    return str(target.relative_to(workspace))
