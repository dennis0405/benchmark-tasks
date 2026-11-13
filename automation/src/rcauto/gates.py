"""Deterministic artifact checks; an agent's success statement is not a receipt."""
import hashlib
import json
import math
import re
from pathlib import Path

from .store import StateError


class GateError(StateError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise GateError(f"path escapes workspace: {relative}")
    return p


def manifest(root, max_file_mb=50):
    root = Path(root)
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise GateError(f"symlink cannot be released: {p.name}")
        if not p.is_file():
            continue
        name = p.relative_to(root).as_posix()
        if any(part in {".git", ".codex", ".venv", "__pycache__"} for part in p.parts):
            raise GateError(f"runtime/private file cannot be released: {name}")
        if p.name.lower() in {"auth.json", "credentials.json", "credentials"} or p.name.startswith(".env"):
            raise GateError(f"secret-like file cannot be released: {name}")
        if p.stat().st_size > max_file_mb * 1024 * 1024:
            raise GateError(f"large file requires explicit storage decision: {name}")
        if p.suffix in {".py", ".md", ".json", ".txt", ".toml", ".yaml", ".yml", ".sh"}:
            text = p.read_text(errors="replace")
            if re.search(r"(?:sk-[A-Za-z0-9_-]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|BEGIN .*PRIVATE KEY)", text):
                raise GateError(f"possible secret in release: {name}")
        files[name] = {"sha256": file_hash(p), "bytes": p.stat().st_size}
    if not files:
        raise GateError("empty release")
    return {"files": files, "digest": digest(files)}


def verify_manifest(root, expected):
    actual = manifest(root)
    if actual != expected:
        raise GateError("release changed; previous approval is invalid")
    return True


def validate_evidence(evidence, sources):
    if not evidence:
        raise GateError("missing industrial evidence")
    normalize = lambda s: " ".join(s.split())
    for e in evidence:
        source = sources.get(e.get("source_id"))
        if not source or not e.get("locator") or len(e.get("quote", "")) < 12:
            raise GateError("incomplete evidence reference")
        if normalize(e["quote"]) not in normalize(source["text"]):
            raise GateError("evidence quote is absent from acquired source")


def validate_checks(report):
    checks = report.get("checks", [])
    if not checks:
        raise GateError("missing numerical checks")
    operations = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
                  "==": lambda a, b: a == b, "<": lambda a, b: a < b, ">": lambda a, b: a > b}
    for c in checks:
        a, b = c.get("measured"), c.get("threshold")
        if (not c.get("name") or not c.get("unit") or type(a) not in (int, float)
                or type(b) not in (int, float) or not math.isfinite(a) or not math.isfinite(b)
                or c.get("operator") not in operations):
            raise GateError("invalid numerical measurement")
        if not operations[c["operator"]](a, b):
            raise GateError(f"failed check {c['name']}: {a} {c['operator']} {b}")
    return checks


def require_fields(value, fields):
    for field in fields:
        if value.get(field) in (None, "", [], {}):
            raise GateError(f"missing required field: {field}")


def validate_spec(spec, require_media=False):
    require_fields(spec, ["title", "category", "instruction", "setup", "scoring", "run_limit",
                          "fidelity", "scores"] + (["media"] if require_media else []))
    if spec["category"] != "Impact on AI timelines":
        raise GateError("task outside the agreed category")
