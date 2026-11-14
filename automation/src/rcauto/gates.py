"""Deterministic artifact checks; an agent's success statement is not a receipt."""

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from .store import StateError


class GateError(StateError):
    pass


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


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
        if p.name.lower() in {"auth.json", "credentials.json", "credentials"} or p.name.startswith(
            ".env"
        ):
            raise GateError(f"secret-like file cannot be released: {name}")
        if p.stat().st_size > max_file_mb * 1024 * 1024:
            raise GateError(f"large file requires explicit storage decision: {name}")
        if p.suffix in {
            ".py",
            ".md",
            ".json",
            ".jsonl",
            ".log",
            ".txt",
            ".toml",
            ".yaml",
            ".yml",
            ".sh",
        }:
            text = p.read_text(errors="replace")
            if re.search(
                r"(?:sk-[A-Za-z0-9_-]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|BEGIN .*PRIVATE KEY)", text
            ):
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


def evaluation_identity(release):
    """Presentation edits may reuse the original episode; input/physics edits invalidate it."""
    tasks = list((Path(release) / "tasks").iterdir())
    if len(tasks) != 1:
        raise GateError("one task required for evaluation identity")
    root = tasks[0]
    spec = json.loads((root / "task.json").read_text())
    inputs = {k: spec[k] for k in ["instruction", "setup", "scoring", "run_limit"]}
    files = {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in root.rglob("*")
        if p.is_file() and p.name not in {"task.json", "README.md", "LICENSE", "LICENSE.txt"}
    }
    return digest({"inputs": inputs, "files": files})


def validate_evidence(evidence, sources):
    if not evidence:
        raise GateError("missing industrial evidence")

    def normalize(s, dehyphenate=False):
        # PDF line-wrap hyphens are layout, not a changed quotation. Keep real
        # inline hyphens, and accept either reading at a line-end boundary.
        s = re.sub(
            r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)",
            "" if dehyphenate else "-",
            s.replace("\u00ad", ""),
        )
        return " ".join(s.split())

    for e in evidence:
        source = sources.get(e.get("source_id"))
        if not source or not e.get("locator") or len(e.get("quote", "")) < 12:
            raise GateError("incomplete evidence reference")
        if not any(
            normalize(e["quote"]) in normalize(source["text"], dehyphenate)
            for dehyphenate in (False, True)
        ):
            raise GateError(
                "evidence quote is absent from acquired source "
                + str(e["source_id"])
                + ": "
                + repr(e["quote"][:160])
            )


def validate_checks(report):
    checks = report.get("checks", [])
    if not checks:
        raise GateError("missing numerical checks")
    operations = {
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
    }
    for c in checks:
        a, b = c.get("measured"), c.get("threshold")
        if (
            not c.get("name")
            or not c.get("unit")
            or type(a) not in (int, float)
            or type(b) not in (int, float)
            or not math.isfinite(a)
            or not math.isfinite(b)
            or c.get("operator") not in operations
        ):
            raise GateError("invalid numerical measurement")
        if not operations[c["operator"]](a, b):
            raise GateError(f"failed check {c['name']}: {a} {c['operator']} {b}")
    return checks


def require_fields(value, fields):
    for field in fields:
        if value.get(field) in (None, "", [], {}):
            raise GateError(f"missing required field: {field}")


def validate_spec(spec, require_media=False):
    require_fields(
        spec,
        ["title", "category", "instruction", "setup", "scoring", "run_limit", "fidelity", "scores"]
        + (["media"] if require_media else []),
    )
    if spec["category"] != "Impact on AI timelines":
        raise GateError("task outside the agreed category")


def validate_media(path, kind):
    path = Path(path)
    limit = 50 if kind == "video" else 10
    if not path.is_file() or not 0 < path.stat().st_size <= limit * 1024 * 1024:
        raise GateError("missing/oversized media")
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode:
        raise GateError("invalid media container: " + path.name)
    probe = json.loads(p.stdout)
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if len(streams) != 1 or streams[0].get("width", 0) < 160 or streams[0].get("height", 0) < 90:
        raise GateError("missing/invalid image dimensions")
    if kind == "video" and (
        streams[0]["codec_name"] != "h264"
        or path.suffix.lower() != ".mp4"
        or float(probe["format"].get("duration", 0)) <= 0
    ):
        raise GateError("submission video must be nonempty H264 MP4")
    if kind == "image" and streams[0]["codec_name"] not in {"png", "mjpeg"}:
        raise GateError("setup screenshot must be PNG or JPEG")
    return {"path": str(path), "sha256": file_hash(path), "probe": probe}


def validate_assets(root, value):
    if not value.get("assets"):
        raise GateError("asset provenance manifest missing")
    for a in value["assets"]:
        require_fields(
            a,
            [
                "path",
                "sha256",
                "source_url",
                "license",
                "license_reference",
                "units",
                "dimensions_m",
                "geometry_kind",
            ],
        )
        path = inside(root, a["path"])
        if not path.is_file() or file_hash(path) != a["sha256"]:
            raise GateError("asset content hash mismatch")
        if (
            a["units"] != "m"
            or len(a["dimensions_m"]) != 3
            or any(
                type(n) not in (int, float) or not math.isfinite(n) or n <= 0
                for n in a["dimensions_m"]
            )
        ):
            raise GateError("invalid asset dimensions/units")
        if not inside(root, a["license_reference"]).is_file():
            raise GateError("asset license text missing")


def validate_controls(report, required):
    cases = {c["name"]: c for c in report.get("cases", [])}
    if len(cases) != len(report.get("cases", [])) or not required:
        raise GateError("invalid scorer control cases")
    for expected in required:
        score = cases.get(expected["name"], {}).get("score")
        target = expected["expected_score"]
        if (
            type(score) not in (int, float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
            or abs(score - target) > 1e-6
        ):
            raise GateError("scorer counterexample failed: " + expected["name"])
