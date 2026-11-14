"""Commit-pinned original task definitions, separate from benchmark overview pages."""

import ast
import fnmatch
import hashlib
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx

from .catalog import GitHub, get_source, public_url
from .gates import GateError, file_hash
from .store import now, write_json

CORPORA = {
    "LIBERO": (
        "Lifelong-Robot-Learning/LIBERO",
        [
            "libero/libero/bddl_files/*.bddl",
            "libero/libero/benchmark/*.py",
            "libero/libero/envs/predicates/*.py",
        ],
    ),
    "RoboCasa": (
        "robocasa/robocasa",
        [
            "robocasa/environments/kitchen/*.py",
            "robocasa/environments/__init__.py",
            "robocasa/utils/dataset_registry.py",
        ],
    ),
    "RoboDojo": (
        "robodojo-benchmark/RoboDojo",
        [
            "task/RoboDojo/tasks/*.py",
            "task/RoboDojo/config/*.yml",
            "task/RoboDojo/task_registry.py",
        ],
    ),
    "BEHAVIOR": (
        "StanfordVL/BEHAVIOR-1K",
        [
            "bddl3/bddl/activity_definitions/*.bddl",
            "OmniGibson/omnigibson/tasks/behavior_task.py",
            "OmniGibson/omnigibson/utils/bddl_utils.py",
        ],
    ),
    "SimplerEnv": (
        "simpler-env/SimplerEnv",
        [
            "simpler_env/__init__.py",
            "simpler_env/utils/env/env_builder.py",
        ],
    ),
}


def select_definitions(tree, patterns):
    if tree.get("truncated") is not False or not patterns:
        raise GateError("benchmark definition tree truncated or selectors absent")
    selected = []
    for item in tree["tree"]:
        if item.get("type") != "blob" or not any(
            fnmatch.fnmatchcase(item["path"], pattern) for pattern in patterns
        ):
            continue
        path = PurePosixPath(item["path"])
        if path.is_absolute() or ".." in path.parts or item.get("mode") not in {"100644", "100755"}:
            raise GateError("unsafe/nonregular benchmark definition path")
        if not 0 <= item.get("size", -1) <= 25 * 1024 * 1024:
            raise GateError("benchmark definition size absent or too large")
        selected.append(item)
    for pattern in patterns:
        if not any(fnmatch.fnmatchcase(item["path"], pattern) for item in selected):
            raise GateError("benchmark definition selector empty: " + pattern)
    return sorted(selected, key=lambda item: item["path"])


def _blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def collect_repository(repo, patterns, cache, api, revision=None, fetch=None):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise GateError("invalid benchmark repository")
    if revision is None:
        revision = api.get("repos/" + repo + "/commits/HEAD")["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise GateError("benchmark commit must be pinned")
    tree = api.get(
        "repos/" + repo + "/git/trees/" + revision + "?recursive=1", cached=True, revision=revision
    )
    entries = select_definitions(tree, patterns)
    cache = Path(cache)
    if fetch is None:
        public_url("https://raw.githubusercontent.com")
    with httpx.Client(timeout=45, follow_redirects=False) as client:

        def download(url, expected_size):
            for attempt in range(3):
                try:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        raw = bytearray()
                        for chunk in response.iter_bytes():
                            raw.extend(chunk)
                            if len(raw) > expected_size:
                                raise GateError("benchmark response exceeds pinned blob size")
                        return bytes(raw)
                except httpx.HTTPError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)

        acquire = fetch or download

        def one(entry):
            relative = PurePosixPath(repo) / revision / entry["path"]
            path = cache / relative
            url = (
                "https://raw.githubusercontent.com/"
                + repo
                + "/"
                + revision
                + "/"
                + quote(entry["path"], safe="/")
            )
            if path.is_symlink() or not path.resolve().is_relative_to(cache.resolve()):
                raise GateError("unsafe benchmark cache path")
            raw = path.read_bytes() if path.exists() else b""
            if not path.exists() or _blob(raw) != entry["sha"]:
                raw = acquire(url, entry["size"])
                if len(raw) != entry["size"] or _blob(raw) != entry["sha"]:
                    raise GateError("benchmark source does not match pinned Git blob")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            return {
                "path": entry["path"],
                "cache_path": relative.as_posix(),
                "url": url,
                "git_blob": entry["sha"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }

        with ThreadPoolExecutor(max_workers=8) as pool:
            files = list(pool.map(one, entries))
    return {
        "repo": repo,
        "commit": revision,
        "tree_sha": tree["sha"],
        "selectors": patterns,
        "complete": True,
        "files": files,
        "gitlinks": {
            item["path"]: item["sha"] for item in tree["tree"] if item.get("type") == "commit"
        },
    }


def _inventory(path):
    """Navigation aid only; reviewers retain access to every complete original file."""
    text = path.read_text()
    if path.suffix == ".bddl":
        return {
            "definition_name": path.parent.name + "/" + path.name,
            "language_lines": [line.strip() for line in text.splitlines() if ":language" in line],
        }
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return {"parse_status": "source retained; Python AST unavailable"}
        return {
            "classes": [
                {"name": n.name, "line": n.lineno, "docstring": ast.get_docstring(n)}
                for n in tree.body
                if isinstance(n, ast.ClassDef)
            ]
        }
    return {"definition_name": path.stem}


def sync_benchmarks(workspace, cache, names, api=None):
    workspace, cache = Path(workspace), Path(cache)
    api = api or GitHub(cache / "github")
    index = {}
    coverage = {}
    for name in names:
        print("benchmark definitions: " + name, file=sys.stderr, flush=True)
        target = workspace / "input/benchmark-corpora" / name
        target.mkdir(parents=True, exist_ok=True)
        if name == "DataCenterBench":
            source = get_source("https://robocurve.org/datacenterbench/", cache / "official-page")
            path = target / "published-task-definitions.txt"
            path.write_text(source["text"])
            files = [
                {
                    "path": "published-task-definitions.txt",
                    "local_path": str(path.relative_to(workspace)),
                    "url": source["url"],
                    "sha256": file_hash(path),
                    "original_response_sha256": source["sha256"],
                }
            ]
            components = []
        else:
            if name not in CORPORA:
                raise GateError("no original-definition selector for benchmark " + name)
            repo, patterns = CORPORA[name]
            primary = collect_repository(repo, patterns, cache / "files", api)
            components = [primary]
            if name == "BEHAVIOR":
                components.append(
                    collect_repository(
                        "StanfordVL/bddl",
                        ["bddl/activity_definitions/*.bddl"],
                        cache / "files",
                        api,
                    )
                )
            if name == "SimplerEnv":
                revision = primary["gitlinks"].get("ManiSkill2_real2sim")
                if not revision:
                    raise GateError("SimplerEnv pinned implementation submodule absent")
                components.append(
                    collect_repository(
                        "simpler-env/ManiSkill2_real2sim",
                        ["mani_skill2_real2sim/envs/*.py"],
                        cache / "files",
                        api,
                        revision=revision,
                    )
                )
            files = []
            for component in components:
                for original in component["files"]:
                    path = target / original["cache_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cache / "files" / original["cache_path"], path)
                    files.append(
                        {
                            **original,
                            "local_path": str(path.relative_to(workspace)),
                            "inventory": _inventory(path),
                        }
                    )
        result = {
            "suite": name,
            "complete": True,
            "acquired_at": now(),
            "components": components,
            "files": files,
            "coverage": "Every original definition file matched by recorded selectors. Reviewer must verify selector/registry coverage; inventory is a navigation aid, not a substitute for originals.",
        }
        path = target / "manifest.json"
        write_json(path, result)
        index[name] = {
            "definition_manifest": str(path.relative_to(workspace)),
            "definition_files": len(files),
            "manifest_sha256": file_hash(path),
            "complete": True,
        }
        coverage[name] = {
            **index[name],
            "components": [{k: c[k] for k in ("repo", "commit", "selectors")} for c in components],
            "scope": result["coverage"],
        }
        print(
            f"benchmark definitions: {name}: {len(files)} original files verified",
            file=sys.stderr,
            flush=True,
        )
    write_json(workspace / "input/benchmark-sources.json", index)
    write_json(workspace / "artifacts/benchmark-definition-coverage.json", coverage)
    validate_benchmarks(workspace, names)
    return index


def validate_benchmarks(workspace, names):
    import json

    workspace = Path(workspace)
    if not names:
        return
    index_path = workspace / "input/benchmark-sources.json"
    if not index_path.exists():
        raise GateError("benchmark original definitions index absent")
    index = json.loads(index_path.read_text())
    for name in names:
        descriptor = index.get(name, {})
        if not descriptor.get("complete") or not descriptor.get("definition_manifest"):
            raise GateError("benchmark original definitions missing: " + name)
        path = workspace / descriptor["definition_manifest"]
        if (
            not path.resolve().is_relative_to(workspace.resolve())
            or file_hash(path) != descriptor["manifest_sha256"]
        ):
            raise GateError("benchmark definition manifest changed: " + name)
        corpus = json.loads(path.read_text())
        if not corpus.get("complete") or not corpus.get("files"):
            raise GateError("benchmark original definition corpus empty: " + name)
        for original in corpus["files"]:
            source = workspace / original["local_path"]
            if (
                not source.resolve().is_relative_to(workspace.resolve())
                or file_hash(source) != original["sha256"]
            ):
                raise GateError("benchmark original missing/changed: " + name)
