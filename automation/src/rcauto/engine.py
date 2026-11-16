"""Task production gates. Runtime execution is separate from human approval/publication."""

import json
import math
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import httpx
from jsonschema import ValidationError, validate
from pypdf.errors import PdfReadError

from . import contracts as C
from .agents import clean_environment, command
from .benchmarks import validate_benchmarks
from .catalog import get_source
from .gates import (
    PRODUCTION_CATEGORIES,
    GateError,
    digest,
    discard_generated_bytecode,
    file_hash,
    inside,
    manifest,
    require_fields,
    validate_assets,
    validate_checks,
    validate_controls,
    validate_evidence,
    validate_media,
    validate_spec,
    verify_manifest,
)
from .store import StateError, now, write_json


def read(path):
    return json.loads(Path(path).read_text())


def numerical_report(report, requirements):
    measured = {c["name"]: c for c in report.get("checks", [])}
    if len(measured) != len(report.get("checks", [])):
        raise GateError("duplicate measurement name")
    if not requirements:
        raise GateError("no frozen numerical requirements")
    for expected in requirements:
        actual = measured.get(expected["name"])
        if not actual or any(actual.get(k) != v for k, v in expected.items()):
            raise GateError("measurement contract changed/missing: " + expected["name"])
    validate_checks(report)


def validate_contract(contract):
    try:
        validate(contract, C.CONTRACT)
    except ValidationError as e:
        raise GateError("invalid task contract: " + e.message) from e
    checks = contract["physics_checks"]
    names = [c["name"].strip().casefold() for c in checks]
    if len(names) != len(set(names)):
        raise GateError("duplicate physics check names in contract")
    if len(checks) < 4 or any(not n for n in names):
        raise GateError("insufficient or unnamed physics checks")
    if any(not math.isfinite(c["threshold"]) or not c["unit"].strip() for c in checks):
        raise GateError("invalid physics threshold or unit")
    controls = contract["required_controls"]
    names = [c["name"].strip().casefold() for c in controls]
    if not names or any(not n for n in names) or len(names) != len(set(names)):
        raise GateError("missing or duplicate contract control names")
    if any(
        not math.isfinite(c["expected_score"]) or not 0 <= c["expected_score"] <= 1
        for c in controls
    ):
        raise GateError("invalid expected control score")
    timing = contract["timing"]
    if any(not math.isfinite(v) or v <= 0 for v in timing.values()):
        raise GateError("invalid contract timebase")
    if timing["control_hz"] != 20:
        raise GateError("contract control rate differs from trusted 20 Hz evaluator")
    ratio = timing["physics_hz"] / timing["control_hz"]
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        raise GateError("physics rate must be an integral multiple of the control rate")
    if timing["refinement_physics_hz"] != 2 * timing["physics_hz"]:
        raise GateError("contract refinement must halve the physics timestep")
    if contract["run_steps"] <= 0 or not math.isclose(
        timing["episode_seconds"], contract["run_steps"] / timing["control_hz"], rel_tol=1e-9
    ):
        raise GateError("contract episode duration must count control steps, not physics steps")


class CoverageError(StateError):
    """Reviewer transport/coverage failure, never evidence a candidate is a duplicate."""


def novelty_batches(records, max_count, max_bytes):
    """Schedule complete records by serialized size; never truncate a definition."""
    batch, size = [], 0
    for record in records:
        record_size = len(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        if batch and (len(batch) >= max_count or size + record_size > max_bytes):
            yield batch
            batch, size = [], 0
        batch.append(record)
        size += record_size
    if batch:
        yield batch


def compare_catalog(agent, workspace, candidate, catalog, batch_size=100, batch_bytes=400_000):
    if not catalog.get("complete") or not catalog.get("records"):
        raise GateError("catalog incomplete/empty; novelty cannot pass")
    if batch_size <= 0 or batch_bytes <= 0:
        raise ValueError("Novelty batch limits must be positive")
    notes_path = Path(workspace) / "input/novelty-review-notes.json"
    review_notes = read(notes_path) if notes_path.exists() else {}
    if not isinstance(review_notes, dict):
        raise GateError("novelty review notes must be indexed by exact record ID")
    comparisons = []
    terms = set(re.findall(r"[a-z][a-z0-9]+", json.dumps(candidate).lower())) - {
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "for",
        "into",
        "both",
        "its",
        "task",
        "score",
        "success",
        "condition",
        "objective",
        "physical",
        "skill",
    }

    def priority(record):
        title = set(re.findall(r"[a-z][a-z0-9]+", record.get("title", "").lower()))
        text = set(re.findall(r"[a-z][a-z0-9]+", json.dumps(record).lower()))
        return (record["id"] in review_notes, 3 * len(terms & title) + len(terms & text))

    # Retrieval changes order only. Every required current claim must be compared.
    records = sorted(catalog["records"], key=priority, reverse=True)
    by_id = {record["id"]: record for record in records}

    def archival(record):
        # All fields must explicitly establish that there is no current claim.
        return (
            record.get("state") == "closed"
            and record.get("status") == "closed"
            and record.get("own_team") is False
            and record.get("merged_at", "unknown") is None
            and record.get("board_required") is False
        )

    def scope_evidence(record):
        return {
            key: record.get(key)
            for key in (
                "state", "status", "merged_at", "own_team", "board_required", "url", "head_sha"
            )
        }

    required_ids = {record["id"] for record in records if not archival(record)}
    archival_records = [
        {
            "id": record["id"],
            "catalog_record_digest": digest(record),
            "scope_evidence": scope_evidence(record),
            "reason": "not a current claim; retained as history, not certified distinct",
        }
        for record in records if archival(record)
    ]

    def checkpoint():
        inactive = [
            {
                "id": row["id"],
                "verdict": row["verdict"],
                "reason": "inactive closed unmerged other-team history; original verdict retained",
                "catalog_record_digest": digest(by_id[row["id"]]),
                "scope_evidence": scope_evidence(by_id[row["id"]]),
            }
            for row in comparisons
            if row["verdict"] in {"uncertain", "overlap"}
            and archival(by_id[row["id"]])
        ]
        exempt = {row["id"] for row in inactive}
        blockers = [
            row for row in comparisons if row["verdict"] != "distinct" and row["id"] not in exempt
        ]
        report = {
            "catalog_digest": catalog["digest"],
            "candidate_digest": digest(candidate),
            "catalog_total": len(records),
            "complete": required_ids <= {row["id"] for row in comparisons},
            "required_total": len(required_ids),
            "required_compared": sum(row["id"] in required_ids for row in comparisons),
            "archival_records": archival_records,
            "compared": len(comparisons),
            "comparisons": comparisons,
            "inactive_history": inactive,
            "inactive_uncertainty": [row for row in inactive if row["verdict"] == "uncertain"],
            "at": now(),
        }
        write_json(Path(workspace) / "artifacts" / "novelty.json", report)
        if blockers:
            raise GateError(
                f"novelty overlap/uncertainty: {len(blockers)} records; inspect novelty.json"
            )
        return report

    policy = (
        C.SCENARIO_POLICY + "\nRead input/novelty-batch.json. "
        "The candidate in that file is authoritative. Compare EVERY requested record on "
        "objective, physical skill and success conditions. Different geometry/tool names "
        "are not sufficient. Shared generic primitives alone do not prove overlap. Distinct "
        "requires a meaningfully different tested ability, not an industrial label. Explicitly "
        "record missing rubric details; missing numeric thresholds for a clearly unrelated "
        "objective are not by themselves a plausible duplicate. Use uncertain when available "
        "evidence cannot establish a meaningful distinction. Return one comparison per exact "
        "requested id. Keep each comparison axis concise."
        "\nThe batch also indexes separate complete JSON files in record_files. Start by "
        "listing IDs/titles and read each record separately, extracting body, ALL original "
        "source_documents for drafts lacking task.json (including task.py/scoring.py), and ALL spec "
        "versions in bounded output chunks. Never dump the entire batch into a truncated "
        "tool output. Output truncation or a record you have not read is a read_failure, "
        "not evidence that its source lacks details and not an uncertain novelty verdict. "
        "Retry bounded reads; if reading still fails return read_failure for that ID. "
        "Reserve uncertain for genuine ambiguity after reading the available evidence."
    )
    request_path = Path(workspace) / "input/request.json"
    request = read(request_path) if request_path.exists() else {}
    context_key = digest(
        {
            "candidate": candidate,
            "policy": policy,
            "request": {k: request.get(k) for k in ("prompt", "revision_request")},
        }
    )
    cache = Path(workspace) / "artifacts/novelty-cache"
    cache.mkdir(parents=True, exist_ok=True)

    def cache_path(record):
        evidence = {"context": context_key, "record": record}
        if record["id"] in review_notes:
            evidence["review_notes"] = review_notes[record["id"]]
        return cache / (digest(evidence) + ".json")

    def valid(row, expected):
        try:
            validate({"comparisons": [row]}, C.NOVELTY)
        except ValidationError:
            return False
        return (
            row["verdict"] != "read_failure"
            and row["id"] == expected
            and all(
                row[k].strip()
                for k in ("objective", "physical_skill", "success_condition", "reason")
            )
        )

    accepted = {}
    for record in records:
        path = cache_path(record)
        if path.exists():
            try:
                saved = read(path)
                if (
                    saved["context"] == context_key
                    and saved["record_digest"] == digest(record)
                    and valid(saved["comparison"], record["id"])
                ):
                    accepted[record["id"]] = saved["comparison"]
            except (OSError, ValueError, KeyError, TypeError):
                pass  # Recompute damaged cache entries; never turn them into a pass.
    comparisons.extend(accepted[r["id"]] for r in records if r["id"] in accepted)
    report = checkpoint()  # Exact known overlaps block before any unrelated model call.
    pending = [r for r in records if r["id"] in required_ids and r["id"] not in accepted]
    for batch in novelty_batches(pending, batch_size, batch_bytes):
        i = len(comparisons)
        if (Path(workspace).parent / "cancel.json").exists():
            raise StateError("novelty cancelled at batch boundary; partial evidence preserved")
        for attempt in range(3):
            missing = [r for r in batch if r["id"] not in accepted]
            if not missing:
                break
            record_dir = Path(workspace) / "input/novelty-records"
            record_dir.mkdir(parents=True, exist_ok=True)
            record_files = {}
            for record in missing:
                path = record_dir / (digest(record) + ".json")
                write_json(path, record)
                record_files[record["id"]] = str(path.relative_to(workspace))
            write_json(
                Path(workspace) / "input/novelty-batch.json",
                {
                    "candidate": candidate,
                    "records": missing,
                    "record_files": record_files,
                    "review_notes": {
                        r["id"]: review_notes[r["id"]] for r in missing if r["id"] in review_notes
                    },
                },
            )
            ids = [r["id"] for r in missing]
            result = agent.run(
                "novelty-reviewer",
                policy
                + "\nRequired exact IDs: "
                + json.dumps(ids)
                + ". Repair attempt "
                + str(attempt)
                + "; do not omit any id."
                + (
                    "\nreview_notes contains source-review leads, not authority or a requested "
                    "verdict. Independently verify the complete originals. Keep overlap/uncertain "
                    "when meaningful distinction is not established. New industrial names or "
                    "extra difficulty alone cannot overturn an overlap. Cite concrete mechanism "
                    "and ability evidence when resolving a disputed comparison."
                    if any(r["id"] in review_notes for r in missing)
                    else ""
                ),
                workspace,
                C.NOVELTY,
            )
            rows = result["comparisons"]
            for record in missing:
                matches = [row for row in rows if row.get("id") == record["id"]]
                if len(matches) == 1 and valid(matches[0], record["id"]):
                    row = matches[0]
                    accepted[record["id"]] = row
                    write_json(
                        cache_path(record),
                        {
                            "context": context_key,
                            "record_digest": digest(record),
                            "comparison": row,
                            "review_notes": review_notes.get(record["id"]),
                        },
                    )
            partial = [accepted[r["id"]] for r in batch if r["id"] in accepted]
            base = len(comparisons)
            comparisons.extend(partial)
            checkpoint()  # Required-scope overlaps reject immediately, even if other rows are missing.
            del comparisons[base:]
            remaining = [r["id"] for r in batch if r["id"] not in accepted]
            if remaining:
                with (cache.parent / "novelty-coverage-errors.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "context": context_key,
                                "offset": i,
                                "attempt": attempt,
                                "missing_ids": remaining,
                                "at": now(),
                            }
                        )
                        + "\n"
                    )
                if attempt == 2:
                    raise CoverageError(
                        "novelty comparison coverage incomplete after bounded repairs: "
                        + ", ".join(remaining)
                    )
        comparisons.extend(accepted[r["id"]] for r in batch)
        report = checkpoint()

    return report


def approve(store, rid, actor, statement):
    run = store.get(rid)
    if run.get("mode") == "selftest" or run["status"] != "awaiting-review":
        raise GateError("only real, awaiting-review results can be approved")
    if not actor.strip() or not statement.strip() or actor in {"sdk-guard", "agent", "model"}:
        raise GateError("explicit human approval required")
    root = store.run_dir(rid)
    release = read(root / "release-manifest.json")
    verify_manifest(root / "release", release)
    approval = {
        "id": uuid.uuid4().hex,
        "purpose": "human-submission-approval",
        "actor": actor,
        "statement": statement,
        "at": now(),
        "release_digest": release["digest"],
        "report_sha256": file_hash(root / "report.json"),
        "run": rid,
        "target": "robocurve/benchmark-tasks:main",
    }
    write_json(root / "approval.json", approval)
    store.update(rid, approval=approval["id"])
    return approval


def approval_valid(store, rid, approval_id):
    run = store.get(rid)
    root = store.run_dir(rid)
    value = read(root / "approval.json")
    if (
        run.get("approval") != approval_id
        or value.get("id") != approval_id
        or value.get("purpose") != "human-submission-approval"
        or value.get("run") != rid
        or run.get("mode") == "selftest"
        or value.get("report_sha256") != file_hash(root / "report.json")
    ):
        raise GateError("approval does not match current reviewed result")
    m = read(root / "release-manifest.json")
    if value["release_digest"] != m["digest"]:
        raise GateError("approval manifest mismatch")
    verify_manifest(root / "release", m)
    return True


class Production:
    STAGES = ("research", "candidates", "novelty", "contract", "environment", "demo", "export")

    def __init__(self, store, rid, agent, config):
        self.store, self.rid, self.agent, self.config = store, rid, agent, config
        self.root = store.run_dir(rid)
        self.workspace = self.root / "workspace"
        for name in ("input", "artifacts", "task", "measurements"):
            (self.workspace / name).mkdir(parents=True, exist_ok=True)

    def artifact(self, name):
        return self.workspace / "artifacts" / (name + ".json")

    def ask(self, role, instruction, schema, writable=False):
        return self.agent.run(
            role,
            C.SCENARIO_POLICY
            + "\nRead input/last-failure.json if present and address that failure.\n"
            + instruction,
            self.workspace,
            schema,
            writable=writable,
        )

    def review(self, name, instruction, *, specification_only=False, implementation_only=False):
        if specification_only and implementation_only:
            raise ValueError("review scopes are mutually exclusive")
        index_path = self.workspace / "input/runtime-api-preflight.json"
        if index_path.is_file():
            index = read(index_path)
            for source in (
                index.get("source_path"),
                index.get("joint_schema_source", {}).get("path"),
            ):
                if source:
                    self.evidence_file(source)
        implementation_scope = (
            "\nThis is GENERATED-SOURCE review BEFORE native execution. Inspect planned scripts "
            "and their imports/geometry, not nonexistent final measurements. Require actual native "
            "data paths, correct observables and complete claimed qualification logic. Discovery "
            "probes may honestly test only their stated API/component questions; they never prove "
            "the whole frozen environment. Missing API knowledge calls for a small real probe, "
            "not invented measurements. Do not require future GPU results at this source gate. "
        )
        scope = (
            "\nThis is a PROSPECTIVE specification review. Existing task/ code and native "
            "diagnostics may be an explicitly unfinished historical scaffold; their lack of "
            "implementation does not refute the proposed contract. Inspect the actual contract, "
            "primary evidence and pinned API/configuration sources. Require explicit computable "
            "observables, units, aggregation/reset rules and internally consistent scores, but "
            "do not demand that future code, native measurements or demos already exist. The "
            "later environment/physics/demo reviewers independently enforce actual implementation "
            "and measured success. Return repair/blocked for unresolved SPECIFICATION issues. "
            if specification_only
            else "\nInspect actual code and evidence files. Read images with image tools and "
            "verify logs; do not infer a pass from another role's assertion. "
        )
        if implementation_only:
            scope = implementation_scope
        result = self.ask(
            name,
            instruction + scope + "Cite actual files relative to the workspace. Permitted "
            "evidence locations are this workspace and this run's ../logs, ../release and "
            "../evaluation directories. Exact hash-verified runtime source snapshots, when "
            "indexed by input/runtime-api-preflight.json, are prepared under "
            "input/pinned-runtime-sources BEFORE this review. Their provenance records preserve "
            "the original source paths. Return pass only with no material unresolved issues.",
            C.REVIEW,
        )
        write_json(self.artifact(name), result)
        if result["verdict"] != "pass":
            raise GateError(name + ": " + "; ".join(result["findings"]))
        if not result["evidence_paths"]:
            raise GateError("review missing evidence paths")
        result["evidence_paths"] = [
            Path(os.path.relpath(self.evidence_file(path), self.workspace)).as_posix()
            for path in result["evidence_paths"]
        ]
        write_json(self.artifact(name), result)
        return result

    def evidence_file(self, path):
        allowed = [self.workspace.resolve()]
        for evidence_folder in ("logs", "release", "evaluation"):
            directory = self.root / evidence_folder
            if not directory.is_symlink() and directory.resolve().is_relative_to(
                self.root.resolve()
            ):
                allowed.append(directory.resolve())
        target = (self.workspace / path).resolve()
        if target.is_file() and any(target.is_relative_to(root) for root in allowed):
            return target
        # Only the controller's pinned source index authorizes an external source.
        # Copy it into the run so later local review needs no runtime mount.
        index_path = self.workspace / "input/runtime-api-preflight.json"
        if index_path.is_file():
            index = read(index_path)
            entries = [(index.get("source_path"), index.get("sha256"))]
            joint = index.get("joint_schema_source", {})
            entries.append((joint.get("path"), joint.get("sha256")))
            for source, sha in entries:
                if (
                    not isinstance(source, str)
                    or not Path(source).is_absolute()
                    or Path(source).resolve() != target
                    or not isinstance(sha, str)
                    or not re.fullmatch(r"[a-f0-9]{64}", sha)
                ):
                    continue
                relative = Path("input/pinned-runtime-sources") / (sha + "-" + target.name)
                snapshot = inside(self.workspace, relative)
                if snapshot.exists():
                    if not snapshot.is_file() or file_hash(snapshot) != sha:
                        raise GateError("pinned review evidence snapshot hash mismatch")
                    return snapshot
                if (
                    not target.is_file()
                    or target.stat().st_size > 25 * 1024 * 1024
                    or file_hash(target) != sha
                ):
                    raise GateError("pinned review evidence source missing or hash mismatch")
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target, snapshot)
                if file_hash(snapshot) != sha:
                    snapshot.unlink()
                    raise GateError("pinned review evidence changed during copy")
                write_json(
                    inside(self.workspace, str(relative) + ".provenance.json"),
                    {
                        "original_path": source,
                        "sha256": sha,
                        "source_index": "input/runtime-api-preflight.json",
                        "isaaclab_commit": index.get("isaaclab_commit"),
                    },
                )
                return snapshot
        raise GateError(f"review evidence missing or outside approved run directories: {path}")

    def run(self, from_stage=None):
        completed = self.store.completed(self.rid)
        if from_stage:
            if from_stage not in self.STAGES:
                raise GateError("invalid resume stage")
            completed = [
                s for s in completed if self.STAGES.index(s) < self.STAGES.index(from_stage)
            ]
            self.store.invalidate(self.rid, self.STAGES[self.STAGES.index(from_stage) :])
        for stage in self.STAGES:
            if self.pause_requested(stage):
                return
            if stage in completed:
                continue
            self.store.update(self.rid, status="working", stage=stage)
            checkpoint = self.root / "worker-status.json"
            write_json(checkpoint, {"status": "working", "stage": stage, "at": now()})
            attempts = self.config.get("repair_attempts", 2) if stage != "novelty" else 0
            for attempt in range(attempts + 1):
                if self.pause_requested(stage):
                    return
                try:
                    receipt = getattr(self, stage)()
                    self.store.stage_done(self.rid, stage, receipt)
                    failure = self.workspace / "input" / "last-failure.json"
                    if failure.exists():
                        failure.unlink()
                    break
                except StateError as e:
                    self.store.update(
                        self.rid, last_repair={"stage": stage, "attempt": attempt, "reason": str(e)}
                    )
                    write_json(
                        self.workspace / "input" / "last-failure.json",
                        {"stage": stage, "attempt": attempt, "reason": str(e)},
                    )
                    if attempt == attempts:
                        raise
            write_json(checkpoint, {"status": "working", "completed": stage, "at": now()})
        self.store.update(self.rid, status="pre-evaluation-ready", stage="export")
        write_json(
            self.root / "production-receipt.json",
            {
                "run": self.rid,
                "revision": self.store.get(self.rid).get("revision", 0),
                "release_digest": read(self.root / "release-manifest.json")["digest"],
                "at": now(),
            },
        )
        write_json(
            self.root / "worker-status.json", {"status": "pre-evaluation-ready", "at": now()}
        )

    def pause_requested(self, stage):
        if not (self.root / "cancel.json").exists():
            return False
        self.store.update(self.rid, status="paused", stage=stage)
        write_json(
            self.root / "worker-status.json",
            {"status": "paused", "stage": stage, "at": now()},
        )
        return True

    def research(self):
        result = self.ask(
            "industrial-researcher",
            "Read input/request.json and search original industrial "
            "manuals/service procedures before proposing a task. Use input/catalog.json early to "
            "identify covered work families and inspect concrete near-neighbor bodies/spec versions; "
            "this retrieval does not replace the later full novelty gate. Follow the current revision "
            "request rather than a rejected historical candidate's assumptions. Find at least 3 usable sources, "
            "including official equipment geometry/dimensions and relevant public asset/licensing "
            "pages. Also find original task-definition sources for each suite listed in "
            "input/runtime.json benchmarks, or these defaults: LIBERO, RoboCasa, SimplerEnv, "
            "BEHAVIOR, DataCenterBench, RoboDojo. Mark their kind exactly benchmark:SUITE_NAME. "
            "Return direct URLs (not search pages). Do not download executables.",
            C.RESEARCH,
        )
        if len(result["sources"]) < 3:
            raise GateError("insufficient industrial sources")
        sources = {}
        failures = []
        for source in result["sources"]:
            try:
                acquired = get_source(source["url"], self.workspace / "input" / "sources")
                acquired.update({k: source[k] for k in ("title", "publisher", "kind")})
                sources[acquired["id"]] = acquired
            except (httpx.HTTPError, StateError, OSError, ValueError, PdfReadError) as e:
                failures.append({"url": source["url"], "reason": str(e)})
        write_json(self.workspace / "input" / "sources.json", sources)
        benchmarks = {
            s["kind"].split(":", 1)[1]: s
            for s in sources.values()
            if s["kind"].startswith("benchmark:")
        }
        existing_index = self.workspace / "input/benchmark-sources.json"
        if existing_index.exists():
            for name, descriptor in read(existing_index).items():
                if descriptor.get("definition_manifest"):
                    benchmarks[name] = descriptor
        write_json(self.workspace / "input" / "benchmark-sources.json", benchmarks)
        write_json(self.artifact("research"), {"sources": list(sources), "failures": failures})
        if len(sources) < 2:
            raise GateError("industrial originals could not be acquired")
        if set(self.config.get("benchmarks", [])) - set(benchmarks):
            raise GateError("required existing benchmark originals missing")
        validate_benchmarks(self.workspace, self.config.get("benchmarks", []))
        return {"sources": list(sources)}

    def candidates(self):
        result = self.ask(
            "task-ideator",
            "Read acquired input/sources.json. Generate at least three distinct "
            "industrially meaningful manipulation objectives and select one. Cite exact verbatim "
            "source snippets with source_id and page/section. Use short exact excerpts from acquired "
            "source text and independently verify diagrams; avoid long quotations with extraction "
            "spacing errors. Do not correct OCR inside a verbatim acquired-text quotation. "
            "Keep extreme difficulty subordinate "
            "to validity and a feasible real robot demo. Match actual equipment geometry. "
            "Mark feasibility_status ready only when source/geometry/mechanism evidence is "
            "sufficient to build and test; list critical gaps in missing_evidence. Missing "
            "manufacturer calibration can be an explicit testable modeling assumption, but "
            "an unknown essential retention/contact mechanism cannot. An independently authored "
            "representative fixture is allowed if its form and operation are grounded and it "
            "does not claim an exact unverified product replica. Select only a ready candidate "
            "with no critical missing_evidence. Unsupported alternatives must not be filler. "
            "Consider available pinned YAM runtime in input/runtime.json.",
            C.CANDIDATES,
        )
        if len(result["candidates"]) < 3:
            raise GateError("must compare several candidate objectives")
        sources = read(self.workspace / "input" / "sources.json")
        for c in result["candidates"]:
            validate_evidence(c["evidence"], sources)
        write_json(self.artifact("candidates"), result)
        selected = [c for c in result["candidates"] if c["id"] == result["selected_id"]]
        if len(selected) != 1:
            raise GateError("candidate selection ambiguous")
        if selected[0].get("feasibility_status") != "ready" or selected[0].get("missing_evidence"):
            raise GateError("selected candidate lacks build-ready industrial/physics evidence")
        write_json(self.artifact("selected"), selected[0])
        self.candidate_preflight(selected[0])
        return {"selected": selected[0]["id"]}

    def candidate_preflight(self, candidate):
        previous = {
            self.artifact(name).name: read(self.artifact(name))
            for name in ("candidate-preflight", "candidate-critic")
            if self.artifact(name).exists()
        }
        if previous:
            archive = (
                self.workspace / "artifacts/history" / ("candidate-review-" + digest(previous))
            )
            archive.mkdir(parents=True, exist_ok=True)
            for filename in previous:
                (self.workspace / "artifacts" / filename).replace(archive / filename)
        sources = read(self.workspace / "input/sources.json")
        review = self.review(
            "candidate-critic",
            "Read artifacts/candidates.json, artifacts/selected.json, input/sources.json and "
            "input/operator-source-notes.json if present. Independently verify the selected "
            "candidate's industrial evidence, actual equipment dimensions, essential mechanism, "
            "units/ratings, source provenance and feasible pinned-YAM build plan. Inspect primary "
            "diagrams/tables with image tools, not just flattened PDF text. Known column-reading "
            "errors must be corrected in the candidate, not deferred as contradictory facts. "
            "A representative prototype and explicitly disclosed testable modeling assumptions "
            "are allowed; do not demand proprietary calibration or already completed physics/demos. "
            "This gate establishes sufficient factual evidence to attempt construction, not measured "
            "physical success. Novelty is a SEPARATE NEXT stage: do not require benchmark/catalog "
            "comparison reports or a novelty pass here, and do not reuse stale novelty-critic "
            "findings as factual blockers. That next stage independently enforces all novelty gates. "
            "The full numerical contract follows. Evidence snippets are exact acquired-text excerpts; "
            "for PDF extraction spacing artifacts use a shorter clean excerpt and verify the numeric "
            "claim against the actual page image rather than falsely calling OCR spacing printed text. "
            "Verify the actual operation sequence and retained parts; where applicable distinguish "
            "captive fastener loosening/tool withdrawal from full fastener removal. "
            "Block missing essential mechanics, "
            "false quotations, unsourced exact-replica claims and unsupported selected readiness.",
        )
        write_json(
            self.artifact("candidate-preflight"),
            {
                "candidate_digest": digest(candidate),
                "source_digest": digest({k: v.get("sha256") for k, v in sources.items()}),
                "review": review,
            },
        )
        return review

    def benchmark_novelty(self, candidate, stage="selection"):
        names = self.config.get("benchmarks", [])
        if not names:
            return {"comparisons": []}
        validate_benchmarks(self.workspace, names)
        index = read(self.workspace / "input/benchmark-sources.json")
        write_json(self.workspace / "input/benchmark-candidate.json", candidate)
        result = self.ask(
            "benchmark-novelty-analyst",
            "Read input/benchmark-candidate.json (authoritative), input/benchmark-sources.json "
            "and every suite's definition_manifest. Produce an EXPLICIT comparison for each exact "
            "required suite: " + json.dumps(names) + ". Verify selector coverage against original "
            "registries and the complete source-file inventories. Inspect original definitions "
            "for the closest potentially similar tasks. Record each suite's coverage method, "
            "objective/physical-ability/success-condition comparison, exact task names, original "
            "source paths (use each manifest entry's exact local_path), verbatim supporting "
            "excerpts, and a reasoned verdict. Inventories are "
            "navigation aids, not task-level evidence. Do not substitute industry names, geometry, "
            "harder parameters or a list of generic primitives for a meaningfully different tested "
            "ability. Derive the closest neighbors from THIS candidate's objective, essential "
            "mechanism and tested ability across each complete registry/inventory; do not copy "
            "neighbors or conclusions from a rejected prior candidate family. Read complete "
            "original files for inspected "
            "neighbors and quote short exact supporting excerpts. When an abstract goal predicate "
            "underspecifies a mechanism, inspect its actual mapping/backend state implementation "
            "in the corpus before declaring the mechanism unavailable. Evaluate ambiguity about "
            "the same central tested ability, rather than a shared ubiquitous primitive alone. "
            "Retain uncertain when the original evidence still cannot resolve a plausible required "
            "overlap or coverage gap. Do not "
            "claim any simulator/demo/model success from this source comparison. This is the "
            + stage
            + " novelty analysis; retain every required source/novelty quality gate.",
            C.BENCHMARK_NOVELTY,
        )
        rows = result["comparisons"]
        if len(rows) != len(names) or {r["suite"] for r in rows} != set(names):
            raise CoverageError("benchmark comparison suite coverage incomplete")
        missing_quotes = []
        for row in rows:
            corpus = read(self.workspace / index[row["suite"]]["definition_manifest"])
            originals = {source["local_path"]: source for source in corpus["files"]}
            if not row["closest_tasks"] or not row["evidence_paths"]:
                raise CoverageError("benchmark comparison lacks original task evidence")
            for path in row["evidence_paths"]:
                self.evidence_file(path)
            for task in row["closest_tasks"]:
                if task["source_path"] not in originals:
                    matches = [
                        source["local_path"]
                        for source in corpus["files"]
                        if source.get("path") == task["source_path"]
                    ]
                    if len(matches) != 1:
                        raise GateError(
                            "benchmark comparison must cite an original suite definition"
                        )
                    task["source_path"] = matches[0]
                text = inside(self.workspace, task["source_path"]).read_text()
                if not task["quote"].strip() or task["quote"] not in text:
                    missing_quotes.append((row["suite"], task))
        if missing_quotes:
            self.repair_benchmark_quotes(missing_quotes, candidate, stage)
        report = {
            **result,
            "candidate_digest": digest(candidate),
            "at": now(),
            "definition_manifests": {name: index[name]["manifest_sha256"] for name in names},
        }
        write_json(self.artifact("benchmark-novelty-" + stage), report)
        if any(
            row["verdict"] != "distinct"
            or any(task["verdict"] != "distinct" for task in row["closest_tasks"])
            for row in rows
        ):
            raise GateError("required benchmark overlap/uncertainty; inspect explicit comparison")
        return report

    def repair_benchmark_quotes(self, missing, candidate, stage):
        targets = {(suite, task["name"], task["source_path"]): task for suite, task in missing}
        if len(targets) != len(missing):
            raise GateError("benchmark quote repair references ambiguous")
        feedback = "Original excerpt is not an exact substring of the verified source."
        for attempt in range(2):
            write_json(
                self.workspace / "input/benchmark-quote-repair.json",
                {
                    "candidate_digest": digest(candidate),
                    "stage": stage,
                    "attempt": attempt,
                    "failure": feedback,
                    "items": [
                        {
                            "suite": suite,
                            "name": name,
                            "source_path": path,
                            "rejected_quote": task["quote"],
                        }
                        for (suite, name, path), task in targets.items()
                    ],
                },
            )
            result = self.ask(
                "benchmark-quote-repairer",
                "Read input/benchmark-quote-repair.json and each complete original source file. "
                "Return exactly one short exact source excerpt for each indexed item, preserving "
                "its suite, name and source_path. Copy actual whitespace/punctuation or choose a "
                "clean shorter excerpt. Do not reevaluate novelty, change neighbors, change source "
                "files or request any verdict: the controller keeps every original judgment and "
                "reason unchanged. An unsupported source cannot be repaired by invention.",
                C.BENCHMARK_QUOTE_REPAIR,
            )
            rows = result["replacements"]
            replacements = {(r["suite"], r["name"], r["source_path"]): r["quote"] for r in rows}
            if len(rows) != len(replacements) or set(replacements) != set(targets):
                feedback = (
                    "Quote repair must preserve exact item/source coverage; no source switch."
                )
                continue
            if any(
                not quote.strip() or quote not in inside(self.workspace, key[2]).read_text()
                for key, quote in replacements.items()
            ):
                feedback = "Quote repair still absent from an original source; copy a shorter exact excerpt."
                continue
            for key, quote in replacements.items():
                targets[key]["quote"] = quote
            return
        raise GateError("benchmark quote repair failed after two bounded attempts: " + feedback)

    def novelty(self):
        validate_benchmarks(self.workspace, self.config.get("benchmarks", []))
        selected = read(self.artifact("selected"))
        if self.config.get("benchmarks"):
            path = self.artifact("candidate-preflight")
            if not path.exists() or read(path).get("candidate_digest") != digest(selected):
                raise GateError("candidate factual preflight missing/stale; resume from candidates")
        pool = read(self.artifact("candidates"))["candidates"]
        pool = [selected] + [c for c in pool if c["id"] != selected["id"]]
        rejected, skipped = [], []
        for candidate in pool:
            if candidate.get("feasibility_status") != "ready" or candidate.get("missing_evidence"):
                skipped.append(
                    {
                        "candidate": candidate["id"],
                        "reason": "not build-ready",
                        "missing_evidence": candidate.get("missing_evidence", []),
                    }
                )
                continue
            write_json(self.artifact("selected"), candidate)
            try:
                if self.config.get("benchmarks") and read(self.artifact("candidate-preflight")).get(
                    "candidate_digest"
                ) != digest(candidate):
                    self.candidate_preflight(candidate)
                self.benchmark_novelty(candidate)
                result = compare_catalog(
                    self.agent,
                    self.workspace,
                    candidate,
                    read(self.workspace / "input" / "catalog.json"),
                )
                self.review(
                    "novelty-critic",
                    "Read artifacts/selected.json, artifacts/novelty.json, "
                    "artifacts/benchmark-novelty-selection.json, "
                    "input/catalog.json and input/benchmark-sources.json. Challenge superficial distinctions "
                    "and compare against LIBERO, RoboCasa, SimplerEnv, BEHAVIOR, DataCenterBench and RoboDojo "
                    "using original suite definitions. Search original task definitions as needed. "
                    "If required coverage or a potentially similar required objective is unknown, block. "
                    "Record source URLs and specific task names in findings even when passing. "
                    "This is the ideation novelty gate: the objective, essential mechanism and tested "
                    "ability must already be clear and sourced. Numeric thresholds and modeling "
                    "assumptions are frozen in the next contract stage; do not demand completed "
                    "environment code, measured physics, demos, scorer or Dream evidence here. "
                    "Their later mandatory gates remain unchanged and no downstream success is claimed."
                    + C.NOVELTY_SCOPE,
                )
                write_json(
                    self.artifact("selection-history"),
                    {"rejected": rejected, "skipped": skipped, "selected": candidate["id"]},
                )
                return {"catalog_digest": result["catalog_digest"], "compared": result["compared"]}
            except GateError as e:
                rejected.append({"candidate": candidate["id"], "reason": str(e)})
        write_json(
            self.artifact("selection-history"),
            {
                "rejected": rejected,
                "skipped": skipped,
                "selected": None,
            },
        )
        raise GateError("no candidate passed the ideation gates; inspect selection-history.json")

    def contract(self):
        result = self.ask(
            "task-specifier",
            "Write frozen TaskContract for artifacts/selected.json. "
            "Numerical physics_checks must include stability, no initial success, joint/contact "
            "mechanics, camera validity and relevant penetration/damage constraints. Units/thresholds "
            "need industrial sources or explicit assumptions. Cover damage shortcuts and measurable "
            "success/partial/failure. Use UNIQUE check and control names; give lower/upper bounds "
            "different names. timing separates control_hz (trusted evaluator:20), physics_hz, "
            "refinement_physics_hz (twice physics_hz) and episode_seconds (run_steps/control_hz). "
            "Read the pinned runtime and actual physics configuration, never infer control time "
            "from physics substeps. Check the untouched initial score and every expected partial "
            "score against the complete scoring formula and hard-failure timing. Bad physical "
            "events must remain possible in simulation and then fail scoring; do not prevent "
            "physical breakage merely because the action is invalid. In the scoring_spec, "
            "damage_rules and approximations, define every measured observable mathematically, "
            "including body/contact sets, force-versus-impulse units, physical step sampling "
            "order, event segmentation, accumulation, reset, signed penetration and deterministic "
            "loss/inaccessibility predicates. Define eligibility and revocation of each bonus. "
            "This is the specification for later construction, not a claim the old task scaffold "
            "already implements it. Slug must be lowercase "
            "kebab-case. Do not develop demo yet.",
            C.CONTRACT,
        )
        if not re.fullmatch(r"[a-z][a-z0-9-]{2,70}", result["slug"]) or result["run_steps"] <= 0:
            raise GateError("invalid task slug or run limit")
        validate_contract(result)
        suffix = "-" + self.rid.rsplit("-", 1)[-1]
        if not result["slug"].endswith(suffix):
            result["slug"] += suffix
        write_json(self.artifact("contract"), result)
        review = self.review(
            "contract-critic",
            "Independently review artifacts/contract.json against artifacts/selected.json, "
            "the actual pinned runtime sources and industrial evidence. Recompute initial, "
            "partial, full and hard-failure scores for EVERY required control from the stated "
            "formula, including when a bonus first becomes available and whether timeout changes "
            "the outcome. Reject contradictions or ambiguous measurement names. Verify the "
            "control/physics timestep distinction, episode duration, refinement rates and "
            "freshness of contact measurements. Ensure bad physical events are simulated and "
            "scored as failures rather than suppressed by a desired-action condition. Respect "
            "physically valid alternatives and preserve the selected essential ability. This "
            "is a specification consistency gate; native tests and demos come later. "
            "Do not lower validity, novelty, damage or numerical criteria to obtain a pass.",
            specification_only=True,
        )
        write_json(
            self.artifact("contract-preflight"),
            {"contract_digest": digest(result), "review": review},
        )
        return {"contract_digest": digest(result)}

    def execute_scripts(self, plan, requirements=None):
        receipts = []
        if len({s["id"] for s in plan["scripts"]}) != len(plan["scripts"]):
            raise GateError("duplicate diagnostic script ids")
        for index, script in enumerate(plan["scripts"]):
            if (self.root / "cancel.json").exists():
                raise StateError("diagnostic cancelled before native launch")
            if not re.fullmatch(r"[a-z][a-z0-9_-]{0,60}", script["id"]):
                raise GateError("invalid diagnostic id")
            relative = Path(script["script"])
            if relative.is_absolute():
                raise GateError("diagnostic script must be task-relative")
            if relative.parts and relative.parts[0] == "task":
                relative = Path(*relative.parts[1:])
            path = inside(self.workspace / "task", relative)
            if path.suffix != ".py" or not path.is_file():
                raise GateError("diagnostic script does not exist under task/: " + str(relative))
            report_path = inside(self.workspace, script["report"])
            if not report_path.is_relative_to((self.workspace / "measurements").resolve()):
                raise GateError("diagnostic reports must stay in measurements/")
            if report_path.exists():
                report_path.unlink()
            # CLI/python path and EULA flags are controlled by runtime config, not by generated argv.
            runtime = self.config["runtime"]
            env = clean_environment(
                {
                    "PATH": str(Path(runtime["python"]).parent) + ":" + clean_environment()["PATH"],
                    "ACCEPT_EULA": "Y",
                    "OMNI_KIT_ACCEPT_EULA": "YES",
                    "PRIVACY_CONSENT": "Y",
                }
            )
            discarded = discard_generated_bytecode(self.workspace / "task")
            source_before = manifest(self.workspace / "task")
            protected = {
                str(p): file_hash(p)
                for base in ["input", "artifacts"]
                for p in (self.workspace / base).rglob("*")
                if p.is_file()
            }
            receipt = command(
                [runtime["python"], str(path), *script["args"]],
                self.workspace / "task",
                self.root / "logs" / (script["id"] + "-" + uuid.uuid4().hex[:10] + ".log"),
                timeout=self.config.get("sim_timeout", 1800),
                env=env,
            )
            discarded += discard_generated_bytecode(self.workspace / "task")
            verify_manifest(self.workspace / "task", source_before)
            if any(not Path(p).is_file() or file_hash(p) != h for p, h in protected.items()):
                raise GateError("diagnostic modified frozen controller inputs/contracts")
            if not report_path.is_file():
                raise GateError("script did not produce a fresh report")
            report = read(report_path)
            validation_scope = "script-requirements"
            expected = requirements or report.get("requirements", [])
            if not requirements and plan.get("phase") == "preflight":
                # These reviewed discovery programs intentionally have no full
                # production contract. Validate every actual probe measurement,
                # but never label the receipt as environment qualification.
                validation_scope = "native-discovery"
                expected = [
                    {k: c.get(k) for k in ("name", "operator", "threshold", "unit")}
                    for c in report.get("checks", [])
                ]
            if requirements:
                if index == len(plan["scripts"]) - 1:
                    validation_scope = "frozen-physics-aggregate"
                else:
                    # A small API/mechanism preflight has its own checks. It
                    # cannot replace the final report's full frozen contract.
                    validation_scope = "auxiliary-preflight"
                    expected = [
                        {k: c.get(k) for k in ("name", "operator", "threshold", "unit")}
                        for c in report.get("checks", [])
                    ]
            numerical_report(report, expected)
            for relative in report.get("artifacts", []):
                artifact = inside(self.workspace, relative)
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise GateError("missing/empty measured artifact")
                if artifact.suffix.lower() == ".mp4":
                    self.sample_video(artifact)
            receipts.append(
                dict(
                    receipt,
                    id=script["id"],
                    report=script["report"],
                    report_sha256=file_hash(report_path),
                    source_digest=source_before["digest"],
                    validation_scope=validation_scope,
                    discarded_bytecode=discarded,
                    measurements=report,
                )
            )
        if not receipts:
            raise GateError("no runtime checks executed")
        return receipts

    def sample_video(self, path):
        """Review roles remain read-only; the controller prepares images they can actually inspect."""
        out = self.workspace / "measurements/samples" / file_hash(path)[:16]
        if (out / "samples.json").exists():
            return read(out / "samples.json")
        out.mkdir(parents=True, exist_ok=True)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode:
            raise GateError("cannot decode actual demo video")
        duration = float(json.loads(probe.stdout)["format"]["duration"])
        samples = []
        for i in range(4):
            at = max(0, duration - 0.05) * i / 3
            target = out / f"frame-{i}.png"
            command(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(at),
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-y",
                    str(target),
                ],
                self.workspace,
                out / f"ffmpeg-{i}.log",
                timeout=60,
            )
            if not target.is_file() or not target.stat().st_size:
                raise GateError("video sample missing")
            samples.append({"at_seconds": at, "path": str(target.relative_to(self.workspace))})
        value = {
            "video": str(Path(path).relative_to(self.workspace)),
            "sha256": file_hash(path),
            "samples": samples,
        }
        write_json(out / "samples.json", value)
        return value

    def environment(self):
        contract = read(self.artifact("contract"))
        validate_contract(contract)
        preflight = self.artifact("contract-preflight")
        if not preflight.exists() or read(preflight).get("contract_digest") != digest(contract):
            raise GateError("missing/stale contract review; resume from contract stage")
        preflight_path = self.artifact("environment-preflights")
        preflights = read(preflight_path) if preflight_path.exists() else []
        max_preflights = self.config.get("environment_preflight_rounds", 3)
        for discovery in range(max_preflights + 1):
            plan = self.ask(
                "environment-builder",
                "Return phase=preflight for small complete native API/component discovery programs "
                "when native behavior is not yet known. Read artifacts/environment-preflights.json "
                "and actual logs before the next iteration. A preflight may cover only its stated "
                "questions and must not claim full frozen physics coverage. Return phase=qualification "
                "ONLY when the complete environment and all frozen measurement implementations exist. "
                "Never fill missing data with constants/proxies: rotation ratio is not torque ratio; "
                "a peak contact is not sustained vented preload; nonzero segmentation is not fixture "
                "segmentation; a constant zero is not a sampled scorer result. "
                "Implement task/scene, actual equipment assets, reset and "
                "component physics checks for artifacts/contract.json. Read pinned runtime source and "
                "examples using paths in input/runtime.json; use public task.py YAM interface. "
                "No complete demo yet. Return Python diagnostic scripts relative to task/ "
                "(e.g. run_component_diagnostics.py) and workspace-relative report paths "
                "(e.g. measurements/physics.json). "
                "This Codex role is sandboxed for source inspection and code generation. Return "
                "COMPLETE native GPU diagnostic scripts for the trusted controller to execute outside "
                "that role sandbox. A CUDA error from a command inside this role is not evidence "
                "that the host lacks a GPU. Do not run GPU rollouts inside the role, substitute static "
                "facts/null values for native measurements, or return only a blocked stub. On host "
                "execution failure read the actual ../logs/ runtime error and repair the script. "
                "Verify launcher arguments against the installed parser, not old examples or mocks. "
                "The controller executes with cwd=workspace/task; plan report paths are workspace-relative, "
                "so scripts may resolve ../measurements from that cwd or their own __file__. "
                "Create task/asset-manifest.json with assets [{path,sha256,source_url,license,license_reference,"
                "units:'m',dimensions_m:[x,y,z],geometry_kind}]. Include license text and actual geometry files, "
                "or procedural geometry source. Asset hashes must match. "
                "Each script must write JSON {checks:[{name,measured,operator,threshold,unit}],artifacts:[...]} "
                "with actual numerical readings. For phase=preflight, return ONLY the small native "
                "probe programs and their own nonempty measured checks; no production manifest "
                "or full aggregate is required and no qualification is claimed. For "
                "phase=qualification, use all frozen contract checks in one aggregate report, "
                "which must be the FINAL script in the plan. Earlier scripts may be auxiliary API "
                "preflights with their own nonempty numerical checks; those never replace any frozen "
                "check in the final aggregate. "
                "not fabricated constants. Reports and render images go under ../measurements/. "
                "On repair read input/last-failure.json; retain criteria and fix the mechanism.",
                C.ENV_BUILD,
                True,
            )
            phase = plan.get("phase")
            if phase not in {"preflight", "qualification"}:
                raise GateError("environment plan must declare preflight or qualification")
            if phase == "preflight" and discovery == max_preflights:
                raise GateError("native discovery did not produce full qualification; inspect preflights")
            if phase == "qualification":
                validate_assets(self.workspace / "task", read(self.workspace / "task/asset-manifest.json"))
            write_json(self.artifact("component-source-plan"), plan)
            source_review = self.review(
                "component-code-reviewer",
                "Read artifacts/component-source-plan.json and its exact planned scripts/imports. "
                "Review phase=" + phase + ". Preflight programs may only measure actual stated "
                "API/component facts; unfinished unexecuted aggregate code is not a preflight pass. "
                "Qualification requires all frozen contract measurements from real native states. "
                "Reject constant score/success readings, rotation ratios substituted for torque, "
                "peak contact substituted for sustained vented N73/cover preload, raster nonzero "
                "counts substituted for fixture segmentation, fake terminal controls, invalid "
                "sampling windows, missing constitutive coupling or altered frozen criteria. "
                "Trace every claimed metric to actual samples and correct bodies/windows/units. "
                "Native results are gathered AFTER this review; do not demand them now. "
                "Executor contract: cwd is workspace/task; plan report paths are workspace-relative. "
                "The executor removes the previous report before launch and binds each successful "
                "receipt to the immutable source digest and actual fresh report hash. These are "
                "controller responsibilities; judge generated code against that real execution context. "
                "Read-only source review, not a physical-success certification.",
                implementation_only=True,
            )
            receipts = self.execute_scripts(
                plan, contract["physics_checks"] if phase == "qualification" else None
            )
            if phase == "qualification":
                break
            preflights.append({
                "qualification": False, "plan": plan, "source_review": source_review,
                "receipts": receipts, "at": now(),
            })
            write_json(preflight_path, preflights)
        write_json(self.artifact("environment-plan"), plan)
        write_json(self.artifact("environment"), receipts)
        self.review(
            "physics-reviewer",
            "Review task code, artifacts/contract.json and executed "
            "artifacts/environment.json. Confirm measurements, contact/collision and mechanism fidelity, "
            "not mock math. Inspect measured setup images against industrial originals. "
            "All contract checks must represent actual simulation state.",
        )
        return {"receipts_digest": digest(receipts)}

    def demo(self):
        plan = self.ask(
            "demo-and-scoring-builder",
            "Environment passed. Implement real YAM action demo, "
            "returning complete GPU runner scripts for controller execution; do not launch GPU "
            "rollouts inside the restricted Codex role. "
            "stateful scoring and raw trace. Run at least two clean-reset full-score episodes, "
            "all contract negative controls and trace replay. Required script ids: demo, controls, "
            "replay. Each writes checks+requirements arrays and actual artifacts as environment scripts. "
            "Demo checks must include full_score_episode_count >= 2 and object_pose_injections == 0. "
            "Controls report includes cases [{name,score}] for EVERY frozen required_control, including "
            "partial credit and damage/forbidden zero scores; wrong_score_count == 0. "
            "Replay includes score_error <= 1e-6. "
            "Provide H264 MP4 video, initial setup screenshot, raw action/state trace and episode logs. "
            "Do not use diagnostic fixtures as robot success. Keep script criteria in contract. "
            "task/task.json must contain official required fields, category matching the actual "
            f"industrial context from {json.dumps(PRODUCTION_CATEGORIES)}, "
            "scores describing actual scripted demo and Dream not run. Add README/license/runtime "
            "pins. Implement a task-local evaluate.py wrapper for final inspect-robots Dream run, "
            "but NEVER execute it. It must honor --output and use the same scorer and run limit.",
            C.BUILD,
            True,
        )
        required = {"demo", "controls", "replay"}
        if not required.issubset({s["id"] for s in plan["scripts"]}):
            raise GateError("demo/controls/replay scripts missing")
        write_json(self.artifact("demo-plan"), plan)
        receipts = self.execute_scripts(plan)
        fixed = {
            "demo": [
                {
                    "name": "full_score_episode_count",
                    "operator": ">=",
                    "threshold": 2,
                    "unit": "episodes",
                },
                {
                    "name": "object_pose_injections",
                    "operator": "==",
                    "threshold": 0,
                    "unit": "writes",
                },
            ],
            "controls": [
                {
                    "name": "wrong_score_count",
                    "operator": "==",
                    "threshold": 0,
                    "unit": "episodes",
                }
            ],
            "replay": [
                {"name": "score_error", "operator": "<=", "threshold": 1e-6, "unit": "score"}
            ],
        }
        for receipt in receipts:
            if receipt["id"] in fixed:
                numerical_report(receipt["measurements"], fixed[receipt["id"]])
            if receipt["id"] == "controls":
                validate_controls(
                    receipt["measurements"], read(self.artifact("contract"))["required_controls"]
                )
        write_json(self.artifact("demo"), receipts)
        self.review(
            "demo-scoring-critic",
            "Read executed demo/controls/replay reports and raw traces. "
            "Check real joint actions, bimanual contact, meaningful scoring, damage cheats, all outcomes, "
            "absence of object teleport/hidden weld/disabled collision. Inspect actual image/video frames. "
            "Confirm reset reproducibility and counterexamples use the same scorer. Block any fake success.",
        )
        return {"receipts_digest": digest(receipts)}

    def export(self):
        validate_spec(read(self.workspace / "task" / "task.json"))
        plan = self.ask(
            "export-builder",
            "Package only portable task code/assets/licenses/README/task.json "
            "under task/; remove generated caches there. Produce actual clean-directory or Docker "
            "reproduction script id portable. It must copy the task to a fresh isolated location, "
            "verify no private path/helper dependence, run demo/scorer/observations and compare score. "
            "Write numerical checks/requirements and artifacts under ../measurements. "
            "Include reproducibility details and native runtime pins, do not call a native rerun a "
            "fresh-container validation. Add evaluation/medias output mapping release-map.json in "
            "workspace root with keys demo_video, screenshot, trace, episode_log (workspace-relative). "
            "Do not put lab/team/user names into task text/media. Do not run Dream.",
            C.BUILD,
            True,
        )
        if "portable" not in {s["id"] for s in plan["scripts"]}:
            raise GateError("portable reproduction missing")
        receipts = self.execute_scripts(plan)
        write_json(self.artifact("portable"), receipts)
        # Demo/export development may have changed scene code. Recheck frozen component gates.
        refreshed = self.execute_scripts(
            read(self.artifact("environment-plan")),
            read(self.artifact("contract"))["physics_checks"],
        )
        write_json(self.artifact("environment-final"), refreshed)
        final_demo = self.execute_scripts(read(self.artifact("demo-plan")))
        previous = {row["id"]: row for row in read(self.artifact("demo"))}
        for receipt in final_demo:
            requirements = [
                {k: c[k] for k in ["name", "operator", "threshold", "unit"]}
                for c in previous[receipt["id"]]["measurements"]["checks"]
            ]
            numerical_report(receipt["measurements"], requirements)
            if receipt["id"] == "controls":
                validate_controls(
                    receipt["measurements"], read(self.artifact("contract"))["required_controls"]
                )
        write_json(self.artifact("demo-final"), final_demo)
        validate_assets(self.workspace / "task", read(self.workspace / "task/asset-manifest.json"))
        mapping = read(self.workspace / "release-map.json")
        require_fields(mapping, ["demo_video", "screenshot", "trace", "episode_log"])
        for p in mapping.values():
            if not inside(self.workspace, p).is_file():
                raise GateError("missing release evidence")
        validate_media(inside(self.workspace, mapping["demo_video"]), "video")
        self.sample_video(inside(self.workspace, mapping["demo_video"]))
        validate_media(inside(self.workspace, mapping["screenshot"]), "image")
        self.review(
            "release-critic",
            "Review all artifacts, code, README, task.json, licenses, "
            "release-map.json, actual initial image and sampled demo video frames. Apply official "
            "input/review-prompt source if available and repository AGENTS.md rules. Check industrial "
            "construct validity, difficult manipulation, scoring completeness, all relevant damage, "
            "blindness and reproducibility. Confirm final objective still matches novelty analysis.",
        )
        release = self.root / "release"
        if release.exists():
            shutil.rmtree(release)
        slug = read(self.artifact("contract"))["slug"]
        shutil.copytree(self.workspace / "task", release / "tasks" / slug)
        media = release / "evidence"
        media.mkdir()
        for key, relative in mapping.items():
            source = inside(self.workspace, relative)
            if source.stat().st_size > 50 * 1024 * 1024:
                raise GateError("large evidence requires storage decision before origin push")
            shutil.copy2(source, media / (key + source.suffix))
        write_json(
            release / "release-map.json",
            {k: "evidence/" + k + Path(v).suffix for k, v in mapping.items()},
        )
        m = manifest(release)
        write_json(self.root / "release-manifest.json", m)
        return {"release_digest": m["digest"], "slug": slug}
