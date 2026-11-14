"""Task production gates. Runtime execution is separate from human approval/publication."""

import json
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
from .catalog import get_source
from .gates import (
    GateError,
    digest,
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


class CoverageError(StateError):
    """Reviewer transport/coverage failure, never evidence a candidate is a duplicate."""


def compare_catalog(agent, workspace, candidate, catalog, batch_size=25):
    if not catalog.get("complete") or not catalog.get("records"):
        raise GateError("catalog incomplete/empty; novelty cannot pass")
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

    # Retrieval changes order only. Passing still requires every catalog record.
    records = sorted(catalog["records"], key=priority, reverse=True)
    by_id = {record["id"]: record for record in records}

    def checkpoint():
        inactive = [
            {"id": row["id"], "reason": "closed unmerged other-team history; uncertainty retained"}
            for row in comparisons
            if row["verdict"] == "uncertain"
            and by_id[row["id"]].get("state") == "closed"
            and by_id[row["id"]].get("status") == "closed"
            and by_id[row["id"]].get("own_team") is False
            and not by_id[row["id"]].get("board_required")
        ]
        exempt = {row["id"] for row in inactive}
        blockers = [
            row for row in comparisons if row["verdict"] != "distinct" and row["id"] not in exempt
        ]
        report = {
            "catalog_digest": catalog["digest"],
            "candidate_digest": digest(candidate),
            "catalog_total": len(records),
            "complete": len(comparisons) == len(records),
            "compared": len(comparisons),
            "comparisons": comparisons,
            "inactive_uncertainty": inactive,
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
        "listing IDs/titles and read each record separately, extracting body and ALL spec "
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

    for i in range(0, len(records), batch_size):
        if (Path(workspace).parent / "cancel.json").exists():
            raise StateError("novelty cancelled at batch boundary; partial evidence preserved")
        batch = records[i : i + batch_size]
        accepted = {}
        for record in batch:
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
                        r["id"]: review_notes[r["id"]]
                        for r in missing if r["id"] in review_notes
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
                    if any(r["id"] in review_notes for r in missing) else ""
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
            checkpoint()  # Known overlaps reject immediately, even if other rows are missing.
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

    def review(self, name, instruction):
        result = self.ask(
            name,
            instruction + "\nInspect actual code and evidence files. "
            "Read images with image tools and verify logs; do not infer a pass from "
            "another role's assertion. Return pass only with no material unresolved issues.",
            C.REVIEW,
        )
        write_json(self.artifact(name), result)
        if result["verdict"] != "pass":
            raise GateError(name + ": " + "; ".join(result["findings"]))
        if not result["evidence_paths"]:
            raise GateError("review missing evidence paths")
        for p in result["evidence_paths"]:
            if not inside(self.workspace, p).exists():
                raise GateError("review references nonexistent evidence")
        return result

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
            if (self.root / "cancel.json").exists():
                self.store.update(self.rid, status="paused", stage=stage)
                write_json(
                    self.root / "worker-status.json",
                    {"status": "paused", "stage": stage, "at": now()},
                )
                return
            if stage in completed:
                continue
            self.store.update(self.rid, status="working", stage=stage)
            checkpoint = self.root / "worker-status.json"
            write_json(checkpoint, {"status": "working", "stage": stage, "at": now()})
            attempts = self.config.get("repair_attempts", 2) if stage != "novelty" else 0
            for attempt in range(attempts + 1):
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

    def research(self):
        result = self.ask(
            "industrial-researcher",
            "Read input/request.json and search original industrial "
            "manuals/service procedures before proposing a task. Find at least 3 usable sources, "
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
        write_json(self.workspace / "input" / "benchmark-sources.json", benchmarks)
        write_json(self.artifact("research"), {"sources": list(sources), "failures": failures})
        if len(sources) < 2:
            raise GateError("industrial originals could not be acquired")
        if set(self.config.get("benchmarks", [])) - set(benchmarks):
            raise GateError("required existing benchmark originals missing")
        return {"sources": list(sources)}

    def candidates(self):
        result = self.ask(
            "task-ideator",
            "Read acquired input/sources.json. Generate at least three distinct "
            "industrially meaningful manipulation objectives and select one. Cite exact verbatim "
            "source snippets with source_id and page/section. Keep extreme difficulty subordinate "
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
        return {"selected": selected[0]["id"]}

    def novelty(self):
        selected = read(self.artifact("selected"))
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
                result = compare_catalog(
                    self.agent,
                    self.workspace,
                    candidate,
                    read(self.workspace / "input" / "catalog.json"),
                )
                self.review(
                    "novelty-critic",
                    "Read artifacts/selected.json, artifacts/novelty.json, "
                    "input/catalog.json and input/benchmark-sources.json. Challenge superficial distinctions "
                    "and compare against LIBERO, RoboCasa, SimplerEnv, BEHAVIOR, DataCenterBench and RoboDojo "
                    "using original suite definitions. Search original task definitions as needed. "
                    "If coverage or a potentially similar objective is unknown, block. Record source URLs "
                    "and specific task names in findings even when passing.",
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
        raise GateError(
            "all proposed candidates duplicate/uncertain; inspect selection-history.json"
        )

    def contract(self):
        result = self.ask(
            "task-specifier",
            "Write frozen TaskContract for artifacts/selected.json. "
            "Numerical physics_checks must include stability, no initial success, joint/contact "
            "mechanics, camera validity and relevant penetration/damage constraints. Units/thresholds "
            "need industrial sources or explicit assumptions. Cover damage shortcuts and measurable "
            "success/partial/failure. Slug must be lowercase kebab-case. Do not develop demo yet.",
            C.CONTRACT,
        )
        if not re.fullmatch(r"[a-z][a-z0-9-]{2,70}", result["slug"]) or result["run_steps"] <= 0:
            raise GateError("invalid task slug or run limit")
        if len(result["physics_checks"]) < 4 or not result["required_controls"]:
            raise GateError("insufficient physics checks/negative controls")
        suffix = "-" + self.rid.rsplit("-", 1)[-1]
        if not result["slug"].endswith(suffix):
            result["slug"] += suffix
        write_json(self.artifact("contract"), result)
        return {"contract_digest": digest(result)}

    def execute_scripts(self, plan, requirements=None):
        receipts = []
        if len({s["id"] for s in plan["scripts"]}) != len(plan["scripts"]):
            raise GateError("duplicate diagnostic script ids")
        for script in plan["scripts"]:
            if not re.fullmatch(r"[a-z][a-z0-9_-]{0,60}", script["id"]):
                raise GateError("invalid diagnostic id")
            path = inside(self.workspace / "task", script["script"])
            if path.suffix != ".py" or not path.is_file():
                raise GateError("runner accepts existing task-local Python scripts only")
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
            verify_manifest(self.workspace / "task", source_before)
            if any(not Path(p).is_file() or file_hash(p) != h for p, h in protected.items()):
                raise GateError("diagnostic modified frozen controller inputs/contracts")
            if not report_path.is_file():
                raise GateError("script did not produce a fresh report")
            report = read(report_path)
            numerical_report(report, requirements or report.get("requirements", []))
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
        plan = self.ask(
            "environment-builder",
            "Implement task/scene, actual equipment assets, reset and "
            "component physics checks for artifacts/contract.json. Read pinned runtime source and "
            "examples using paths in input/runtime.json; use public task.py YAM interface. "
            "No complete demo yet. Return Python diagnostic scripts and relative report paths. "
            "Create task/asset-manifest.json with assets [{path,sha256,source_url,license,license_reference,"
            "units:'m',dimensions_m:[x,y,z],geometry_kind}]. Include license text and actual geometry files, "
            "or procedural geometry source. Asset hashes must match. "
            "Each script must write JSON {checks:[{name,measured,operator,threshold,unit}],artifacts:[...]} "
            "with actual numerical readings. Use all frozen contract checks in one aggregate report, "
            "not fabricated constants. Reports and render images go under ../measurements/. "
            "On repair read input/last-failure.json; retain criteria and fix the mechanism.",
            C.BUILD,
            True,
        )
        validate_assets(self.workspace / "task", read(self.workspace / "task/asset-manifest.json"))
        receipts = self.execute_scripts(plan, contract["physics_checks"])
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
            "stateful scoring and raw trace. Run at least two clean-reset full-score episodes, "
            "all contract negative controls and trace replay. Required script ids: demo, controls, "
            "replay. Each writes checks+requirements arrays and actual artifacts as environment scripts. "
            "Demo checks must include full_score_episode_count >= 2 and object_pose_injections == 0. "
            "Controls report includes cases [{name,score}] for EVERY frozen required_control, including "
            "partial credit and damage/forbidden zero scores; wrong_score_count == 0. "
            "Replay includes score_error <= 1e-6. "
            "Provide H264 MP4 video, initial setup screenshot, raw action/state trace and episode logs. "
            "Do not use diagnostic fixtures as robot success. Keep script criteria in contract. "
            "task/task.json must contain official required fields, category Impact on AI timelines, "
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
