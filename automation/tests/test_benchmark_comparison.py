import pytest

from rcauto.engine import CoverageError, Production
from rcauto.gates import GateError, digest, file_hash
from rcauto.store import Store, write_json


def setup_production(tmp_path, verdict="distinct", quote="(:goal (turned stove))"):
    store = Store(tmp_path / "state")
    run = store.create("industrial task", instance="test")
    p = Production(store, run["id"], None, {"benchmarks": ["LIBERO"]})
    source = p.workspace / "input/task.bddl"
    source.write_text("(:language turn the stove)\n(:goal (turned stove))")
    corpus = p.workspace / "input/manifest.json"
    write_json(
        corpus,
        {
            "complete": True,
            "files": [
                {
                    "local_path": "input/task.bddl",
                    "sha256": file_hash(source),
                    "url": "https://example.com/task.bddl",
                }
            ],
        },
    )
    write_json(
        p.workspace / "input/benchmark-sources.json",
        {
            "LIBERO": {
                "complete": True,
                "definition_manifest": "input/manifest.json",
                "definition_files": 1,
                "manifest_sha256": file_hash(corpus),
            }
        },
    )
    report = {
        "comparisons": [
            {
                "suite": "LIBERO",
                "coverage_method": "Read the complete registered task inventory",
                "objective": "Dynamic commissioning vs stove on",
                "physical_skill": "Tool lock vs direct knob",
                "success_condition": "Two safe stops vs switched on",
                "evidence_paths": ["input/task.bddl"],
                "closest_tasks": [
                    {
                        "name": "turn the stove",
                        "source_path": "input/task.bddl",
                        "quote": quote,
                        "objective": "Different objective",
                        "physical_skill": "Different constraint",
                        "success_condition": "Different terminal state",
                        "verdict": verdict,
                        "reason": "Original goal and mechanism evidence",
                    }
                ],
                "verdict": verdict,
                "reason": "Source-grounded comparison",
            }
        ]
    }

    class Agent:
        def run(self, role, *args, **kwargs):
            assert role == "benchmark-novelty-analyst", "benchmark gate must precede catalog review"
            return report

    p.agent = Agent()
    return p, report


def test_benchmark_analysis_is_bound_to_candidate_and_verified_original_files(tmp_path):
    p, _ = setup_production(tmp_path)
    candidate = {"id": "damper", "objective": "commission"}
    report = p.benchmark_novelty(candidate)
    assert report["candidate_digest"] == digest(candidate)
    assert report["definition_manifests"]["LIBERO"]
    assert p.artifact("benchmark-novelty-selection").exists()


@pytest.mark.parametrize(
    "failure", ["overlap", "missing-suite", "fabricated-quote", "self-citation"]
)
def test_benchmark_gate_rejects_overlap_incomplete_coverage_and_unsupported_evidence(
    tmp_path, failure
):
    p, report = setup_production(
        tmp_path, verdict="overlap" if failure == "overlap" else "distinct"
    )
    if failure == "missing-suite":
        report["comparisons"] = []
    elif failure == "fabricated-quote":
        report["comparisons"][0]["closest_tasks"][0]["quote"] = "invented definition"
    elif failure == "self-citation":
        path = p.workspace / "input/proposal.txt"
        path.write_text("(:goal (turned stove))")
        report["comparisons"][0]["closest_tasks"][0]["source_path"] = "input/proposal.txt"
    with pytest.raises((GateError, CoverageError)):
        p.benchmark_novelty({"id": "damper"})
