import pytest

from rcauto.engine import CoverageError, approval_valid, approve, compare_catalog, numerical_report
from rcauto.gates import GateError, manifest
from rcauto.store import Store, write_json


def test_script_cannot_replace_contract_threshold_with_easier_value():
    with pytest.raises(GateError):
        numerical_report(
            {
                "checks": [
                    {
                        "name": "penetration",
                        "measured": 0.1,
                        "operator": "<=",
                        "threshold": 10,
                        "unit": "m",
                    }
                ]
            },
            [{"name": "penetration", "operator": "<=", "threshold": 0.002, "unit": "m"}],
        )


def test_catalog_coverage_requires_every_record_and_every_comparison_axis(tmp_path):
    class TruncatedAgent:
        def run(self, *args, **kwargs):
            return {"comparisons": []}

    cat = {"complete": True, "records": [{"id": "a"}], "digest": "d"}
    with pytest.raises(CoverageError, match="coverage"):
        compare_catalog(TruncatedAgent(), tmp_path, {"objective": "x"}, cat)


def test_novelty_repairs_only_missing_rows_and_reuses_exact_evidence_on_resume(tmp_path):
    import json

    class PartialAgent:
        def __init__(self):
            self.sizes = []

        def run(self, *args, **kwargs):
            records = json.loads((tmp_path / "input/novelty-batch.json").read_text())["records"]
            self.sizes.append(len(records))
            if len(self.sizes) == 1:
                records = records[:-1]
            return {
                "comparisons": [
                    {
                        "id": r["id"],
                        "objective": "Different objective",
                        "physical_skill": "Different ability",
                        "success_condition": "Different goal",
                        "verdict": "distinct",
                        "reason": "Distinct tested ability",
                    }
                    for r in records
                ]
            }

    (tmp_path / "input").mkdir()
    (tmp_path / "artifacts").mkdir()
    cat = {"complete": True, "digest": "d", "records": [{"id": str(n)} for n in range(5)]}
    agent = PartialAgent()
    report = compare_catalog(agent, tmp_path, {"objective": "x"}, cat, batch_size=3)
    assert report["complete"] and report["compared"] == 5
    assert agent.sizes == [3, 1, 2]
    report = compare_catalog(agent, tmp_path, {"objective": "x"}, cat, batch_size=3)
    assert report["complete"] and agent.sizes == [3, 1, 2]
    cat["records"][0]["body"] = "changed task specification"
    compare_catalog(agent, tmp_path, {"objective": "x"}, cat, batch_size=3)
    assert agent.sizes[-1] == 1


def test_missing_catalog_cannot_mean_novel(tmp_path):
    with pytest.raises(GateError):
        compare_catalog(None, tmp_path, {}, {"complete": False})


@pytest.mark.parametrize(
    "record,blocks",
    [
        ({"state": "closed", "status": "closed", "own_team": False}, False),
        ({"state": "closed", "status": "closed", "own_team": False, "board_required": True}, True),
        ({"state": "closed", "status": "merged", "own_team": False}, True),
        ({"state": "open", "status": "open-unclassified", "own_team": False}, True),
        ({"state": "closed", "status": "closed", "own_team": True}, True),
        ({}, True),
    ],
)
def test_unavailable_inactive_history_is_not_an_active_novelty_claim(tmp_path, record, blocks):
    class UncertainAgent:
        def run(self, *args, **kwargs):
            return {
                "comparisons": [
                    {
                        "id": "pr-1",
                        "objective": "Unknown",
                        "physical_skill": "Unknown",
                        "success_condition": "Unknown",
                        "verdict": "uncertain",
                        "reason": "Empty test PR has no task specification",
                    }
                ]
            }

    (tmp_path / "input").mkdir()
    (tmp_path / "artifacts").mkdir()
    catalog = {"complete": True, "digest": "catalog", "records": [{"id": "pr-1", **record}]}
    if blocks:
        with pytest.raises(GateError, match="uncertainty"):
            compare_catalog(UncertainAgent(), tmp_path, {}, catalog)
    else:
        report = compare_catalog(UncertainAgent(), tmp_path, {}, catalog)
        assert report["compared"] == 1
        assert report["comparisons"][0]["verdict"] == "uncertain"
        assert report["inactive_uncertainty"][0]["id"] == "pr-1"


def test_known_overlap_stops_before_unnecessary_batches_and_keeps_partial_evidence(tmp_path):
    import json

    class OverlapAgent:
        calls = 0

        def run(self, *args, **kwargs):
            self.calls += 1
            assert self.calls == 1, "must stop after a confirmed overlap"
            batch = json.loads((tmp_path / "input/novelty-batch.json").read_text())
            return {
                "comparisons": [
                    {
                        "id": r["id"],
                        "objective": "Same",
                        "physical_skill": "Same",
                        "success_condition": "Same",
                        "verdict": "overlap",
                        "reason": "Same objective",
                    }
                    for r in batch["records"]
                ]
            }

    (tmp_path / "input").mkdir()
    (tmp_path / "artifacts").mkdir()
    agent = OverlapAgent()
    cat = {"complete": True, "digest": "c", "records": [{"id": "a"}, {"id": "b"}]}
    with pytest.raises(GateError, match="overlap"):
        compare_catalog(agent, tmp_path, {}, cat, batch_size=1)
    report = json.loads((tmp_path / "artifacts/novelty.json").read_text())
    assert not report["complete"] and report["compared"] == 1
    assert agent.calls == 1


def test_approval_is_bound_to_report_and_payload(tmp_path):
    s = Store(tmp_path / "state")
    r = s.create("task", instance="vm")
    root = s.run_dir(r["id"])
    release = root / "release"
    release.mkdir()
    (release / "task.py").write_text("x")
    write_json(root / "release-manifest.json", manifest(release))
    write_json(root / "report.json", {"valid": True})
    s.update(r["id"], status="awaiting-review")
    token = approve(s, r["id"], "human-user", "approved after viewing report")
    assert approval_valid(s, r["id"], token["id"])
    write_json(root / "report.json", {"valid": False})
    with pytest.raises(GateError):
        approval_valid(s, r["id"], token["id"])


def test_fixture_run_and_sdk_guard_event_cannot_be_submission_approval(tmp_path):
    s = Store(tmp_path)
    r = s.create("fixture", instance="fake", mode="selftest")
    s.update(r["id"], status="awaiting-review")
    with pytest.raises(GateError):
        approve(s, r["id"], "sdk-guard", "approval")


def test_presentation_changes_can_reuse_episode_but_physics_changes_cannot(tmp_path):
    from rcauto.gates import evaluation_identity

    task = tmp_path / "tasks/example"
    task.mkdir(parents=True)
    write_json(
        task / "task.json",
        {"instruction": "insert", "setup": "fixture", "scoring": "depth", "run_limit": "10"},
    )
    (task / "task.py").write_text("depth = 0.1")
    (task / "README.md").write_text("old command")
    before = evaluation_identity(tmp_path)
    (task / "README.md").write_text("corrected command")
    assert evaluation_identity(tmp_path) == before
    (task / "task.py").write_text("depth = 0.2")
    assert evaluation_identity(tmp_path) != before
