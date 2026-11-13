import pytest

from rcauto.engine import approval_valid, approve, compare_catalog, numerical_report
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
    with pytest.raises(GateError, match="coverage"):
        compare_catalog(TruncatedAgent(), tmp_path, {"objective": "x"}, cat)


def test_missing_catalog_cannot_mean_novel(tmp_path):
    with pytest.raises(GateError):
        compare_catalog(None, tmp_path, {}, {"complete": False})


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
