"""Operational batch sizing must never skip definitions or weaken novelty gates."""

import json

import pytest

from rcauto.engine import compare_catalog
from rcauto.gates import GateError
from rcauto.store import write_json


class RecordingReviewer:
    def __init__(self, workspace, overlap=None):
        self.workspace = workspace
        self.overlap = overlap
        self.batches = []

    def run(self, *args):
        batch = json.loads((self.workspace / "input/novelty-batch.json").read_text())["records"]
        self.batches.append(batch)
        return {
            "comparisons": [
                {
                    "id": record["id"],
                    "objective": "Complete purpose comparison",
                    "physical_skill": "Complete physical ability comparison",
                    "success_condition": "Complete success criteria comparison",
                    "verdict": "overlap" if record["id"] == self.overlap else "distinct",
                    "reason": "Independently compared original task definition",
                }
                for record in batch
            ]
        }


def setup_catalog(workspace, records):
    write_json(workspace / "input/request.json", {"prompt": "TEST FIXTURE ONLY"})
    (workspace / "artifacts").mkdir()
    return {"complete": True, "digest": "test-only", "records": records}


def test_short_definitions_need_fewer_dispatches_but_every_record_is_compared(tmp_path):
    records = [{"id": f"task-{i}", "body": "Different complete original definition"} for i in range(240)]
    reviewer = RecordingReviewer(tmp_path)
    catalog = setup_catalog(tmp_path, records)
    report = compare_catalog(reviewer, tmp_path, {}, catalog)
    assert report["complete"] and report["compared"] == 240
    assert [len(batch) for batch in reviewer.batches] == [100, 100, 40]
    assert sorted(row["id"] for row in report["comparisons"]) == sorted(r["id"] for r in records)
    # Changing scheduling must not discard exact per-record cache evidence.
    compare_catalog(reviewer, tmp_path, {}, catalog, batch_size=25)
    assert len(reviewer.batches) == 3


def test_large_definition_stays_whole_and_alone_instead_of_being_truncated(tmp_path):
    records = [
        {"id": "short-a", "body": "a" * 20},
        {"id": "large", "body": "한" * 500},
        {"id": "short-b", "body": "b" * 20},
    ]
    reviewer = RecordingReviewer(tmp_path)
    report = compare_catalog(
        reviewer, tmp_path, {}, setup_catalog(tmp_path, records), batch_bytes=300
    )
    assert report["complete"]
    seen = [record for batch in reviewer.batches for record in batch]
    assert sorted(seen, key=lambda r: r["id"]) == sorted(records, key=lambda r: r["id"])
    for batch in reviewer.batches:
        if any(r["id"] == "large" for r in batch):
            assert batch == [records[1]]


def test_a_late_overlap_still_blocks_after_two_large_successful_batches(tmp_path):
    records = [{"id": f"task-{i}"} for i in range(240)]
    reviewer = RecordingReviewer(tmp_path, overlap="task-239")
    with pytest.raises(GateError, match="overlap"):
        compare_catalog(reviewer, tmp_path, {}, setup_catalog(tmp_path, records))
    report = json.loads((tmp_path / "artifacts/novelty.json").read_text())
    assert report["compared"] == 240
    assert any(r["verdict"] == "overlap" for r in report["comparisons"])


def test_cached_large_records_do_not_fragment_small_changed_definitions(tmp_path):
    records = []
    for i in range(20):
        records.extend(
            [
                {"id": f"large-{i}", "body": "complete original " * 1000},
                {"id": f"small-{i}", "body": "original short definition"},
            ]
        )
    catalog = setup_catalog(tmp_path, records)
    reviewer = RecordingReviewer(tmp_path)
    compare_catalog(reviewer, tmp_path, {}, catalog, batch_bytes=10_000)
    reviewer.batches.clear()
    for record in records:
        if record["id"].startswith("small-"):
            record["body"] = "changed complete short definition"
    report = compare_catalog(reviewer, tmp_path, {}, catalog, batch_bytes=10_000)
    assert report["complete"] and report["compared"] == 40
    assert len(reviewer.batches) == 1
    assert len(reviewer.batches[0]) == 20
    assert all(r["body"] == "changed complete short definition" for r in reviewer.batches[0])


def test_an_exact_cached_overlap_blocks_before_unrelated_fresh_reviews(tmp_path):
    duplicate = {"id": "duplicate", "body": "Known overlap"}
    catalog = setup_catalog(tmp_path, [duplicate])
    with pytest.raises(GateError, match="overlap"):
        compare_catalog(RecordingReviewer(tmp_path, overlap="duplicate"), tmp_path, {}, catalog)
    catalog["records"].append({"id": "fresh", "body": "New unrelated definition"})
    write_json(tmp_path / "input/novelty-review-notes.json", {"fresh": {"source": "Review first"}})

    class NoCalls:
        def run(self, *args):
            pytest.fail("A known exact overlap must block before extra model calls")

    with pytest.raises(GateError, match="overlap"):
        compare_catalog(NoCalls(), tmp_path, {}, catalog, batch_size=1)


@pytest.mark.parametrize("limits", [{"batch_size": 0}, {"batch_bytes": 0}])
def test_invalid_operational_limits_fail_closed(tmp_path, limits):
    with pytest.raises(ValueError, match="positive"):
        compare_catalog(
            RecordingReviewer(tmp_path), tmp_path, {}, setup_catalog(tmp_path, [{"id": "a"}]), **limits
        )
