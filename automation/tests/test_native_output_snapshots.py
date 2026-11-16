"""Failed native attempts keep their evidence even when repair reuses output paths."""

import json
import sys

import pytest

from rcauto.engine import Production
from rcauto.evidence_archive import native_output_inventory, snapshot_native_outputs
from rcauto.gates import GateError, file_hash
from rcauto.store import StateError, Store


def diagnostic(tmp_path, *, exit_code=3, report=True):
    store = Store(tmp_path / "state")
    run = store.create("evidence retention", instance="test", mode="selftest")
    production = Production(store, run["id"], None, {"runtime": {"python": sys.executable}})
    script = production.workspace / "task/probe.py"
    script.write_text(
        "from pathlib import Path\nimport json,sys\n"
        "out=Path(__file__).resolve().parent.parent/'measurements'\n"
        "(out/'trace.json').write_text(json.dumps({'actual_sample':sys.argv[1]}))\n"
        + (
            "(out/'result.json').write_text(json.dumps({'checks':[{'name':'observed',"
            "'measured':0,'operator':'==','threshold':1,'unit':'count'}],"
            "'artifacts':['measurements/trace.json']}))\n"
            if report else ""
        )
        + f"print('native diagnostic stderr',file=sys.stderr)\nsys.exit({exit_code})\n"
    )
    plan = {"phase": "preflight", "scripts": [{
        "id": "probe", "script": "probe.py", "args": ["first"],
        "report": "measurements/result.json",
    }]}
    return production, plan


def saved_attempts(production):
    return [json.loads(p.read_text()) for p in sorted(
        (production.workspace / "measurements/native-output-snapshots").glob("*.json")
    )]


def test_nonzero_exit_preserves_exact_report_trace_and_log_across_repair(tmp_path):
    p, plan = diagnostic(tmp_path)
    with pytest.raises(StateError, match="command exited 3"):
        p.execute_scripts(plan)
    first = saved_attempts(p)[0]
    assert first["qualification"] is False
    assert first["execution_error"]
    original = (p.workspace / "measurements/trace.json").read_bytes()
    plan["scripts"][0]["args"] = ["second"]
    with pytest.raises(StateError, match="command exited 3"):
        p.execute_scripts(plan)
    assert len(saved_attempts(p)) == 2
    entry = first["files"]["measurements/trace.json"]
    assert (p.workspace / entry["blob"]).read_bytes() == original
    assert file_hash(p.workspace / entry["blob"]) == entry["sha256"]
    assert b"native diagnostic stderr" in (p.workspace / first["log"]["blob"]).read_bytes()
    assert "measurements/result.json" in first["files"]
    assert len(list((p.workspace / "measurements/native-output-blobs").iterdir())) < 6


def test_successful_process_with_failed_numeric_gate_still_keeps_outputs(tmp_path):
    p, plan = diagnostic(tmp_path, exit_code=0)
    with pytest.raises(GateError, match="failed check observed"):
        p.execute_scripts(plan)
    attempt = saved_attempts(p)[0]
    assert attempt["execution_error"] is None
    assert attempt["qualification"] is False
    assert "measurements/trace.json" in attempt["files"]
    assert not p.artifact("environment").exists()


def test_runtime_failure_before_report_keeps_partial_trace(tmp_path):
    p, plan = diagnostic(tmp_path, report=False)
    with pytest.raises(StateError, match="command exited 3"):
        p.execute_scripts(plan)
    assert "measurements/trace.json" in saved_attempts(p)[0]["files"]


def test_unchanged_unrelated_measurements_are_not_recopied(tmp_path):
    p, plan = diagnostic(tmp_path)
    (p.workspace / "measurements/unrelated.json").write_text('{"prior":true}')
    with pytest.raises(StateError, match="command exited 3"):
        p.execute_scripts(plan)
    assert "measurements/unrelated.json" not in saved_attempts(p)[0]["files"]


def test_report_cannot_archive_other_directories_or_snapshot_itself(tmp_path):
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    (tmp_path / "private.txt").write_text("do not publish this sentinel")
    prior = measurements / "native-output-blobs"
    prior.mkdir()
    (prior / "previous.bin").write_bytes(b"previous retained evidence")
    before = native_output_inventory(tmp_path)
    report = measurements / "report.json"
    report.write_text(json.dumps({"artifacts": [
        "measurements/../private.txt", "../private.txt",
        "measurements/native-output-blobs/previous.bin",
    ]}))
    relative = snapshot_native_outputs(
        tmp_path, before, report, "source-digest", tmp_path / "missing.log", None,
    )
    saved = json.loads((tmp_path / relative).read_text())
    assert set(saved["files"]) == {"measurements/report.json"}
    assert saved["log"] is None


def test_corrupted_retained_output_blocks_silent_reuse(tmp_path):
    p, plan = diagnostic(tmp_path)
    with pytest.raises(StateError, match="command exited 3"):
        p.execute_scripts(plan)
    saved = saved_attempts(p)[0]
    entry = saved["files"]["measurements/result.json"]
    (p.workspace / entry["blob"]).write_text("corrupted")
    with pytest.raises(GateError, match="output snapshot is corrupt"):
        p.execute_scripts(plan)


@pytest.mark.parametrize("contents", ['{"artifacts":null}', '[1]', '{incomplete'])
def test_malformed_report_does_not_prevent_partial_output_retention(tmp_path, contents):
    measurements = tmp_path / "measurements"
    measurements.mkdir()
    report = measurements / "report.json"
    report.write_text(contents)
    partial = measurements / "partial.txt"
    partial.write_text("actual partial runtime output")
    relative = snapshot_native_outputs(
        tmp_path, {}, report, "source-digest", tmp_path / "missing.log", "failed",
    )
    saved = json.loads((tmp_path / relative).read_text())
    assert (tmp_path / saved["files"]["measurements/report.json"]["blob"]).read_text() == contents
    assert "measurements/partial.txt" in saved["files"]
