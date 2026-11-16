import hashlib
import json
import shutil

import pytest

from rcauto.gates import GateError
from rcauto.store import write_json


def prepared(tmp_path):
    root = tmp_path / "run"
    workspace = root / "workspace"
    raw = b'{"contacts":"' + b"x" * (1024 * 1024) + b'"}'
    sha = hashlib.sha256(raw).hexdigest()
    blob = f"measurements/native-output-blobs/{sha}.json"
    alias = "measurements/example-trace.json"
    for name in [blob, alias]:
        p = workspace / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    backup = root / "private-experimental-evidence" / (sha + ".json")
    backup.parent.mkdir()
    backup.write_bytes(raw)
    report = {"qualification": False, "overall_pass": False,
              "native_trial_id": "failed-trial", "artifacts": [alias]}
    report_bytes = json.dumps(report).encode()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    (workspace / f"measurements/native-output-blobs/{report_sha}.json").write_bytes(report_bytes)
    summary = "measurements/example-state-trace.json"
    write_json(workspace / summary, {"provenance": {"qualification": False,
               "raw_trace_sha256": sha}, "samples": [{"q": 0.1}]})
    summary_sha = hashlib.sha256((workspace / summary).read_bytes()).hexdigest()
    policy = {"scope": "failed-native-diagnostics", "actor": "fixture-user",
              "statement": "Keep raw data on Mac and Brev; publish derived states.",
              "files": [{"sha256": sha, "bytes": len(raw), "report_sha256": report_sha,
                         "trial": "failed-trial", "aliases": [alias], "summary": summary,
                         "summary_sha256": summary_sha}]}
    write_json(root / "external-evidence-policy.json", policy)
    public = tmp_path / "public"
    shutil.copytree(workspace, public)
    return root, public, policy, backup


def test_externalization_changes_only_public_copy_and_records_recovery(tmp_path):
    from rcauto.external_evidence import externalize_failed_diagnostics

    root, public, policy, backup = prepared(tmp_path)
    item = policy["files"][0]
    before = backup.read_bytes()
    externalize_failed_diagnostics(root, public)
    assert not (public / item["aliases"][0]).exists()
    assert not (public / f"measurements/native-output-blobs/{item['sha256']}.json").exists()
    assert (root / "workspace" / item["aliases"][0]).read_bytes() == before
    assert backup.read_bytes() == before
    assert (public / item["summary"]).is_file()
    index = json.loads((public / "external-evidence.json").read_text())
    assert index["files"][0]["sha256"] == item["sha256"]
    assert index["full_raw_in_checkpoint"] is False
    assert "private-experimental-evidence" not in json.dumps(index)


@pytest.mark.parametrize("change", ["backup", "summary", "source", "report", "path"])
def test_externalization_rejects_unverifiable_or_non_diagnostic_omission(tmp_path, change):
    from rcauto.external_evidence import externalize_failed_diagnostics

    root, public, policy, backup = prepared(tmp_path)
    item = policy["files"][0]
    if change == "backup":
        backup.write_bytes(b"changed")
    elif change == "summary":
        (public / item["summary"]).write_text("{}")
    elif change == "source":
        (root / "workspace" / f"measurements/native-output-blobs/{item['sha256']}.json").write_bytes(b"changed")
    elif change == "report":
        policy["files"][0]["report_sha256"] = item["sha256"]
        write_json(root / "external-evidence-policy.json", policy)
    else:
        policy["files"][0]["aliases"] = ["task/task.py"]
        write_json(root / "external-evidence-policy.json", policy)
    with pytest.raises(GateError):
        externalize_failed_diagnostics(root, public)
    assert (public / "measurements/example-trace.json").exists()


def test_changed_alias_is_not_hidden_by_an_old_storage_decision(tmp_path):
    from rcauto.external_evidence import externalize_failed_diagnostics

    root, public, policy, _ = prepared(tmp_path)
    alias = policy["files"][0]["aliases"][0]
    (root / "workspace" / alias).write_bytes(b"new native trial")
    (public / alias).write_bytes(b"new native trial")
    externalize_failed_diagnostics(root, public)
    assert (public / alias).read_bytes() == b"new native trial"


def test_no_policy_does_not_exclude_anything(tmp_path):
    from rcauto.external_evidence import externalize_failed_diagnostics

    assert externalize_failed_diagnostics(tmp_path, tmp_path) == []


@pytest.mark.parametrize("field", ["overall_pass", "qualification"])
def test_a_success_or_qualified_trial_cannot_use_failed_diagnostic_exclusion(tmp_path, field):
    from rcauto.external_evidence import externalize_failed_diagnostics

    root, public, policy, _ = prepared(tmp_path)
    entry = policy["files"][0]
    old = root / f"workspace/measurements/native-output-blobs/{entry['report_sha256']}.json"
    report = json.loads(old.read_text())
    report[field] = True
    data = json.dumps(report).encode()
    entry["report_sha256"] = hashlib.sha256(data).hexdigest()
    (old.parent / (entry["report_sha256"] + ".json")).write_bytes(data)
    write_json(root / "external-evidence-policy.json", policy)
    with pytest.raises(GateError, match="failed native preflight"):
        externalize_failed_diagnostics(root, public)
