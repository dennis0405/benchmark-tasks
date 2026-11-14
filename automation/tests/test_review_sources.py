import json

import pytest

from rcauto.engine import Production
from rcauto.gates import GateError, file_hash
from rcauto.store import Store, write_json


def fixture(tmp_path):
    store = Store(tmp_path / "state")
    run = store.create("pinned source review", instance="test", mode="selftest")
    p = Production(store, run["id"], None, {})
    source = tmp_path / "runtime/physics_manager.py"
    source.parent.mkdir()
    source.write_text("# pinned public runtime source\n")
    write_json(
        p.workspace / "input/runtime-api-preflight.json",
        {"source_path": str(source), "sha256": file_hash(source), "isaaclab_commit": "pinned"},
    )
    return p, source


def test_review_mirrors_only_hash_verified_pinned_source_and_preserves_verdict(tmp_path):
    p, source = fixture(tmp_path)
    p.ask = lambda *args, **kwargs: {
        "verdict": "pass",
        "findings": ["Source API exists; native behavior remains unvalidated."],
        "evidence_paths": [str(source)],
    }
    result = p.review("contract-critic", "Check the prospective API", specification_only=True)
    assert result["verdict"] == "pass"
    assert result["findings"] == ["Source API exists; native behavior remains unvalidated."]
    copied = p.workspace / result["evidence_paths"][0]
    assert copied.is_relative_to(p.workspace / "input/pinned-runtime-sources")
    assert copied.read_bytes() == source.read_bytes()
    assert json.loads(p.artifact("contract-critic").read_text())["evidence_paths"] == result["evidence_paths"]


def test_pinned_source_hash_mismatch_is_not_an_evidence_exemption(tmp_path):
    p, source = fixture(tmp_path)
    source.write_text("# source changed after preflight\n")
    with pytest.raises(GateError, match="evidence"):
        p.evidence_file(str(source))
    assert not (p.workspace / "input/pinned-runtime-sources").exists()


def test_identical_unindexed_external_file_still_cannot_be_cited(tmp_path):
    p, source = fixture(tmp_path)
    other = tmp_path / "private.py"
    other.write_bytes(source.read_bytes())
    with pytest.raises(GateError, match="evidence"):
        p.evidence_file(str(other))


def test_verified_snapshot_remains_usable_without_original_runtime_mount(tmp_path):
    p, source = fixture(tmp_path)
    snapshot = p.evidence_file(str(source))
    source.unlink()
    assert p.evidence_file(str(source)) == snapshot


def test_source_snapshot_directory_cannot_escape_through_symlink(tmp_path):
    p, source = fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (p.workspace / "input/pinned-runtime-sources").symlink_to(outside, target_is_directory=True)
    with pytest.raises(GateError):
        p.evidence_file(str(source))
    assert list(outside.iterdir()) == []
