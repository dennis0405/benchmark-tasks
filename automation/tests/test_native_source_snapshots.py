import json

import pytest

from rcauto.evidence_archive import snapshot_native_sources
from rcauto.gates import GateError, manifest, verify_manifest


def test_exact_native_source_and_assets_survive_later_rewrites(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "probe.py").write_bytes(b"# original native program\n")
    (task / "assets").mkdir()
    (task / "assets/asset.usdc").write_bytes(b"\x00original binary asset\xff")
    original = manifest(task)
    relative = snapshot_native_sources(tmp_path, original)
    receipt = json.loads((tmp_path / relative).read_text())
    (task / "probe.py").write_text("# a later repair")
    (task / "assets/asset.usdc").unlink()
    restored = tmp_path / "restored"
    restored.mkdir()
    for name, blob in receipt["blobs"].items():
        (restored / name).parent.mkdir(parents=True, exist_ok=True)
        (restored / name).write_bytes((tmp_path / blob).read_bytes())
    verify_manifest(restored, original)
    assert receipt["qualification"] is False
    assert receipt["source_manifest"] == original


def test_unchanged_native_files_are_deduplicated(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "probe.py").write_text("# first version")
    (task / "asset.usda").write_text("# same asset")
    first = snapshot_native_sources(tmp_path, manifest(task))
    (task / "probe.py").write_text("# second version")
    second = snapshot_native_sources(tmp_path, manifest(task))
    assert first != second
    assert len(list((tmp_path / "measurements/native-source-blobs").iterdir())) == 3


def test_corrupted_native_source_snapshot_is_not_silently_reused(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "probe.py").write_text("# original native program")
    original = manifest(task)
    relative = snapshot_native_sources(tmp_path, original)
    receipt = json.loads((tmp_path / relative).read_text())
    (tmp_path / receipt["blobs"]["probe.py"]).write_text("corrupted")
    with pytest.raises(GateError, match="snapshot"):
        snapshot_native_sources(tmp_path, original)
