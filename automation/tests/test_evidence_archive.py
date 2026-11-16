import gzip
import json
import subprocess
import sys

import pytest

from rcauto.evidence_archive import archive_evidence


def test_checkpoint_evidence_round_trip_and_repeatable_bytes(tmp_path):
    root = tmp_path / "checkpoint"
    trace = root / "measurements/nested/trace.json"
    trace.parent.mkdir(parents=True)
    original = b'{"sample":[1,2,3]}\n' * 500
    trace.write_bytes(original)
    report = root / "measurements/report.json"
    report.write_text('{"passed":false}')
    video = root / "measurements/demo.mp4"
    video.write_bytes(b"video" * 500)
    archive_evidence(root, min_bytes=100)
    compressed = trace.with_suffix(".json.gz")
    first = compressed.read_bytes()
    assert gzip.decompress(first) == original
    assert not trace.exists()
    assert report.read_text() == '{"passed":false}'
    assert video.read_bytes() == b"video" * 500
    subprocess.run([sys.executable, str(root / "restore-evidence.py")], check=True)
    assert trace.read_bytes() == original
    compressed.unlink()
    archive_evidence(root, min_bytes=100)
    assert compressed.read_bytes() == first


def test_restore_refuses_corrupted_evidence(tmp_path):
    trace = tmp_path / "measurements/trace.json"
    trace.parent.mkdir()
    trace.write_bytes(b"sample\n" * 1000)
    archive_evidence(tmp_path, min_bytes=100)
    trace.with_suffix(".json.gz").write_bytes(gzip.compress(b"different evidence"))
    result = subprocess.run(
        [sys.executable, str(tmp_path / "restore-evidence.py")], capture_output=True, check=False
    )
    assert result.returncode != 0
    assert not trace.exists()


def test_restore_refuses_changed_existing_file(tmp_path):
    trace = tmp_path / "measurements/trace.json"
    trace.parent.mkdir()
    trace.write_bytes(b"sample\n" * 1000)
    archive_evidence(tmp_path, min_bytes=100)
    trace.write_bytes(b"a later experiment")
    result = subprocess.run(
        [sys.executable, str(tmp_path / "restore-evidence.py")], capture_output=True, check=False
    )
    assert result.returncode != 0
    assert trace.read_bytes() == b"a later experiment"


def test_archive_failure_keeps_original(tmp_path, monkeypatch):
    trace = tmp_path / "measurements/trace.json"
    trace.parent.mkdir()
    trace.write_bytes(b"sample\n" * 1000)

    def failed_copy(*args):
        raise OSError("disk full")

    monkeypatch.setattr("rcauto.evidence_archive.shutil.copyfileobj", failed_copy)
    with pytest.raises(OSError, match="disk full"):
        archive_evidence(tmp_path, min_bytes=100)
    assert trace.read_bytes() == b"sample\n" * 1000
    assert not trace.with_suffix(".json.gz").exists()
    assert not (tmp_path / "evidence-archive.json").exists()


def test_archive_index_retains_exact_original_digest(tmp_path):
    import hashlib

    trace = tmp_path / "measurements/trace.json"
    trace.parent.mkdir()
    original = b"sample\n" * 1000
    trace.write_bytes(original)
    archive_evidence(tmp_path, min_bytes=100)
    entry = json.loads((tmp_path / "evidence-archive.json").read_text())["files"][0]
    assert entry["original"] == "measurements/trace.json"
    assert entry["sha256"] == hashlib.sha256(original).hexdigest()
    assert entry["bytes"] == len(original)
