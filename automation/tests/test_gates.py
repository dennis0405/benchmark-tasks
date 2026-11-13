
import pytest

from rcauto.gates import GateError, manifest, validate_checks, validate_evidence, verify_manifest


def test_changed_or_added_release_file_invalidates_approval(tmp_path):
    (tmp_path / "task.py").write_text("original")
    m = manifest(tmp_path)
    assert verify_manifest(tmp_path, m)
    (tmp_path / "new.py").write_text("new")
    with pytest.raises(GateError):
        verify_manifest(tmp_path, m)


def test_symlink_cannot_export_host_files(tmp_path):
    (tmp_path / "outside").symlink_to("/etc/passwd")
    with pytest.raises(GateError, match="symlink"):
        manifest(tmp_path)


def test_fabricated_quote_cannot_support_candidate():
    sources = {"s1": {"text": "Press the release latch before removing the drive."}}
    validate_evidence([{"source_id": "s1", "quote": "Press the release latch", "locator": "procedure"}], sources)
    with pytest.raises(GateError):
        validate_evidence([{"source_id": "s1", "quote": "80 percent automation", "locator": "p2"}], sources)


def test_success_claim_without_measurements_is_not_a_physics_check():
    with pytest.raises(GateError):
        validate_checks({"checks": [{"name": "stable", "passed": True}]})
    validate_checks({"checks": [{"name": "stable", "measured": 0.001,
                                 "operator": "<=", "threshold": 0.002, "unit": "m"}]})


def test_nan_or_boundary_violation_fails():
    for value in [float("nan"), 10]:
        with pytest.raises(GateError):
            validate_checks({"checks": [{"name": "penetration", "measured": value,
                                         "operator": "<=", "threshold": 0.002, "unit": "m"}]})


def test_secrets_cannot_be_released(tmp_path):
    (tmp_path / "auth.json").write_text('{}')
    with pytest.raises(GateError):
        manifest(tmp_path)
