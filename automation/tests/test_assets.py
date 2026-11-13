import pytest

from rcauto.gates import GateError, file_hash, validate_assets, validate_controls


def test_asset_source_license_and_file_hash_are_required(tmp_path):
    p = tmp_path / "parts.py"
    p.write_text("# authored geometry")
    item = {
        "path": "parts.py",
        "sha256": file_hash(p),
        "source_url": "https://example.com/manual",
        "license": "original authored code",
        "license_reference": "LICENSE",
        "units": "m",
        "dimensions_m": [0.1, 0.2, 0.3],
        "geometry_kind": "procedural",
    }
    (tmp_path / "LICENSE").write_text("license")
    validate_assets(tmp_path, {"assets": [item]})
    p.write_text("# geometry changed")
    with pytest.raises(GateError):
        validate_assets(tmp_path, {"assets": [item]})


def test_partial_credit_control_is_distinct_from_damage_zero():
    required = [
        {"name": "partial", "expected_score": 0.5, "reason": "subgoal"},
        {"name": "damage", "expected_score": 0, "reason": "forbidden"},
    ]
    validate_controls(
        {"cases": [{"name": "partial", "score": 0.5}, {"name": "damage", "score": 0}]}, required
    )
    with pytest.raises(GateError):
        validate_controls(
            {"cases": [{"name": "partial", "score": 0.5}, {"name": "damage", "score": 1}]}, required
        )
