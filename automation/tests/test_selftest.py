import pytest

from rcauto.engine import approve
from rcauto.gates import GateError
from rcauto.selftest import run
from rcauto.store import Store


def test_offline_full_flow_executes_scripts_and_stays_nonpublishable(tmp_path):
    s = Store(tmp_path)
    result = run(s)
    assert result["stages"] == ["research", "candidates", "novelty", "contract", "environment", "demo", "export"]
    assert result["dream_calls"] == result["submissions"] == 0
    assert (s.run_dir(result["run"]) / "release-manifest.json").exists()
    with pytest.raises(GateError):
        approve(s, result["run"], "human", "approve fixture")
