import pytest

from rcauto.engine import approve
from rcauto.gates import GateError
from rcauto.selftest import run
from rcauto.store import Store


def test_offline_full_flow_executes_scripts_and_stays_nonpublishable(tmp_path):
    s = Store(tmp_path)
    result = run(s)
    assert result["stages"] == [
        "research",
        "candidates",
        "novelty",
        "contract",
        "environment",
        "demo",
        "export",
    ]
    assert result["dream_calls"] == result["submissions"] == 0
    assert (s.run_dir(result["run"]) / "release-manifest.json").exists()
    with pytest.raises(GateError):
        approve(s, result["run"], "human", "approve fixture")


def test_resolved_failure_does_not_direct_later_stages_to_repair_old_problem(tmp_path):
    import sys

    from rcauto.selftest import FixtureAgent, FixtureProduction
    from rcauto.store import StateError, write_json

    s = Store(tmp_path)
    r = s.create("fixture", instance="none", mode="selftest")

    class RetryProduction(FixtureProduction):
        calls = 0

        def research(self):
            self.calls += 1
            if self.calls == 1:
                raise StateError("temporary source failure")
            return super().research()

    p = RetryProduction(
        s, r["id"], FixtureAgent(), {"runtime": {"python": sys.executable}, "repair_attempts": 1}
    )
    write_json(p.workspace / "input/request.json", r)
    write_json(
        p.workspace / "input/catalog.json",
        {"complete": True, "digest": "fixture", "records": [{"id": "fixture"}]},
    )
    p.run()
    assert not (p.workspace / "input/last-failure.json").exists()
