"""Auxiliary probes cannot replace or dilute the frozen physical aggregate."""

import json
import sys

import pytest

from rcauto.engine import Production
from rcauto.gates import GateError
from rcauto.store import Store


def setup_plan(tmp_path, aggregate, *, preflight_value=1):
    store = Store(tmp_path / "state")
    run = store.create("diagnostic gate", instance="test", mode="selftest")
    p = Production(store, run["id"], None, {"runtime": {"python": sys.executable}})
    requirements = [
        {"name": "stable", "operator": "<=", "threshold": 0.1, "unit": "m/s"},
        {"name": "contact", "operator": ">=", "threshold": 1, "unit": "rows"},
    ]
    reports = [
        {"checks": [{"name": "api_probe", "measured": preflight_value,
                     "operator": "==", "threshold": 1, "unit": "count"}]},
        {"checks": aggregate},
    ]
    scripts = []
    for i, report in enumerate(reports):
        script = p.workspace / "task" / f"probe_{i}.py"
        script.write_text(
            "from pathlib import Path\n"
            f"out=Path(__file__).resolve().parent.parent/'measurements/result_{i}.json'\n"
            f"out.write_text({json.dumps(report)!r})\n"
        )
        scripts.append({"id": f"probe-{i}", "script": script.name,
                        "args": [], "report": f"measurements/result_{i}.json"})
    return p, {"scripts": scripts}, requirements


def passing_aggregate():
    return [
        {"name": "stable", "measured": 0.01, "operator": "<=", "threshold": 0.1, "unit": "m/s"},
        {"name": "contact", "measured": 2, "operator": ">=", "threshold": 1, "unit": "rows"},
    ]


def test_auxiliary_probe_runs_before_full_unchanged_physics_aggregate(tmp_path):
    p, plan, req = setup_plan(tmp_path, passing_aggregate())
    receipts = p.execute_scripts(plan, req)
    assert len(receipts) == 2
    assert receipts[0]["validation_scope"] == "auxiliary-preflight"
    assert receipts[1]["validation_scope"] == "frozen-physics-aggregate"


@pytest.mark.parametrize("failure", ["missing", "threshold", "only_preflight"])
def test_auxiliary_success_cannot_replace_missing_or_weakened_frozen_checks(tmp_path, failure):
    aggregate = passing_aggregate()
    if failure == "missing":
        aggregate.pop()
    if failure == "threshold":
        aggregate[0]["threshold"] = 10
    p, plan, req = setup_plan(tmp_path, aggregate)
    if failure == "only_preflight":
        plan["scripts"].pop()
    with pytest.raises(GateError, match="measurement contract changed/missing"):
        p.execute_scripts(plan, req)


def test_failed_auxiliary_probe_stops_before_full_physics_launch(tmp_path):
    p, plan, req = setup_plan(tmp_path, passing_aggregate(), preflight_value=0)
    with pytest.raises(GateError, match="failed check api_probe"):
        p.execute_scripts(plan, req)
    assert not (p.workspace / "measurements/result_1.json").exists()
