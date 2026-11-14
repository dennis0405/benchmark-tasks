from copy import deepcopy

import pytest

from rcauto.gates import GateError
from rcauto.selftest import FixtureAgent, FixtureProduction
from rcauto.store import StateError, Store, write_json


def production(tmp_path):
    store = Store(tmp_path / "state")
    run = store.create("contract readiness", instance="test", mode="selftest")
    p = FixtureProduction(store, run["id"], FixtureAgent(), {"repair_attempts": 0})
    write_json(p.workspace / "input/request.json", run)
    return p


def test_duplicate_physics_check_is_rejected_before_contract_review(tmp_path):
    p = production(tmp_path)
    contract = p.agent.run("task-specifier", "", p.workspace, {})
    contract["physics_checks"].append(deepcopy(contract["physics_checks"][0]))
    p.ask = lambda *args, **kwargs: contract
    p.review = lambda *args: pytest.fail("invalid contract reached reviewer")
    with pytest.raises(GateError, match="duplicate physics check"):
        p.contract()


def test_control_steps_cannot_be_mistaken_for_physics_steps(tmp_path):
    p = production(tmp_path)
    contract = p.agent.run("task-specifier", "", p.workspace, {})
    contract["run_steps"] = 7200
    contract["timing"] = {
        "control_hz": 20,
        "physics_hz": 240,
        "refinement_physics_hz": 480,
        "episode_seconds": 120,
    }
    p.ask = lambda *args, **kwargs: contract
    with pytest.raises(GateError, match="episode duration"):
        p.contract()


def test_semantic_contract_rejection_blocks_environment(tmp_path):
    p = production(tmp_path)

    def reject(role, instruction, *, specification_only=False):
        assert role == "contract-critic"
        assert specification_only
        assert "initial" in instruction and "partial" in instruction
        assert "physical" in instruction
        raise GateError("initial score contradicts controls")

    p.review = reject
    for stage in p.STAGES[: p.STAGES.index("contract")]:
        p.store.stage_done(p.rid, stage, {"fixture": True})
    p.environment = lambda: pytest.fail("rejected contract reached environment")
    with pytest.raises(GateError, match="initial score"):
        p.run(from_stage="contract")


def test_edited_contract_cannot_reuse_a_prior_review(tmp_path):
    import json

    p = production(tmp_path)
    p.contract()
    path = p.artifact("contract")
    changed = json.loads(path.read_text())
    changed["physics_checks"][0]["threshold"] = 1
    write_json(path, changed)
    with pytest.raises(GateError, match="stale contract review"):
        p.environment()


def test_contract_reviewer_checks_plan_without_demanding_future_code(tmp_path):
    p = production(tmp_path)
    original = p.agent.run

    def reviewer(role, prompt, workspace, schema, writable=False):
        if role == "contract-critic":
            assert "PROSPECTIVE specification" in prompt
            assert "historical scaffold" in prompt
            assert "computable" in prompt
            return {
                "verdict": "repair",
                "findings": ["damage energy formula is undefined"],
                "evidence_paths": ["artifacts/contract.json"],
            }
        return original(role, prompt, workspace, schema, writable)

    p.agent.run = reviewer
    with pytest.raises(GateError, match="energy formula"):
        p.contract()
    assert not p.artifact("contract-preflight").exists()


@pytest.mark.parametrize("reported", ["probe.py", "task/probe.py"])
def test_diagnostic_accepts_both_confined_relative_path_forms(tmp_path, reported):
    import sys

    p = production(tmp_path)
    p.config["runtime"] = {"python": sys.executable}
    (p.workspace / "task/probe.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "p=Path(__file__).resolve().parent.parent/'measurements/probe.json'\n"
        "p.write_text(json.dumps({'checks':[{'name':'actual file executed',"
        "'measured':1,'operator':'==','threshold':1,'unit':'count'}]}))\n"
    )
    receipts = p.execute_scripts(
        {
            "scripts": [
                {"id": "probe", "script": reported, "args": [], "report": "measurements/probe.json"}
            ]
        },
        [{"name": "actual file executed", "operator": "==", "threshold": 1, "unit": "count"}],
    )
    assert receipts[0]["exit_code"] == 0


def test_diagnostic_workspace_prefix_cannot_escape_task(tmp_path):
    import sys

    p = production(tmp_path)
    p.config["runtime"] = {"python": sys.executable}
    (p.workspace / "input/unsafe.py").write_text("raise AssertionError('must not execute')\n")
    with pytest.raises(GateError):
        p.execute_scripts(
            {
                "scripts": [
                    {
                        "id": "escape",
                        "script": "task/../input/unsafe.py",
                        "args": [],
                        "report": "measurements/bad.json",
                    }
                ]
            }
        )


def test_cancel_after_failed_attempt_prevents_automatic_retry(tmp_path):
    p = production(tmp_path)
    p.config["repair_attempts"] = 2
    calls = []

    def failed_attempt():
        calls.append(1)
        write_json(p.root / "cancel.json", {"reason": "contract repair"})
        raise GateError("cannot satisfy current contract")

    p.environment = failed_attempt
    for stage in p.STAGES[: p.STAGES.index("environment")]:
        p.store.stage_done(p.rid, stage, {"fixture": True})
    p.run(from_stage="environment")
    assert calls == [1]
    assert p.store.get(p.rid)["status"] == "paused"


def test_cancel_after_codegen_prevents_native_launch(tmp_path):
    import sys

    p = production(tmp_path)
    p.config["runtime"] = {"python": sys.executable}
    marker = tmp_path / "native-was-launched"
    (p.workspace / "task/probe.py").write_text(
        "from pathlib import Path\n" + f"Path({str(marker)!r}).write_text('launched')\n"
    )
    write_json(p.root / "cancel.json", {"reason": "stop arrived during source generation"})
    with pytest.raises(StateError, match="cancelled before native launch"):
        p.execute_scripts(
            {"scripts": [{"id": "probe", "script": "probe.py", "args": [],
                          "report": "measurements/probe.json"}]}
        )
    assert not marker.exists()
