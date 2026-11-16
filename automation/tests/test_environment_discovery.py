"""Native API discovery is useful evidence, never full environment qualification."""

import json
import sys

import pytest

from rcauto.gates import GateError
from rcauto.selftest import FixtureAgent, FixtureProduction
from rcauto.store import Store, write_json


def prepared(tmp_path):
    store = Store(tmp_path / "state")
    run = store.create("SELFTEST ONLY: environment discovery", "none", mode="selftest")
    p = FixtureProduction(store, run["id"], FixtureAgent(), {"runtime": {"python": sys.executable}})
    write_json(p.workspace / "input/request.json", run)
    p.contract()
    return p


def test_preflight_executes_then_full_qualification_still_uses_frozen_requirements(tmp_path):
    p = prepared(tmp_path)
    original = p.agent.run
    phases = []

    def agent(role, prompt, workspace, schema, writable=False):
        result = original(role, prompt, workspace, schema, writable)
        if role == "environment-builder":
            phase = "preflight" if not phases else "qualification"
            phases.append(phase)
            result["phase"] = phase
            if phase == "preflight":
                result.update(p.agent.scripts(workspace, {"api": [
                    {"name": "API fixture only", "operator": "==", "threshold": 0, "unit": "fixture"}
                ]}))
        return result

    p.agent.run = agent
    p.environment()
    assert phases == ["preflight", "qualification"]
    auxiliary = json.loads(p.artifact("environment-preflights").read_text())
    assert auxiliary[0]["qualification"] is False
    assert auxiliary[0]["receipts"][0]["id"] == "api"
    qualified = json.loads(p.artifact("environment").read_text())
    assert qualified[-1]["validation_scope"] == "frozen-physics-aggregate"


def test_repeated_discovery_cannot_be_reported_as_environment_success(tmp_path):
    p = prepared(tmp_path)
    original = p.agent.run

    def agent(role, prompt, workspace, schema, writable=False):
        result = original(role, prompt, workspace, schema, writable)
        if role == "environment-builder":
            result["phase"] = "preflight"
            # API discovery must not require unfinished production assets.
            (workspace / "task/asset-manifest.json").unlink()
        return result

    p.agent.run = agent
    with pytest.raises(GateError, match="qualification"):
        p.environment()
    assert not p.artifact("environment").exists()
    assert len(json.loads(p.artifact("environment-preflights").read_text())) == 3


def test_measurement_source_rejection_prevents_gpu_execution(tmp_path):
    p = prepared(tmp_path)
    original = p.review

    def review(role, instruction, **kwargs):
        if role == "component-code-reviewer":
            assert kwargs["implementation_only"]
            assert "rotation" in instruction and "torque" in instruction
            assert "preload" in instruction and "constant" in instruction
            raise GateError("a contact peak is not sustained vented preload")
        return original(role, instruction, **kwargs)

    p.review = review
    p.execute_scripts = lambda *a, **k: pytest.fail("rejected code reached GPU execution")
    with pytest.raises(GateError, match="preload"):
        p.environment()
