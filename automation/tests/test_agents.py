import pytest

from rcauto.agents import Agent, clean_environment, command
from rcauto.store import StateError


def test_credentials_are_not_passed_to_production_agents(monkeypatch):
    monkeypatch.setenv("DREAM_API_KEY", "private")
    monkeypatch.setenv("ROBOCURVE_TOKEN", "private")
    monkeypatch.setenv("GH_TOKEN", "private")
    env = clean_environment()
    assert all(k not in env for k in ["DREAM_API_KEY", "ROBOCURVE_TOKEN", "GH_TOKEN"])
    assert env["PATH"]


def test_timeout_kills_process_and_preserves_log(tmp_path):
    import sys

    with pytest.raises(StateError, match="timeout"):
        command(
            [sys.executable, "-c", "import time;print('started',flush=True);time.sleep(30)"],
            tmp_path,
            tmp_path / "timeout.log",
            timeout=0.1,
        )
    assert "started" in (tmp_path / "timeout.log").read_text()


def test_invalid_agent_output_never_becomes_success(tmp_path):
    executable = tmp_path / "fake-codex"
    executable.write_text(
        '#!/usr/bin/env python3\nimport pathlib,sys\np=sys.argv[sys.argv.index("-o")+1]\npathlib.Path(p).write_text("not json")\n'
    )
    executable.chmod(0o700)
    a = Agent(str(executable), timeout=3)
    with pytest.raises(StateError, match="JSON"):
        a.run("research", "test", tmp_path, {"type": "object"})


def test_missing_tool_runtime_cannot_be_hidden_by_success_json(tmp_path):
    executable = tmp_path / "fake-codex"
    executable.write_text("""#!/usr/bin/env python3
import pathlib,sys,json
p=sys.argv[sys.argv.index('-o')+1]
pathlib.Path(p).write_text('{"ok":true}')
print(json.dumps({'type':'item.completed','item':{'type':'error','message':'Code Mode is unavailable because host executable was not found'}}))
""")
    executable.chmod(0o700)
    with pytest.raises(StateError, match="tool runtime"):
        Agent(str(executable), timeout=3).run("smoke", "test", tmp_path, {"type": "object"})
