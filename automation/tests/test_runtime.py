import pytest

from rcauto.runtime import Brev, RuntimeErrorState


def test_instance_id_mismatch_prevents_start_or_stop():
    class B(Brev):
        def inventory(self):
            return [{"id": "wrong", "name": "vm", "status": "STOPPED"}]

    b = B({"instance": "vm", "instance_id": "expected"})
    with pytest.raises(RuntimeErrorState, match="identity"):
        b.status()


def test_missing_instance_does_not_create_replacement():
    class B(Brev):
        def inventory(self):
            return []

    with pytest.raises(RuntimeErrorState, match="missing"):
        B({"instance": "vm", "instance_id": "id"}).status()


def test_foreign_instance_name_is_not_accepted():
    with pytest.raises(RuntimeErrorState):
        Brev({"instance": "https://brev.nvidia.com/new", "instance_id": "id"})


def test_ssh_payload_does_not_interpolate_prompt_as_shell_code():
    b = Brev({"instance": "vm", "instance_id": "id"})
    argv = b.ssh_argv("python3 -c 'print(1)'")
    assert argv[-1] == "python3 -c 'print(1)'"
    assert argv[-2] == "vm"


@pytest.mark.parametrize("expected", ["RUNNING", "STOPPED"])
def test_terminal_provider_failure_does_not_wait_until_startup_timeout(monkeypatch, expected):
    from rcauto import runtime

    class FailedBrev(Brev):
        def status(self):
            return {"status": "FAILURE", "health_status": "UNHEALTHY"}

    monkeypatch.setattr(runtime.time, "sleep", lambda _: pytest.fail("terminal failure must fail fast"))
    b = FailedBrev({"instance": "vm", "instance_id": "id"})
    with pytest.raises(RuntimeErrorState, match="FAILURE"):
        b.wait(expected)


def test_start_returns_control_to_our_state_monitor_instead_of_blocking_cli(monkeypatch):
    from rcauto import runtime

    calls = []
    monkeypatch.setattr(runtime, "capture", lambda argv, **kw: calls.append(argv) or "")

    class TestBrev(Brev):
        def status(self):
            return {"status": "STOPPED"}

        def wait(self, expected):
            assert expected == "RUNNING"

        def ssh(self, script):
            assert script == "true"

    TestBrev({"instance": "vm", "instance_id": "id"}).start()
    assert calls[0] == ["brev", "start", "vm", "--detached"]
