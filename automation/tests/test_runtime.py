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


def test_transient_transfer_failure_recovers_the_actual_payload(monkeypatch, tmp_path):
    from rcauto import runtime

    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    data = b"complete evidence\n" * 1000
    (source / "evidence.json").write_bytes(data)
    actual_capture = runtime.capture
    attempts = []

    def interrupted_then_local_transfer(argv, **kwargs):
        attempts.append(argv)
        if len(attempts) == 1:
            (target / "evidence.json").write_bytes(b"partial")
            raise RuntimeErrorState("rsync: unexpected end of file")
        local = [arg.removeprefix("vm:") for arg in argv]
        local[0] = "/usr/bin/rsync"
        return actual_capture(local, **kwargs)

    monkeypatch.setattr(runtime, "capture", interrupted_then_local_transfer)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    Brev({"instance": "vm"}).rsync(str(source) + "/", target, download=True)
    assert (target / "evidence.json").read_bytes() == data
    assert len(attempts) == 2


@pytest.mark.parametrize("error,expected_attempts", [
    ("rsync: unexpected end of file", 3),
    ("rsync: Permission denied", 1),
])
def test_transfer_retries_are_bounded_and_do_not_retry_permission_errors(monkeypatch, error, expected_attempts):
    from rcauto import runtime

    attempts = []

    def failed(argv, **kwargs):
        attempts.append(argv)
        raise RuntimeErrorState(error)

    monkeypatch.setattr(runtime, "capture", failed)
    monkeypatch.setattr(runtime.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeErrorState, match="rsync"):
        Brev({"instance": "vm"}).rsync("/source/", "/target/", download=True)
    assert len(attempts) == expected_attempts
