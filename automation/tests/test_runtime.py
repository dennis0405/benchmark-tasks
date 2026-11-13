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
