import pytest

from rcauto.coordinator import Coordinator
from rcauto.runtime import RuntimeErrorState
from rcauto.store import Store


def test_failed_deploy_stops_newly_started_instance_without_a_worker(tmp_path):
    class FakeBrev:
        stopped = False

        def start(self):
            pass

        def deploy(self, root):
            raise RuntimeErrorState("deployment failure")

        def ssh(self, script, **kw):
            return ""

        def rsync(self, *a, **kw):
            raise RuntimeErrorState("worker folder does not exist")

        def stop(self):
            self.stopped = True

    s = Store(tmp_path)
    r = s.create("x", instance="vm")
    c = Coordinator(
        s, {"instance": "vm", "instance_id": "id", "remote_root": "/data/test"}, tmp_path
    )
    c.brev = FakeBrev()
    c.sync = lambda: {"complete": True}
    c.prepare_input = lambda *a: None
    c.produce(r["id"])
    assert c.brev.stopped
    assert s.get(r["id"])["compute"] == "stopped"


def test_different_active_worker_is_never_stopped(tmp_path):
    class FakeBrev:
        def start(self):
            pass

        def ssh(self, script, **kw):
            return "BUSY"

        def stop(self):
            pytest.fail("must not stop someone else's active worker")

    s = Store(tmp_path)
    r = s.create("x", instance="vm")
    c = Coordinator(
        s, {"instance": "vm", "instance_id": "id", "remote_root": "/data/test"}, tmp_path
    )
    c.brev = FakeBrev()
    c.sync = lambda: {"complete": True}
    c.prepare_input = lambda *a: None
    c.produce(r["id"])
    assert s.get(r["id"])["status"] == "blocked"


def test_worker_download_replaces_managed_release_without_deleting_local_approval(tmp_path):
    import shutil

    s = Store(tmp_path / "state")
    r = s.create("fixture", instance="vm")
    root = s.run_dir(r["id"])
    (root / "release").mkdir()
    (root / "release/stale.py").write_text("stale")
    (root / "approval.json").write_text("local authority")
    remote = tmp_path / "remote"
    (remote / "release").mkdir(parents=True)
    (remote / "release/new.py").write_text("new")

    class FakeBrev:
        def rsync(self, src, dst, download=False):
            shutil.copytree(remote, dst, dirs_exist_ok=True, symlinks=True)

    c = Coordinator(s, {"instance": "vm", "instance_id": "id"}, tmp_path)
    c.brev = FakeBrev()
    c.download_run(r["id"], "/remote")
    assert not (root / "release/stale.py").exists()
    assert (root / "release/new.py").read_text() == "new"
    assert (root / "approval.json").read_text() == "local authority"


def test_downloaded_symlink_cannot_dereference_mac_credentials(tmp_path):
    s = Store(tmp_path / "state")
    r = s.create("fixture", instance="vm")

    class FakeBrev:
        def rsync(self, src, dst, download=False):
            from pathlib import Path

            (Path(dst) / "escape").symlink_to("/etc/passwd")

    c = Coordinator(s, {"instance": "vm", "instance_id": "id"}, tmp_path)
    c.brev = FakeBrev()
    from rcauto.gates import GateError

    with pytest.raises(GateError, match="symlink"):
        c.download_run(r["id"], "/remote")
