import pytest

from rcauto.coordinator import Coordinator
from rcauto.runtime import RuntimeErrorState
from rcauto.store import Store


def test_missing_team_auth_cannot_claim_complete_required_board_coverage(tmp_path, monkeypatch):
    from rcauto import coordinator, secrets
    from rcauto.gates import GateError

    class API:
        def __init__(self, *args):
            pass

        def sync(self, *args, **kwargs):
            return {"complete": True, "missing": [], "records": [], "scope": "github"}

    monkeypatch.setattr(coordinator, "GitHub", API)
    monkeypatch.setattr(secrets, "credentials", lambda *a: {})
    c = Coordinator(Store(tmp_path / "state"), {"instance": "vm", "instance_id": "id"}, tmp_path)
    with pytest.raises(GateError, match="catalog incomplete"):
        c.sync()


def test_source_review_queue_survives_replaced_worker_input_without_overwriting_authority(tmp_path):
    import json

    s = Store(tmp_path / "state")
    r = s.create("original objective", instance="vm", mode="selftest")
    root = s.run_dir(r["id"])
    queue = root / "pending-source-review"
    queue.mkdir()
    (queue / "novelty-review-notes.json").write_text('{"pr-1":{"source":"original"}}')
    c = Coordinator(s, {"instance": "vm", "instance_id": "id", "runtime": {}}, tmp_path)
    for _ in range(2):
        (root / "workspace/input").mkdir(parents=True, exist_ok=True)
        (root / "workspace/input/novelty-review-notes.json").write_text("{}")
        c.prepare_input(r["id"], {"complete": True})
        assert json.loads((root / "workspace/input/novelty-review-notes.json").read_text()) == {
            "pr-1": {"source": "original"},
        }
    (queue / "request.json").write_text('{"prompt":"different objective"}')
    from rcauto.gates import GateError

    with pytest.raises(GateError, match="source-review"):
        c.prepare_input(r["id"], {"complete": True})
    assert (
        json.loads((root / "workspace/input/request.json").read_text())["prompt"]
        == "original objective"
    )


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


def test_stop_arriving_during_startup_prevents_deploy_or_worker_launch(tmp_path):
    from rcauto.store import write_json

    s = Store(tmp_path / "state")
    r = s.create("x", instance="vm")

    class FakeBrev:
        stopped = False

        def start(self):
            write_json(s.run_dir(r["id"]) / "cancel-requested.json", {"requested": "operator"})

        def ssh(self, *a, **kw):
            pytest.fail("stop requested at startup; no deployment or remote launch")

        def stop(self):
            self.stopped = True

    c = Coordinator(
        s, {"instance": "vm", "instance_id": "id", "remote_root": "/data/test"}, tmp_path
    )
    c.brev = FakeBrev()
    c.sync = lambda: {"complete": True}
    c.prepare_input = lambda *a: None
    c.produce(r["id"])
    assert c.brev.stopped
    assert "stop requested" in s.get(r["id"])["error"]
    assert s.effect(r["id"], "dream") is None


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


def test_pending_stop_prevents_dream_preparation_and_budget_reservation(tmp_path):
    from rcauto.gates import GateError
    from rcauto.store import write_json

    s = Store(tmp_path / "state")
    r = s.create("native production", instance="vm")
    write_json(s.run_dir(r["id"]) / "cancel-requested.json", {"requested": "human"})
    c = Coordinator(s, {"instance": "vm", "instance_id": "id"}, tmp_path)
    with pytest.raises(GateError, match="stop requested before Dream"):
        c.evaluate(r["id"], credentials={"DREAM_API_KEY": "unused-test-value"})
    assert s.effect(r["id"], "dream") is None


def test_stop_arriving_during_evaluation_transfer_prevents_dream_call(tmp_path):
    from rcauto.gates import GateError, manifest
    from rcauto.store import write_json

    s = Store(tmp_path / "state")
    r = s.create("native production", instance="vm")
    root = s.run_dir(r["id"])
    task = root / "release/tasks/probe"
    task.mkdir(parents=True)
    (task / "task.py").write_text("# validated fixture for cancellation test\n")
    write_json(task / "task.json", {"instruction": "fixture", "setup": "fixture",
                                    "scoring": "fixture", "run_limit": "7200"})
    write_json(root / "release-manifest.json", manifest(root / "release"))
    write_json(root / "workspace/artifacts/contract.json", {"run_steps": 7200})

    class FakeBrev:
        def rsync(self, *args, **kwargs):
            write_json(root / "cancel-requested.json", {"requested": "human"})

        def ssh(self, *args, **kwargs):
            pytest.fail("Dream must not launch after a stop during transfer")

    c = Coordinator(s, {"instance": "vm", "instance_id": "id", "remote_root": "/test",
                         "dream_episode_reserve_usd": 20, "dream_cumulative_usd": 20,
                         "runtime": {"python": "/usr/bin/python3"}}, tmp_path)
    c.brev = FakeBrev()
    with pytest.raises(GateError, match="stop requested before Dream"):
        c.evaluate(r["id"], credentials={"DREAM_API_KEY": "unused-test-value"})
    assert s.effect(r["id"], "dream") is None
