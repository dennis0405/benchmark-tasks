import pytest

from rcauto.gates import GateError
from rcauto.publish import Publisher
from rcauto.store import Store


def test_submit_without_approval_cannot_make_external_requests(tmp_path):
    s = Store(tmp_path)
    r = s.create("task", instance="vm")
    p = Publisher(s, {"fork": "user/fork"})
    with pytest.raises((GateError, FileNotFoundError)):
        p.submit(r["id"], "forged", fresh_novelty=lambda: None)
    assert s.effect(r["id"], "upload") is None
    assert s.effect(r["id"], "pr") is None


def test_fresh_novelty_failure_stops_before_upload(tmp_path):
    from rcauto.engine import approve
    from rcauto.gates import manifest
    from rcauto.store import write_json

    s = Store(tmp_path)
    r = s.create("task", instance="vm")
    root = s.run_dir(r["id"])
    (root / "release").mkdir()
    (root / "release" / "task.py").write_text("code")
    write_json(root / "release-manifest.json", manifest(root / "release"))
    write_json(root / "report.json", {})
    s.update(r["id"], status="awaiting-review")
    a = approve(s, r["id"], "user", "submit these files")

    def blocked():
        raise GateError("new duplicate")

    with pytest.raises(GateError, match="duplicate"):
        Publisher(s, {"fork": "user/fork"}).submit(r["id"], a["id"], fresh_novelty=blocked)
    assert s.effect(r["id"], "upload") is None


def test_existing_main_checkout_is_never_reused_for_push(tmp_path):
    import subprocess

    s = Store(tmp_path)
    r = s.create("task", instance="vm")
    repo = s.run_dir(r["id"]) / "automation-checkout"
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    with pytest.raises(GateError, match="branch mismatch"):
        Publisher(s, {"fork": "user/fork"}).checkout(r["id"], "automation")


def test_fork_checkpoint_pushes_exact_task_and_verifies_remote_sha(tmp_path):
    import subprocess

    from rcauto.gates import manifest
    from rcauto.store import write_json

    def git(*args):
        return subprocess.run(
            ["git", *map(str, args)], check=True, capture_output=True, text=True
        ).stdout.strip()

    origin = tmp_path / "origin.git"
    git("init", "--bare", origin)
    work = tmp_path / "work"
    git("clone", origin, work)
    git("-C", work, "config", "user.name", "fixture")
    git("-C", work, "config", "user.email", "fixture@example.invalid")
    git("-C", work, "checkout", "-b", "automation/fixture")
    (work / "README.md").write_text("fixture baseline")
    git("-C", work, "add", "README.md")
    git("-C", work, "commit", "-m", "baseline")
    s = Store(tmp_path / "state")
    r = s.create("fixture", instance="none", mode="selftest")
    root = s.run_dir(r["id"])
    task = root / "release/tasks/fixture"
    task.mkdir(parents=True)
    (task / "task.py").write_text("# fixture")
    write_json(root / "release-manifest.json", manifest(root / "release"))

    class LocalPublisher(Publisher):
        def checkout(self, *args):
            return work

    publisher = LocalPublisher(s, {"fork": "fixture/fixture"})
    first = publisher.checkpoint(r["id"])
    assert first["sha"] == git("--git-dir", origin, "rev-parse", "refs/heads/automation/fixture")
    assert (work / "tasks/fixture/task.py").read_text() == "# fixture"
    assert publisher.checkpoint(r["id"])["sha"] == first["sha"]
