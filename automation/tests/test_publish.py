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
