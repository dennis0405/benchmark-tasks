import json

import pytest

from rcauto.coordinator import Coordinator
from rcauto.gates import GateError, validate_media
from rcauto.store import Store


def test_review_report_escapes_user_text_and_links_actual_bundle(tmp_path):
    store = Store(tmp_path)
    r = store.create('<script>alert("x")</script>', instance="none")
    c = Coordinator(store, {"instance": "vm", "instance_id": "id"}, tmp_path)
    c.report(r["id"], "blocked", '<img src=x onerror="x">')
    html = (store.run_dir(r["id"]) / "report.html").read_text()
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert json.loads((store.run_dir(r["id"]) / "report.json").read_text())["status"] == "blocked"


def test_fake_mp4_is_not_release_media(tmp_path):
    p = tmp_path / "fake.mp4"
    p.write_text("fake success")
    with pytest.raises(GateError):
        validate_media(p, "video")
