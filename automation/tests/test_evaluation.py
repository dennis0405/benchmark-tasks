import json

import pytest

from rcauto.evaluation import EpisodeGuard


def test_episode_cannot_restart_after_failure(tmp_path):
    g = EpisodeGuard(tmp_path / "journal", 2)
    g.start()
    g.request()
    with pytest.raises(RuntimeError, match="second"):
        g.start()
    assert len((tmp_path / "journal").read_text().splitlines()) == 2


def test_wire_attempts_count_even_if_reply_is_lost(tmp_path):
    g = EpisodeGuard(tmp_path / "journal", 1)
    g.start()
    g.request()
    with pytest.raises(RuntimeError, match="guard"):
        g.request()
    assert json.loads((tmp_path / "journal").read_text().splitlines()[-1])["request"] == 1
