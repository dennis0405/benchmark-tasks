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


def test_full_episode_can_use_more_than_eighty_action_chunks(tmp_path):
    g = EpisodeGuard(tmp_path / "journal", None)
    g.start()
    # 7,200 controls need at least 150 full 48-row SDK action chunks.
    # Short replies may require more requests within the same physical episode.
    for _ in range(151):
        g.request()
    entries = [json.loads(line) for line in (tmp_path / "journal").read_text().splitlines()]
    assert g.requests == 151
    assert entries[-1]["request"] == 151
    assert sum(e["kind"] == "episode-start" for e in entries) == 1
    with pytest.raises(RuntimeError, match="second"):
        g.start()


def test_request_limit_is_optional_by_default(tmp_path):
    g = EpisodeGuard(tmp_path / "journal")
    g.start()
    for _ in range(81):
        g.request()
    assert g.requests == 81
