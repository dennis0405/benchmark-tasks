import pytest

from rcauto import board
from rcauto.catalog import CatalogError, RateLimited

META = {"id": "TK-ABCDEF", "title": "Original task", "status": "withdrawn"}


def item(number):
    return {
        "number": number,
        "html_url": f"https://github.com/team/tasks/pull/{number}",
        "pull_request": {"url": f"https://api.github.com/repos/team/tasks/pulls/{number}"},
    }


def receipt(author="official", task="TK-ABCDEF"):
    return {"user": {"login": author}, "body": f"Received as **{task}**, and waiting for review."}


class API:
    def __init__(self, candidates, comments, *, incomplete=False):
        self.candidates, self.comments, self.incomplete = candidates, comments, incomplete

    def get(self, endpoint):
        from urllib.parse import parse_qs, urlsplit

        query = parse_qs(urlsplit(endpoint).query)["q"][0]
        assert 'repo:team/tasks is:pr "TK-ABCDEF" in:comments' == query
        return {
            "items": self.candidates,
            "total_count": len(self.candidates),
            "incomplete_results": self.incomplete,
        }

    def pages(self, endpoint):
        number = int(endpoint.split("/")[-2])
        value = self.comments[number]
        if isinstance(value, Exception):
            raise value
        return value


def test_only_exact_trusted_receipt_can_select_original_commit_recovery(monkeypatch):
    api = API(
        [item(1), item(2), item(3)],
        {1: [receipt("someone")], 2: [receipt()], 3: [receipt(task="TK-ABCDEFX")]},
    )
    calls = []

    def recover(metadata, api, repo, number, authors):
        calls.append((metadata, repo, number, authors))
        return {"recovered_original": True}

    monkeypatch.setattr(board, "recover_own_history", recover)
    result = board.discover_own_history(META, api, "team/tasks", ["official"])
    assert result == {"recovered_original": True}
    assert calls == [(META, "team/tasks", 2, ["official"])]


def test_two_official_receipts_are_ambiguous_not_an_arbitrary_first_match(monkeypatch):
    api = API([item(1), item(2)], {1: [receipt()], 2: [receipt()]})
    monkeypatch.setattr(board, "recover_own_history", lambda *a: pytest.fail("ambiguous receipt"))
    with pytest.raises(CatalogError, match="ambiguous"):
        board.discover_own_history(META, api, "team/tasks", ["official"])


def test_untrusted_or_missing_receipt_cannot_recover_history():
    api = API([item(1)], {1: [receipt("someone")]})
    with pytest.raises(CatalogError, match="official receipt"):
        board.discover_own_history(META, api, "team/tasks", ["official"])


def test_incomplete_search_cannot_certify_unique_receipt():
    api = API([item(1)], {1: [receipt()]}, incomplete=True)
    with pytest.raises(CatalogError, match="incomplete"):
        board.discover_own_history(META, api, "team/tasks", ["official"])


def test_wrong_repository_result_is_rejected_before_comment_acquisition():
    candidate = item(1)
    candidate["html_url"] = "https://github.com/other/tasks/pull/1"
    api = API([candidate], {})
    with pytest.raises(CatalogError, match="repository"):
        board.discover_own_history(META, api, "team/tasks", ["official"])


@pytest.mark.parametrize("error", [RateLimited("limit"), CatalogError("HTTP503")])
def test_transport_failure_is_never_silently_treated_as_a_nonmatching_candidate(error):
    api = API([item(1)], {1: error})
    with pytest.raises(type(error), match=str(error)):
        board.discover_own_history(META, api, "team/tasks", ["official"])
