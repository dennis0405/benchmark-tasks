import pytest

from rcauto.catalog import CatalogError, collect_pages, get_source, normalize_board


def test_pagination_does_not_publish_partial_catalog():
    def fetch(path):
        if path == "next":
            raise CatalogError("403")
        return [{"id": 1}], "next"

    with pytest.raises(CatalogError):
        collect_pages("first", fetch)


def test_all_pages_preserve_duplicates_by_revision():
    pages = {
        "first": ([{"id": 1, "sha": "old"}], "last"),
        "last": ([{"id": 1, "sha": "new"}], None),
    }
    assert len(collect_pages("first", pages.__getitem__)) == 2


def test_open_pr_is_not_automatically_pending():
    task = normalize_board(
        {
            "id": "T1",
            "status": "accepted",
            "title": "x",
            "instruction": "do x",
            "setup": "x",
            "scoring": "y",
            "category": "Other",
        }
    )
    assert task["status"] == "accepted"
    with pytest.raises(CatalogError):
        normalize_board({"title": "missing setup and scoring"})


def test_fetch_blocks_private_network_targets():
    for u in ["http://127.0.0.1/a", "http://169.254.169.254/latest", "file:///etc/passwd"]:
        with pytest.raises(CatalogError):
            get_source(u)


def test_documented_draft_without_task_json_retains_original_rubric():
    import base64

    from rcauto.catalog import GitHub

    class DraftAPI(GitHub):
        def __init__(self, *, draft=True, missing=None):
            self.draft, self.missing = draft, missing

        def pages(self, endpoint, **kwargs):
            if endpoint.endswith("/files"):
                return [
                    {"filename": "tasks/draft/" + name}
                    for name in ("README.md", "task.py", "scoring.py")
                ]
            if endpoint.endswith("/commits"):
                return [{"sha": "head"}]
            return [
                {
                    "number": 1,
                    "html_url": "https://github.com/a/b/pull/1",
                    "title": "Draft physical operation",
                    "body": "Incomplete draft",
                    "state": "open",
                    "draft": self.draft,
                    "merged_at": None,
                    "user": {"login": "author"},
                    "head": {"sha": "head", "repo": {"full_name": "a/fork"}},
                }
            ]

        def get(self, endpoint, **kwargs):
            name = endpoint.split("?")[0].rsplit("/", 1)[-1]
            if name == self.missing:
                raise CatalogError("source unavailable")
            return {"content": base64.b64encode((name + " original source").encode()).decode()}

    result = DraftAPI().sync("a/b")
    assert result["complete"]
    record = result["records"][0]
    assert record["specs"] == []  # Never invent a formal submission or a rubric.
    assert record["state"] == "open"
    assert record["draft"] is True
    assert {s["path"] for s in record["source_documents"]} == {
        "tasks/draft/README.md",
        "tasks/draft/task.py",
        "tasks/draft/scoring.py",
    }
    assert all(s["sha"] == "head" and s["sha256"] for s in record["source_documents"])
    assert not DraftAPI(draft=False).sync("a/b")["complete"]
    assert not DraftAPI(missing="scoring.py").sync("a/b")["complete"]
