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
    pages = {"first": ([{"id": 1, "sha": "old"}], "last"),
             "last": ([{"id": 1, "sha": "new"}], None)}
    assert len(collect_pages("first", pages.__getitem__)) == 2


def test_open_pr_is_not_automatically_pending():
    task = normalize_board({"id": "T1", "status": "accepted", "title": "x", "instruction": "do x",
                            "setup": "x", "scoring": "y", "category": "Other"})
    assert task["status"] == "accepted"
    with pytest.raises(CatalogError):
        normalize_board({"title": "missing setup and scoring"})


def test_fetch_blocks_private_network_targets():
    for u in ['http://127.0.0.1/a','http://169.254.169.254/latest','file:///etc/passwd']:
        with pytest.raises(CatalogError):
            get_source(u)
