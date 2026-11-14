import json

import httpx
import pytest

from rcauto.board import collect_board, merge_board_catalog
from rcauto.catalog import CatalogError


def task(id, status, yours=False):
    return {
        "id": id,
        "status": status,
        "yours": yours,
        "title": "Extract a terminal",
        "instruction": "Extract",
        "setup": "Housing",
        "scoring": "Released and extracted",
        "category": "Impact on AI timelines",
        "feedback": "Private feedback must not enter worker inputs",
        "screenshot_url": "private-url",
    }


def test_board_keeps_authoritative_team_history_without_credentials_or_private_feedback():
    def transport(request):
        if request.url.path == "/api/login":
            assert json.loads(request.content)["token"] == "test-secret"
            return httpx.Response(200, json={"role": "team", "team": "private-team"})
        if request.url.path == "/api/board/summary":
            return httpx.Response(200, json={"accepted": 1, "pending": 1, "yours": {"pending": 1}})
        if request.url.path == "/api/team/tasks":
            return httpx.Response(200, json={"tasks": [{"id": "TK-OWN"}, {"id": "TK-OLD"}]})
        if request.url.params.get("id") == "TK-OLD":
            return httpx.Response(200, json={"tasks": [task("TK-OLD", "withdrawn", True)]})
        view = request.url.params.get("view")
        rows = {
            "accepted": [task("TK-A", "accepted")],
            "pending": [task("TK-OWN", "pending", True)],
            "revision": [],
        }[view]
        return httpx.Response(200, json={"tasks": rows})

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        snapshot = collect_board("team", "test-secret", client=client)
    assert snapshot["complete"]
    assert len(snapshot["records"]) == 3
    old = next(r for r in snapshot["records"] if r["id"] == "board-TK-OLD")
    assert old["own_team"] and old["board_status"] == "withdrawn"
    assert snapshot["own_history"] == 2
    serialized = json.dumps(snapshot)
    assert "test-secret" not in serialized and "private-team" not in serialized
    assert "Private feedback" not in serialized and "private-url" not in serialized


def test_board_auth_failure_never_means_empty_complete_catalog():
    with (
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401))) as client,
        pytest.raises(CatalogError, match="authentication"),
    ):
        collect_board("team", "secret", client=client)


def test_exact_board_match_promotes_team_identity_without_losing_distinct_versions():
    from copy import deepcopy

    board = {
        "complete": True,
        "records": [
            {
                "id": "board-A",
                "url": "board/A",
                "board_status": "rejected",
                "own_team": True,
                "specs": [{"sha": "board", "spec": {"title": "Task", "instruction": "Do work"}}],
            },
        ],
        "own_history": 1,
    }
    github = {
        "scope": "github",
        "records": [
            {
                "id": "pr-1",
                "own_team": False,
                "specs": [{"sha": "old", "spec": {"title": "TASK", "instruction": "Do  work"}}],
            },
        ],
    }
    changed = deepcopy(board["records"][0])
    changed["id"] = "board-B"
    changed["specs"][0]["spec"]["instruction"] = "Do different work"
    board["records"].append(changed)
    merged = merge_board_catalog(github, board)
    assert len(merged["records"]) == 2
    assert merged["records"][0]["own_team"]
    assert {r["sha"] for r in merged["records"][0]["specs"]} == {"old", "board"}
    assert merged["board"]["matched_github"] == 1


def test_pilot_definitions_do_not_invent_missing_setup_or_change_pending_count():
    def transport(request):
        if request.url.path == "/api/login":
            return httpx.Response(200, json={"role": "team"})
        if request.url.path == "/api/board/summary":
            return httpx.Response(200, json={"accepted": 0, "pending": 1, "yours": {}})
        if request.url.path == "/api/team/tasks":
            return httpx.Response(200, json={"tasks": []})
        rows = []
        if request.url.params.get("view") == "pending":
            rows = [
                task("TK-P", "pending"),
                {
                    "id": "TK-PILOT",
                    "status": "pilot",
                    "title": "Set a table",
                    "instruction": "Set a table with a human",
                },
            ]
        return httpx.Response(200, json={"tasks": rows})

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        snapshot = collect_board("team", "secret", client=client)
    pilot = next(r for r in snapshot["records"] if r["board_status"] == "pilot")
    assert set(pilot["missing_fields"]) == {"setup", "scoring"}
    assert pilot["specs"][0]["spec"]["scoring"] == ""
    assert snapshot["view_status_counts"]["pending"]["pending"] == 1


def test_archival_recovery_requires_trusted_receipt_and_uses_receipt_time_commit():
    import base64

    from rcauto.board import recover_own_history

    class API:
        trusted = True

        def pages(self, path, **kwargs):
            if path.endswith("comments"):
                return [
                    {
                        "user": {"login": "official" if self.trusted else "unknown"},
                        "body": "Received as **TK-X**",
                        "created_at": "2026-11-01T00:00:00Z",
                        "html_url": "https://example.com/receipt",
                    }
                ]
            if path.endswith("commits"):
                return [
                    {
                        "sha": "receipt-sha",
                        "commit": {"committer": {"date": "2026-10-31T00:00:00Z"}},
                    },
                    {"sha": "later-sha", "commit": {"committer": {"date": "2026-11-02T00:00:00Z"}}},
                ]
            return [{"filename": "tasks/example/task.json"}]

        def get(self, path, **kwargs):
            if "/pulls/" in path:
                return {
                    "head": {"sha": "head", "repo": {"full_name": "fork/repo"}},
                    "html_url": "https://example.com/pr",
                }
            assert "ref=receipt-sha" in path
            return {
                "content": base64.b64encode(
                    json.dumps({"title": "Task", "instruction": "Before receipt"}).encode()
                ).decode()
            }

    metadata = {"id": "TK-X", "title": "Task", "status": "withdrawn"}
    api = API()
    recovered = recover_own_history(metadata, api, "upstream/repo", 1, ["official"])
    assert recovered["instruction"] == "Before receipt"
    assert recovered["_source"]["commit"] == "receipt-sha"
    api.trusted = False
    with pytest.raises(CatalogError, match="trusted"):
        recover_own_history(metadata, api, "upstream/repo", 1, ["official"])
