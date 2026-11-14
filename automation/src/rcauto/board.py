"""Authenticated read-only board snapshots. Credentials stay in the Mac controller."""

import base64
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx

from .catalog import CatalogError, normalize_board
from .gates import digest
from .store import now


def collect_board(team, token, client=None, own_detail=None):
    if client is None:
        with httpx.Client(timeout=45) as owned:
            return collect_board(team, token, client=owned, own_detail=own_detail)
    base = "https://tasks.robocurve.org"
    response = client.post(base + "/api/login", json={"code": team, "token": token})
    if response.status_code != 200:
        raise CatalogError("board authentication failed; credentials are not logged")

    def get(path, **params):
        response = client.get(base + path, params=params)
        response.raise_for_status()
        return response.json()

    before = get("/api/board/summary")
    tasks, views, counts = {}, {}, {}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    ignored_history = 0
    for view in ("accepted", "pending", "revision"):
        seen = set()
        statuses = Counter()
        for offset in range(0, 10000, 100):
            page = get("/api/tasks", view=view, limit=100, offset=offset)["tasks"]
            for task in page:
                if task["id"] in seen:
                    raise CatalogError("board pagination changed/repeated; snapshot incomplete")
                seen.add(task["id"])
                statuses[task["status"]] += 1
                # The revision view may also return archive/pilot entries. Only
                # current claims belong here; own history is acquired below.
                if view == "revision":
                    held = task["status"] == "rejected" and not task.get("final_reject")
                    if held and task.get("reviewed_at"):
                        reviewed = datetime.fromisoformat(
                            task["reviewed_at"].replace("Z", "+00:00")
                        )
                        held = reviewed.replace(tzinfo=reviewed.tzinfo or timezone.utc) >= cutoff
                    if not held:
                        ignored_history += 1
                        continue
                if task["status"] == "pilot":
                    # Official pilot benchmark definitions expose instructions,
                    # not invented setup/rubric fields. Review their actual limits.
                    if not all(task.get(k) for k in ("id", "status", "title", "instruction")):
                        raise CatalogError("pilot task definition missing")
                else:
                    normalize_board(task)
                tasks[task["id"]] = task
            if len(page) < 100:
                break
        else:
            raise CatalogError("board pagination limit reached")
        views[view] = len(seen)
        counts[view] = dict(statuses)
    own = get("/api/team/tasks")["tasks"]
    for metadata in own:
        tid = metadata["id"]
        if tid not in tasks:
            rows = get("/api/tasks", id=tid)["tasks"]
            if len(rows) != 1 or rows[0]["id"] != tid:
                if own_detail is None:
                    raise CatalogError("own board history detail unavailable: " + tid)
                rows = [own_detail(metadata)]
            normalize_board(rows[0])
            tasks[tid] = rows[0]
        tasks[tid]["yours"] = True
    after = get("/api/board/summary")
    for key in ("accepted", "pending"):
        if before[key] != after[key] or counts[key].get(key, 0) != after[key]:
            raise CatalogError("board counts changed/incomplete: " + key)
    if before.get("yours") != after.get("yours"):
        raise CatalogError("team history changed during board snapshot")
    fields = (
        "title",
        "category",
        "keywords",
        "instruction",
        "context",
        "setup",
        "scoring",
        "run_limit",
        "fidelity",
        "reset_notes",
    )
    records = []
    for tid, task in sorted(tasks.items()):
        spec = {k: task.get(k) or "" for k in fields}
        accepted = task["status"] == "accepted"
        yours = bool(task.get("yours"))
        records.append(
            {
                "id": "board-" + tid,
                "board_task_id": tid,
                "url": base + "/board?task=" + tid,
                "title": task["title"],
                "body": task["instruction"],
                "own_team": yours,
                "state": "closed"
                if accepted or (yours and task["status"] not in {"pending", "accepted"})
                else "open",
                "status": "merged"
                if accepted
                else "closed"
                if yours and task["status"] not in {"pending", "accepted"}
                else "open-unclassified",
                "board_status": task["status"],
                "board_required": True,
                "missing_fields": [k for k in ("setup", "scoring") if not task.get(k)],
                "recovered_source": task.get("_source"),
                "specs": [
                    {"sha": digest(spec), "path": "board/" + tid + "/task.json", "spec": spec}
                ],
            }
        )
    return {
        "complete": True,
        "views": views,
        "view_status_counts": counts,
        "own_history": len(own),
        "ignored_nonheld_revision_history": ignored_history,
        "records": records,
        "at": now(),
    }


def merge_board_catalog(snapshot, board):
    # Exact task text only: retrieval is never a semantic novelty verdict.
    normalize = lambda s: " ".join(str(s).lower().split())
    index = {}
    for record in snapshot["records"]:
        for entry in record.get("specs", []):
            spec = entry["spec"]
            index.setdefault(
                (normalize(spec.get("title")), normalize(spec.get("instruction"))), record
            )
    matched = 0
    for record in board["records"]:
        spec = record["specs"][0]["spec"]
        target = index.get((normalize(spec["title"]), normalize(spec["instruction"])))
        if target is None:
            snapshot["records"].append(record)
        else:
            target.setdefault("board_records", []).append(
                {k: record[k] for k in ("id", "url", "board_status")}
            )
            target["own_team"] = target["own_team"] or record["own_team"]
            target["board_required"] = True
            target["specs"].extend(record["specs"])
            matched += 1
    snapshot["board"] = {k: v for k, v in board.items() if k != "records"}
    snapshot["board"]["matched_github"] = matched
    snapshot["scope"] += "; authenticated board accepted/pending/revision and complete own history"
    snapshot["digest"] = digest(snapshot["records"])
    return snapshot


def recover_own_history(metadata, api, repo, number, trusted_authors):
    """Recover hidden/withdrawn text only from its verified public receipt and commit."""
    comments = api.pages(f"repos/{repo}/issues/{number}/comments")
    receipts = [
        c
        for c in comments
        if c.get("user", {}).get("login") in trusted_authors
        and re.search(
            r"Received as\s+\*\*" + re.escape(metadata["id"]) + r"\*\*", c.get("body", "")
        )
    ]
    if not receipts:
        raise CatalogError("configured archival PR has no trusted matching receipt")
    receipt = receipts[0]
    received_at = datetime.fromisoformat(receipt["created_at"].replace("Z", "+00:00"))
    pr = api.get(f"repos/{repo}/pulls/{number}")
    head = pr["head"]["sha"]
    commits = api.pages(f"repos/{repo}/pulls/{number}/commits", cached=True, revision=head)
    eligible = [
        c
        for c in commits
        if datetime.fromisoformat(c["commit"]["committer"]["date"].replace("Z", "+00:00"))
        <= received_at
    ]
    if not eligible:
        raise CatalogError("no archival commit predates the official receipt")
    sha = eligible[-1]["sha"]
    files = api.pages(f"repos/{repo}/pulls/{number}/files", cached=True, revision=head)
    source_repo = (pr["head"].get("repo") or {}).get("full_name")
    if not source_repo:
        raise CatalogError("archival fork is unavailable")
    for f in files:
        path = f["filename"]
        if not path.startswith("tasks/") or not path.endswith("/task.json"):
            continue
        blob = api.get(f"repos/{source_repo}/contents/{quote(path)}?ref={sha}", cached=True)
        spec = json.loads(base64.b64decode(blob["content"]))
        if " ".join(spec.get("title", "").lower().split()) != " ".join(
            metadata["title"].lower().split()
        ):
            continue
        return {
            **spec,
            "id": metadata["id"],
            "status": metadata["status"],
            "yours": True,
            "_source": {
                "pr": pr["html_url"],
                "receipt": receipt["html_url"],
                "commit": sha,
                "path": path,
                "spec_digest": digest(spec),
            },
        }
    raise CatalogError("receipt commit does not contain the matching task title")
