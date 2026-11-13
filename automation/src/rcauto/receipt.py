"""Only configured receipt authors can establish receipt; merging establishes acceptance."""

import json
import re
import time

from .runtime import capture
from .store import write_json


def classify(value, trusted_authors):
    if value.get("mergedAt"):
        return {"status": "accepted", "merged_at": value["mergedAt"]}
    for c in reversed(value.get("comments", [])):
        if (c.get("author") or {}).get("login") not in trusted_authors:
            continue
        match = re.search(r"Received as\s+\*\*(TK-[A-Z0-9]+)\*\*", c.get("body", ""))
        if match:
            return {
                "status": "received-waiting-review",
                "task_id": match.group(1),
                "source_comment": c.get("url"),
                "author": c["author"]["login"],
            }
    return {"status": "submitted-awaiting-receipt", "github_state": value.get("state")}


def poll(store, rid, authors, timeout=0):
    run = store.get(rid)
    deadline = time.monotonic() + timeout
    while True:
        value = json.loads(
            capture(["gh", "pr", "view", run["pr"], "--json", "comments,state,mergedAt"])
        )
        result = classify(value, authors)
        write_json(store.run_dir(rid) / "receipt.json", {"parsed": result, "raw": value})
        store.update(rid, status=result["status"], receipt=result)
        if result["status"] != "submitted-awaiting-receipt" or time.monotonic() >= deadline:
            return dict(result, pr=run["pr"])
        time.sleep(min(15, max(0, deadline - time.monotonic())))
