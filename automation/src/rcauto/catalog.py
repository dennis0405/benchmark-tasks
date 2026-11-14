"""Source acquisition and conservative GitHub catalog. Never publish partial snapshots."""

import hashlib
import ipaddress
import json
import socket
import subprocess
import sys
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader

from .gates import digest
from .store import StateError, now, write_json


class CatalogError(StateError):
    pass


class RateLimited(StateError):
    """Abort collection without advancing the last complete snapshot."""


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def public_url(url):
    u = urlparse(url)
    if u.scheme not in {"https", "http"} or not u.hostname or u.username or u.password:
        raise CatalogError("source requires a public HTTP(S) URL")
    try:
        addresses = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except OSError as e:
        raise CatalogError(f"source host unresolved: {u.hostname}") from e
    if any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise CatalogError("private network source refused")


def get_source(url, destination=None, max_bytes=25 * 1024 * 1024):
    original = url
    with httpx.Client(headers={"User-Agent": "robocurve-automation/0.1"}, timeout=60) as client:
        for _ in range(6):
            public_url(url)
            with client.stream("GET", url) as res:
                if res.is_redirect:
                    url = str(res.url.join(res.headers["location"]))
                    continue
                res.raise_for_status()
                raw = bytearray()
                for chunk in res.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > max_bytes:
                        raise CatalogError("source too large; acquire asset separately")
                content_type = res.headers.get("content-type", "")
                break
        else:
            raise CatalogError("too many source redirects")
    raw = bytes(raw)
    if "pdf" in content_type or raw.startswith(b"%PDF"):
        text = "\n".join(
            f"PAGE {i + 1}\n{p.extract_text() or ''}"
            for i, p in enumerate(PdfReader(BytesIO(raw)).pages)
        )
    elif "html" in content_type:
        parser = Text()
        parser.feed(raw.decode("utf-8", errors="replace"))
        text = "\n".join(parser.parts)
    else:
        text = raw.decode("utf-8", errors="replace")
    if len(text.strip()) < 100 or "<title>Sign in" in raw.decode(errors="ignore"):
        raise CatalogError("source text absent or login required")
    result = {
        "id": "s-" + hashlib.sha256(original.encode()).hexdigest()[:16],
        "url": original,
        "resolved_url": url,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "text": text,
        "content_type": content_type,
        "acquired_at": now(),
    }
    if destination:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / (result["id"] + ".raw")).write_bytes(raw)
        write_json(destination / (result["id"] + ".json"), result)
    return result


def collect_pages(first, fetch):
    result, visited = [], set()
    page = first
    while page:
        if page in visited:
            raise CatalogError("pagination loop")
        visited.add(page)
        items, page = fetch(page)
        if not isinstance(items, list):
            raise CatalogError("invalid page; cannot infer empty catalog")
        result.extend(items)
    return result


def normalize_board(task):
    for field in ["id", "status", "title", "instruction", "setup", "scoring"]:
        if not task.get(field):
            raise CatalogError(f"board record {task.get('id', '(unknown)')} lacks {field}")
    return dict(task)


class GitHub:
    def __init__(self, cache):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def get(self, endpoint, cached=False, revision=None):
        key = self.cache / (
            hashlib.sha256((endpoint + (":" + revision if revision else "")).encode()).hexdigest()
            + ".json"
        )
        if cached and key.exists():
            return json.loads(key.read_text())
        p = subprocess.run(
            ["gh", "api", endpoint], capture_output=True, text=True, timeout=120, check=False
        )
        if p.returncode:
            if "rate limit" in p.stderr.lower() or "HTTP 429" in p.stderr:
                raise RateLimited(
                    "GitHub rate limit reached; retain old snapshot and resume after reset"
                )
            raise CatalogError(f"GitHub request failed: {endpoint}: {p.stderr[-600:]}")
        value = json.loads(p.stdout)
        if cached:
            write_json(key, value)
        return value

    def pages(self, endpoint, cached=False, revision=None):
        def fetch(page):
            items = self.get(page, cached=cached, revision=revision)
            if not isinstance(items, list):
                raise CatalogError("expected GitHub array")
            page_number = int(page.rsplit("page=", 1)[1])
            next_page = page.rsplit("page=", 1)[0] + "page=" + str(page_number + 1)
            return items, next_page if len(items) == 100 else None

        return collect_pages(
            endpoint + ("&" if "?" in endpoint else "?") + "per_page=100&page=1", fetch
        )

    def sync(self, repo="robocurve/benchmark-tasks", own_authors=()):
        started = now()
        pulls = self.pages(f"repos/{repo}/pulls?state=all&sort=created&direction=asc")
        records, missing = [], []
        for index, pr in enumerate(pulls):
            if index % 25 == 0:
                print(f"catalog: {index}/{len(pulls)} PRs inspected", file=sys.stderr, flush=True)
            active = pr["state"] == "open" or bool(pr.get("merged_at"))
            record = {
                "id": f"pr-{pr['number']}",
                "url": pr["html_url"],
                "title": pr["title"],
                "body": pr.get("body") or "",
                "state": pr["state"],
                "status": "merged"
                if pr.get("merged_at")
                else "open-unclassified"
                if active
                else "closed",
                "own_team": pr["user"]["login"] in own_authors,
                "head_sha": pr["head"]["sha"],
                "specs": [],
            }
            if active or record["own_team"]:
                try:
                    files = self.pages(
                        f"repos/{repo}/pulls/{pr['number']}/files",
                        cached=True,
                        revision=pr["head"]["sha"],
                    )
                    if len(files) >= 3000:
                        raise CatalogError("PR file list truncated")
                    paths = [f["filename"] for f in files if f["filename"].endswith("/task.json")]
                    if not paths:
                        if not any(f["filename"].startswith("tasks/") for f in files):
                            record["excluded_detail_reason"] = (
                                "non-task PR; metadata still compared"
                            )
                            records.append(record)
                            continue
                        raise CatalogError("active task PR has no accessible task.json")
                    head_repo = pr["head"].get("repo")
                    if not head_repo and pr.get("merged_at"):
                        head_repo = {"full_name": repo}
                    if not head_repo:
                        raise CatalogError("head repository deleted")
                    revisions = [pr["head"]["sha"]]
                    if pr["state"] == "open":
                        commits = self.pages(
                            f"repos/{repo}/pulls/{pr['number']}/commits",
                            cached=True,
                            revision=pr["head"]["sha"],
                        )
                        if len(commits) >= 250:
                            raise CatalogError(
                                "PR commit history may be truncated at GitHub's 250-commit limit"
                            )
                        revisions = list(dict.fromkeys(c["sha"] for c in commits))
                    for sha in revisions:
                        for path in paths:
                            endpoint = f"repos/{head_repo['full_name']}/contents/{path}?ref={sha}"
                            try:
                                import base64

                                blob = self.get(endpoint, cached=True)
                                spec = json.loads(base64.b64decode(blob["content"]))
                            except (CatalogError, KeyError, ValueError):
                                # A task may be introduced in a later commit. Require at least its head below.
                                continue
                            record["specs"].append({"sha": sha, "path": path, "spec": spec})
                    if not record["specs"] or not any(
                        x["sha"] == pr["head"]["sha"] for x in record["specs"]
                    ):
                        raise CatalogError("head task specification missing")
                except (CatalogError, KeyError) as e:
                    missing.append({"pr": pr["number"], "reason": str(e)})
            records.append(record)
        snapshot = {
            "scope": "all PR metadata; all open/merged and registered team task specs",
            "repo": repo,
            "started_at": started,
            "completed_at": now(),
            "records": records,
            "missing": missing,
            "complete": not missing,
            "source": "github-conservative-all-open",
        }
        snapshot["digest"] = digest(snapshot)
        return snapshot
