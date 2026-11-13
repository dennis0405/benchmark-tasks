"""Preserve in the fork; publish only the exact reviewed task after explicit approval."""

import json
import os
import shutil
import subprocess

from .engine import approval_valid, read
from .gates import GateError, file_hash, manifest, validate_spec, verify_manifest
from .runtime import capture
from .store import write_json


class Publisher:
    def __init__(self, store, config):
        self.store, self.config = store, config

    def git(self, repo, *args):
        return capture(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "credential.helper=",
                "-c",
                "credential.helper=!gh auth git-credential",
                *args,
            ],
            timeout=300,
        )

    def checkout(self, rid, kind):
        root = self.store.run_dir(rid)
        repo = root / (kind + "-checkout")
        expected = f"{kind}/{rid}"
        initializing = root / (kind + "-initializing")
        if not repo.exists():
            initializing.touch()
            capture(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    "https://github.com/" + self.config["fork"] + ".git",
                    str(repo),
                ],
                timeout=300,
            )
        current = self.git(repo, "branch", "--show-current").strip()
        if current != expected and not initializing.exists():
            raise GateError("private checkout branch mismatch; refusing to push a different branch")
        if current != expected:
            self.git(repo, "fetch", "https://github.com/robocurve/benchmark-tasks.git", "main")
            self.git(repo, "sparse-checkout", "init", "--cone")
            self.git(repo, "sparse-checkout", "set", "tools")
            self.git(repo, "checkout", "-b", expected, "FETCH_HEAD")
            self.git(repo, "config", "rcauto.base", self.git(repo, "rev-parse", "HEAD").strip())
        if initializing.exists():
            self.git(
                repo, "config", "user.name", self.config.get("git_name", "RoboCurve contributor")
            )
            self.git(
                repo,
                "config",
                "user.email",
                self.config.get("git_email", "contributor@users.noreply.github.com"),
            )
            initializing.unlink()
        origin = (
            self.git(repo, "remote", "get-url", "origin")
            .strip()
            .replace("git@github.com:", "https://github.com/")
            .removesuffix(".git")
        )
        if origin != "https://github.com/" + self.config["fork"]:
            raise GateError("checkpoint remote does not match registered fork")
        return repo

    def push(self, repo):
        self.git(repo, "push", "-u", "origin", "HEAD")
        local = self.git(repo, "rev-parse", "HEAD").strip()
        branch = self.git(repo, "branch", "--show-current").strip()
        remote = self.git(repo, "ls-remote", "origin", "refs/heads/" + branch).split()[0]
        if local != remote:
            raise GateError("origin SHA verification failed")
        return {
            "branch": branch,
            "sha": local,
            "url": f"https://github.com/{self.config['fork']}/tree/{branch}",
        }

    def checkpoint(self, rid):
        root = self.store.run_dir(rid)
        release = root / "release"
        if release.exists():
            verify_manifest(release, read(root / "release-manifest.json"))
        repo = self.checkout(rid, "automation")
        if (release / "tasks").exists():
            for task in (release / "tasks").iterdir():
                target = repo / "tasks" / task.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(task, repo / "tasks" / task.name, dirs_exist_ok=True)
                self.git(repo, "add", "--sparse", "--", "tasks/" + task.name)
        destination = repo / "automation-results" / rid
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        # Preserve only explicit output files. Never rsync credentials/state/home into a public fork.
        for name in ["release", "release-manifest.json", "report.json", "report.html", "report.md"]:
            source = root / name
            target = destination / name
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            elif source.is_file():
                shutil.copy2(source, target)
        artifacts = root / "workspace" / "artifacts"
        if artifacts.exists():
            shutil.copytree(artifacts, destination / "artifacts", dirs_exist_ok=True)
        for source, name in [
            (root / "workspace/measurements", "measurements"),
            (root / "logs", "logs"),
        ]:
            if source.exists():
                shutil.copytree(source, destination / name, dirs_exist_ok=True)
        dream = root / "dream-result"
        if dream.exists():
            for p in dream.rglob("*"):
                if p.is_file() and p.suffix in {".json", ".jsonl", ".mp4"}:
                    target = destination / "dream-result" / p.relative_to(dream)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
        # A commit cannot contain its own SHA. Omit bookkeeping-only push events from its payload.
        journal_run = {
            k: v for k, v in self.store.get(rid).items() if k not in {"checkpoint", "updated_at"}
        }
        journal_events = [
            e for e in self.store.events(rid) if set(json.loads(e["data"])) != {"checkpoint"}
        ]
        write_json(
            destination / "execution-journal.json",
            {
                "run": journal_run,
                "events": journal_events,
                "effects": {k: self.store.effect(rid, k) for k in ["dream", "upload", "pr"]},
            },
        )
        if not any(destination.iterdir()):
            write_json(destination / "status.json", self.store.get(rid))
        m = manifest(destination)
        total_bytes = sum(x["bytes"] for x in m["files"].values())
        if (release / "tasks").exists():
            total_bytes += sum(
                v["bytes"]
                for task in (release / "tasks").iterdir()
                for v in manifest(task)["files"].values()
            )
        if total_bytes > 100 * 1024 * 1024:
            raise GateError("checkpoint exceeds 100MB; storage decision required before push")
        self.git(repo, "add", "--sparse", "--", str(destination.relative_to(repo)))
        if self.git(repo, "status", "--porcelain").strip():
            self.git(repo, "commit", "-m", "Preserve RoboCurve production " + rid)
        result = self.push(repo)
        self.store.update(rid, checkpoint=result)
        return result

    def submit(self, rid, approval_id, fresh_novelty, credentials=None):
        root = self.store.run_dir(rid)
        approval_valid(self.store, rid, approval_id)
        if self.store.effect(rid, "pr"):
            raise GateError(
                "PR already attempted; reconcile recorded effect instead of creating another"
            )
        fresh_novelty()
        # Fresh inspection is read-only with respect to the approved bundle.
        approval_valid(self.store, rid, approval_id)
        credentials = credentials or {}
        if not credentials.get("ROBOCURVE_TOKEN") or not credentials.get("ROBOCURVE_TEAM"):
            raise GateError("team authentication missing; run rcauto auth board locally")
        release = root / "release"
        folders = list((release / "tasks").iterdir())
        if len(folders) != 1 or not folders[0].is_dir():
            raise GateError("exactly one task folder must be reviewed")
        repo = self.checkout(rid, "submit")
        target = repo / "tasks" / folders[0].name
        if self.store.effect(rid, "upload") is None:
            shutil.copytree(folders[0], target, dirs_exist_ok=True)
            mapping = read(release / "release-map.json")
            upload = repo / "tools" / "upload_media.py"
            self.store.reserve_effect(
                rid,
                "upload",
                {
                    "release_digest": read(root / "release-manifest.json")["digest"],
                    "uploader_sha256": file_hash(upload),
                },
            )
            env = dict(os.environ)
            env.update(credentials)
            args = [
                "python3",
                str(upload),
                str(target),
                "--video",
                str(release / mapping["model_video"]),
                "--demo",
                str(release / mapping["demo_video"]),
                "--screenshot",
                str(release / mapping["screenshot"]),
            ]
            p = subprocess.run(
                args, capture_output=True, text=True, env=env, timeout=660, check=False
            )
            if p.returncode:
                raise GateError("media upload failed/unknown; do not retry automatically")
            spec = read(target / "task.json")
            validate_spec(spec, require_media=True)
            self.store.finish_effect(rid, "upload", {"media": spec["media"]})
        elif self.store.effect(rid, "upload")["status"] != "finished":
            raise GateError("media upload status unknown; reconcile before continuing")
        validate_spec(read(target / "task.json"), require_media=True)
        self.git(repo, "add", "--sparse", "--", "tasks/" + folders[0].name)
        changed = self.git(repo, "diff", "--cached", "--name-only").splitlines()
        if any(not p.startswith("tasks/" + folders[0].name + "/") for p in changed):
            raise GateError("submission contains unapproved paths")
        if changed:
            self.git(repo, "commit", "-m", "Add " + read(target / "task.json")["title"])
        pushed = self.push(repo)
        body = root / "pr-body.md"
        body.write_text(
            "Add one reproducible task under `tasks/"
            + folders[0].name
            + "`. Setup, physics limitations, demo and scoring commands are in its README.\n"
        )
        owner = self.config["fork"].split("/")[0]
        self.store.reserve_effect(rid, "pr", dict(pushed, approval=approval_id))
        url = capture(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                "robocurve/benchmark-tasks",
                "--base",
                "main",
                "--head",
                owner + ":" + pushed["branch"],
                "--title",
                read(target / "task.json")["title"],
                "--body-file",
                str(body),
            ],
            timeout=120,
        ).strip()
        self.store.finish_effect(rid, "pr", {"url": url})
        self.store.update(rid, status="submitted-awaiting-receipt", pr=url)
        from .receipt import poll

        result = poll(self.store, rid, self.config.get("receipt_authors", []), timeout=180)
        try:
            self.checkpoint(rid)
        except (GateError, OSError) as e:
            self.store.update(rid, preservation_error=str(e))
            result["preservation_error"] = str(e)
        return result
