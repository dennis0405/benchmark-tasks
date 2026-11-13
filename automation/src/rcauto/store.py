"""Durable local journal. Unknown external effects are deliberately never reset."""

import contextlib
import fcntl
import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class StateError(RuntimeError):
    pass


class Busy(StateError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "runs.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, run TEXT, at TEXT, data TEXT);
          CREATE TABLE IF NOT EXISTS stages(run TEXT, stage TEXT, data TEXT, PRIMARY KEY(run,stage));
          CREATE TABLE IF NOT EXISTS effects(run TEXT, kind TEXT, status TEXT, data TEXT,
                                             PRIMARY KEY(run,kind));
        """)

    def create(self, prompt, instance, **extra):
        if not prompt.strip():
            raise StateError("prompt must not be empty")
        rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        value = dict(
            id=rid,
            prompt=prompt,
            instance=instance,
            created_at=now(),
            status="created",
            stage=None,
            **extra,
        )
        with self.db:
            self.db.execute("INSERT INTO runs VALUES(?,?)", (rid, json.dumps(value)))
            self._event(rid, {"event": "created", "instance": instance})
        self.run_dir(rid).mkdir(parents=True)
        return value

    def register(self, value):
        """Import a controller-issued run on its bound worker, without inventing a second id."""
        self.run_dir(value["id"])
        existing = self.db.execute("SELECT data FROM runs WHERE id=?", (value["id"],)).fetchone()
        if existing:
            if json.loads(existing[0])["prompt"] != value["prompt"]:
                raise StateError("worker run identity changed")
            return self.update(
                value["id"],
                revision=value.get("revision", 0),
                revision_request=value.get("revision_request"),
            )
        with self.db:
            self.db.execute("INSERT INTO runs VALUES(?,?)", (value["id"], json.dumps(value)))
            self._event(value["id"], {"event": "registered"})
        return value

    def invalidate(self, rid, stages):
        with self.db:
            for stage in stages:
                self.db.execute("DELETE FROM stages WHERE run=? AND stage=?", (rid, stage))
            value = self.get(rid)
            value.pop("approval", None)
            self.db.execute("UPDATE runs SET data=? WHERE id=?", (json.dumps(value), rid))
            self._event(rid, {"invalidated": list(stages)})

    def run_dir(self, rid):
        if not rid or any(
            c not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_" for c in rid
        ):
            raise StateError("invalid run id")
        return self.root / "runs" / rid

    def get(self, rid):
        row = self.db.execute("SELECT data FROM runs WHERE id=?", (rid,)).fetchone()
        if row is None:
            raise StateError(f"unknown run: {rid}")
        return json.loads(row[0])

    def list(self):
        return [json.loads(r[0]) for r in self.db.execute("SELECT data FROM runs ORDER BY id DESC")]

    def _event(self, rid, data):
        self.db.execute(
            "INSERT INTO events(run,at,data) VALUES(?,?,?)", (rid, now(), json.dumps(data))
        )

    def update(self, rid, **changes):
        with self.db:
            value = self.get(rid)
            value.update(changes, updated_at=now())
            self.db.execute("UPDATE runs SET data=? WHERE id=?", (json.dumps(value), rid))
            self._event(rid, changes)
        return value

    def events(self, rid):
        return [
            dict(r)
            for r in self.db.execute("SELECT * FROM events WHERE run=? ORDER BY seq", (rid,))
        ]

    def stage_done(self, rid, stage, receipt):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO stages VALUES(?,?,?)", (rid, stage, json.dumps(receipt))
            )
            self._event(rid, {"stage_done": stage, "receipt": receipt})

    def completed(self, rid):
        return [
            r[0]
            for r in self.db.execute("SELECT stage FROM stages WHERE run=? ORDER BY rowid", (rid,))
        ]

    def effect(self, rid, kind):
        row = self.db.execute(
            "SELECT * FROM effects WHERE run=? AND kind=?", (rid, kind)
        ).fetchone()
        return dict(row) if row else None

    def reserve_effect(self, rid, kind, data, budget=None):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.effect(rid, kind):
                raise StateError(
                    f"{kind} already reserved/executed; reconcile, never retry automatically"
                )
            if kind == "dream" and data.get("input_digest"):
                for row in self.db.execute("SELECT data FROM effects WHERE kind='dream'"):
                    if json.loads(row[0]).get("input_digest") == data["input_digest"]:
                        raise StateError(
                            "same Dream inputs already attempted under another run name"
                        )
            if budget is not None:
                if (
                    type(budget) not in (int, float)
                    or not math.isfinite(budget)
                    or budget <= 0
                    or type(data.get("cap_usd")) not in (int, float)
                    or not math.isfinite(data["cap_usd"])
                ):
                    raise StateError("budget must be finite positive currency")
                reserved = 0
                for row in self.db.execute("SELECT status,data FROM effects WHERE kind=?", (kind,)):
                    entry = json.loads(row[1])
                    reserved += entry.get("actual_usd", entry["cap_usd"])
                if data["cap_usd"] <= 0 or reserved + data["cap_usd"] > budget:
                    raise StateError("cumulative budget exceeded")
            self.db.execute(
                "INSERT INTO effects VALUES(?,?,?,?)", (rid, kind, "unknown", json.dumps(data))
            )
            self._event(rid, {"reserved_effect": kind, "data": data})
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def finish_effect(self, rid, kind, result):
        if "actual_usd" in result and (
            type(result["actual_usd"]) not in (int, float)
            or not math.isfinite(result["actual_usd"])
            or result["actual_usd"] < 0
        ):
            raise StateError("actual billing must be finite and nonnegative")
        with self.db:
            row = self.effect(rid, kind)
            if not row:
                raise StateError("effect has not been reserved")
            data = json.loads(row["data"])
            data.update(result)
            self.db.execute(
                "UPDATE effects SET status='finished',data=? WHERE run=? AND kind=?",
                (json.dumps(data), rid, kind),
            )
            self._event(rid, {"finished_effect": kind, "result": result})

    @contextlib.contextmanager
    def lock(self, key):
        import hashlib

        p = self.root / ("lock-" + hashlib.sha256(key.encode()).hexdigest())
        with p.open("a+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise Busy(f"already owned: {key}") from e
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def backup(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(destination / "runs.sqlite3") as other:
            self.db.backup(other)
