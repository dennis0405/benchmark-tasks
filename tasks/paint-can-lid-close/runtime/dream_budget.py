"""Persistent local spending guard. Amounts are conservative estimates, not account billing."""
import argparse
import json
import math
import os
from pathlib import Path
import sqlite3
import time

DEFAULTS = {"call_micro_usd": 1500, "task_micro_usd": 30000,
            "day_micro_usd": 600000, "total_micro_usd": 12000000}
DEFAULT_PATH = Path(os.environ.get('DREAM_BUDGET_PATH', '/dream-budget/ledger.sqlite3'))


class BudgetStopped(RuntimeError):
    pass


class Ledger:
    def __init__(self, path=DEFAULT_PATH, clock=time.time):
        self.path = Path(path)
        self.clock = clock

    @classmethod
    def initialize(cls, path=DEFAULT_PATH, *, limits=None, clock=time.time):
        """Explicit initialization only; never reset an existing ledger on startup."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        config = dict(DEFAULTS if limits is None else limits)
        config.update(version=1, expires_at=clock() + 20 * 86400)
        with sqlite3.connect(path) as db:
            db.executescript("""
                CREATE TABLE config (body TEXT NOT NULL);
                CREATE TABLE calls (id INTEGER PRIMARY KEY, ts REAL NOT NULL,
                    task TEXT NOT NULL, run TEXT NOT NULL, amount INTEGER NOT NULL,
                    outcome TEXT NOT NULL DEFAULT 'reserved');
                CREATE TABLE stops (ts REAL NOT NULL, task TEXT, run TEXT, reason TEXT NOT NULL);
            """)
            db.execute("INSERT INTO config VALUES (?)", (json.dumps(config),))
        return cls(path, clock)

    def _connect(self):
        # mode=rw is essential: a deleted/missing ledger must not become an empty budget.
        db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _config(self, db):
        records = db.execute("SELECT body FROM config").fetchall()
        if len(records) != 1:
            raise BudgetStopped("budget_config_invalid")
        cfg = json.loads(records[0][0])
        if cfg.get("version") != 1 or cfg.get("call_micro_usd") != DEFAULTS["call_micro_usd"]:
            raise BudgetStopped("budget_config_invalid")
        for name in ("task_micro_usd", "day_micro_usd", "total_micro_usd"):
            if type(cfg.get(name)) is not int or not 0 < cfg[name] <= DEFAULTS[name]:
                raise BudgetStopped("budget_config_invalid")
        if not isinstance(cfg.get("expires_at"), (int, float)) or not math.isfinite(cfg["expires_at"]):
            raise BudgetStopped("budget_config_invalid")
        return cfg

    @staticmethod
    def _usage(db, task, run, now):
        def amount(where="1", values=()):
            return db.execute(f"SELECT COALESCE(SUM(amount),0) FROM calls WHERE {where}", values).fetchone()[0]
        return {"total_micro_usd": amount(),
                "day_micro_usd": amount("ts > ?", (now-86400,)),
                "task_micro_usd": amount("task = ?", (task,)),
                "run_micro_usd": amount("run = ?", (run,)),
                "run_calls": db.execute("SELECT COUNT(*) FROM calls WHERE run = ?", (run,)).fetchone()[0]}

    def reserve(self, task, run):
        """Commit the full call estimate before any bytes are sent. Never refund errors."""
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                cfg = self._config(db)
                now = self.clock()
                previous = db.execute("SELECT MAX(ts) FROM calls").fetchone()[0]
                usage = self._usage(db, task, run, now)
                reason = None
                if now >= cfg["expires_at"]:
                    reason = "budget_period_expired"
                elif previous is not None and now < previous:
                    reason = "clock_moved_backwards"
                else:
                    for name in ("task_micro_usd", "day_micro_usd", "total_micro_usd"):
                        if usage[name] + cfg["call_micro_usd"] > cfg[name]:
                            reason = name.replace("_micro_usd", "") + "_budget_limit"
                            break
                if reason:
                    db.execute("INSERT INTO stops VALUES (?,?,?,?)", (now, task, run, reason))
                    db.commit()
                    raise BudgetStopped(reason)
                row = db.execute("INSERT INTO calls (ts,task,run,amount) VALUES (?,?,?,?)",
                                 (now, task, run, cfg["call_micro_usd"]))
                return row.lastrowid
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
            raise BudgetStopped("budget_ledger_unavailable_or_invalid") from None

    def finish(self, reservation, outcome):
        if outcome not in ("response", "network_error", "model_error"):
            raise ValueError("Unknown outcome")
        try:
            with self._connect() as db:
                db.execute("UPDATE calls SET outcome=? WHERE id=?", (outcome, reservation))
        except sqlite3.Error:
            raise BudgetStopped("budget_ledger_write_failed") from None

    def status(self, task="", run=""):
        try:
            with self._connect() as db:
                cfg = self._config(db)
                usage = self._usage(db, task, run, self.clock())
                last = db.execute("SELECT ts,task,run,reason FROM stops ORDER BY rowid DESC LIMIT 1").fetchone()
                return {"limits": cfg, "reserved": usage, "last_stop": last,
                        "published_rate_estimate_usd": usage["run_calls"] * 3 / 2250,
                        "scope": "Only requests through this runner; not the Dream account balance."}
        except (sqlite3.Error, OSError, ValueError, KeyError, TypeError):
            raise BudgetStopped("budget_ledger_unavailable_or_invalid") from None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "status"))
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--task", default="")
    parser.add_argument("--run", default="")
    args = parser.parse_args()
    ledger = Ledger.initialize(args.path) if args.action == "init" else Ledger(args.path)
    print(json.dumps(ledger.status(args.task, args.run), indent=2))
