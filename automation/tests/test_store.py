import json
import subprocess
import sys

import pytest

from rcauto.store import Busy, StateError, Store


def test_run_reopens_without_losing_step_or_prompt(tmp_path):
    s = Store(tmp_path)
    r = s.create("데이터센터 정비", instance="test-instance")
    s.stage_done(r["id"], "research", {"artifact": "evidence.json"})
    reopened = Store(tmp_path)
    assert reopened.get(r["id"])["prompt"] == "데이터센터 정비"
    assert reopened.completed(r["id"]) == ["research"]


def test_process_death_releases_lock_but_keeps_unknown_effect(tmp_path):
    script = '''from rcauto.store import Store
import os,sys
s=Store(sys.argv[1]); r=s.create("x",instance="vm")
with s.lock("instance-vm"):
 s.reserve_effect(r["id"],"dream",{"cap_usd":2})
 print(r["id"],flush=True)
 os._exit(9)
'''
    p = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, check=False)
    assert p.returncode == 9
    rid = p.stdout.strip()
    s = Store(tmp_path)
    with s.lock("instance-vm"), pytest.raises(StateError, match="already"):
        s.reserve_effect(rid, "dream", {"cap_usd": 2})


def test_same_instance_cannot_have_two_owners(tmp_path):
    s = Store(tmp_path)
    with s.lock("vm"), pytest.raises(Busy), Store(tmp_path).lock("vm"):
        pass


def test_backup_includes_committed_wal_data(tmp_path):
    s = Store(tmp_path / "state")
    r = s.create("preserve", instance="vm")
    destination = tmp_path / "copy"
    s.backup(destination)
    assert Store(destination).get(r["id"])["prompt"] == "preserve"


def test_effect_reservation_is_cumulative_and_never_auto_retried(tmp_path):
    s = Store(tmp_path)
    a = s.create("a", instance="vm")["id"]
    b = s.create("b", instance="vm")["id"]
    s.reserve_effect(a, "dream", {"cap_usd": 15}, budget=20)
    with pytest.raises(StateError, match="budget"):
        s.reserve_effect(b, "dream", {"cap_usd": 6}, budget=20)
    s.finish_effect(a, "dream", {"actual_usd": 2})
    s.reserve_effect(b, "dream", {"cap_usd": 6}, budget=20)
    with pytest.raises(StateError, match="already"):
        s.reserve_effect(a, "dream", {"cap_usd": 1}, budget=20)


def test_journal_is_readable_json(tmp_path):
    s = Store(tmp_path)
    r = s.create("x", instance="vm")
    assert json.loads(s.events(r["id"])[0]["data"])["instance"] == "vm"
