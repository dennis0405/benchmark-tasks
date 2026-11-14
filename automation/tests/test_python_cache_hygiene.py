import py_compile
import sys

import pytest

from rcauto.engine import Production
from rcauto.gates import GateError, discard_generated_bytecode
from rcauto.store import Store


def test_native_script_can_compile_owned_source_without_changing_frozen_sources(tmp_path):
    store = Store(tmp_path / "state")
    run = store.create("cache hygiene", instance="test", mode="selftest")
    p = Production(store, run["id"], None, {"runtime": {"python": sys.executable}})
    script = p.workspace / "task/probe.py"
    script.write_text(
        "import json,py_compile\nfrom pathlib import Path\n"
        "py_compile.compile(__file__,doraise=True)\n"
        "out=Path(__file__).resolve().parent.parent/'measurements/probe.json'\n"
        "out.write_text(json.dumps({'checks':[{'name':'executed','measured':1,"
        "'operator':'==','threshold':1,'unit':'count'}]}))\n"
    )
    original = script.read_bytes()
    py_compile.compile(str(script), doraise=True)
    receipts = p.execute_scripts(
        {"scripts": [{"id": "probe", "script": "probe.py", "args": [],
                      "report": "measurements/probe.json"}]},
        [{"name": "executed", "operator": "==", "threshold": 1, "unit": "count"}],
    )
    assert receipts[0]["exit_code"] == 0
    assert script.read_bytes() == original
    assert not list((p.workspace / "task").rglob("__pycache__"))


def test_cache_cleanup_does_not_delete_unrecognized_files(tmp_path):
    (tmp_path / "module.py").write_text("x=1\n")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    unknown = cache / "important.txt"
    unknown.write_text("keep")
    with pytest.raises(GateError):
        discard_generated_bytecode(tmp_path)
    assert unknown.read_text() == "keep"


def test_cache_symlink_cannot_delete_external_files(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "module.cpython-312.pyc").write_bytes(b"keep")
    (root / "__pycache__").symlink_to(outside, target_is_directory=True)
    with pytest.raises(GateError):
        discard_generated_bytecode(root)
    assert (outside / "module.cpython-312.pyc").read_bytes() == b"keep"
