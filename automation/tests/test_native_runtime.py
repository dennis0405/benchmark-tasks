import json
import sys
import types

import pytest

from rcauto.gates import GateError, file_hash
from rcauto.native_runtime import prepare_native_runtime, register_native_runtime


@pytest.fixture
def base_sdk(monkeypatch):
    class BaseYam:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def reset(self):
            pass

        def step(self):
            pass

        def _observation(self):
            pass

        def _task_step(self):
            pass

        def _ensure_sim(self):
            pass

    parent = types.ModuleType("yam_arena")
    parent.__path__ = []
    sdk = types.ModuleType("yam_arena.inspect_embodiment")
    sdk.YamArenaEmbodiment = BaseYam
    registry = types.ModuleType("inspect_robots.registry")
    registered = {}

    def embodiment(name):
        def decorate(cls):
            registered[name] = cls
            return cls
        return decorate

    registry.embodiment = embodiment
    inspect = types.ModuleType("inspect_robots")
    inspect.__path__ = []
    for name, module in [("yam_arena", parent), (sdk.__name__, sdk),
                         ("inspect_robots", inspect), (registry.__name__, registry)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    return BaseYam, registered


def native_task(tmp_path, extra=""):
    task = tmp_path / "task.py"
    task.write_text("# unchanged public task declarations\n")
    module = tmp_path / "native_runtime.py"
    module.write_text(
        "from yam_arena.inspect_embodiment import YamArenaEmbodiment\n"
        "class NativeYam(YamArenaEmbodiment):\n"
        "    def _ensure_sim(self, goal=None):\n        pass\n" + extra
    )
    descriptor = {"module": module.name, "class_name": "NativeYam", "sha256": file_hash(module)}
    (tmp_path / "native-runtime.json").write_text(json.dumps(descriptor))
    return task, descriptor


def test_primitive_tasks_keep_original_embodiment_without_loading_plugins(tmp_path):
    task = tmp_path / "task.py"
    task.write_text("# primitive scene")
    assert prepare_native_runtime(task) is None
    assert register_native_runtime(task, None) == "yam_arena"


def test_native_extension_keeps_all_standard_yam_interfaces(tmp_path, base_sdk):
    task, descriptor = native_task(tmp_path)
    assert prepare_native_runtime(task) == descriptor
    name = register_native_runtime(task, descriptor)
    base, registered = base_sdk
    cls = registered[name]
    for method in ("__init__", "reset", "step", "_observation", "_task_step"):
        assert getattr(cls, method) is getattr(base, method)
    assert cls._ensure_sim is not base._ensure_sim
    assert cls(task=str(task), control_hz=20).kwargs["control_hz"] == 20
    assert not (tmp_path / "__pycache__").exists()


@pytest.mark.parametrize("method", ["__init__", "reset", "step", "_observation", "_task_step"])
def test_native_extension_cannot_override_control_or_scoring_protocol(tmp_path, base_sdk, method):
    task, descriptor = native_task(tmp_path, f"    def {method}(self, *args, **kwargs):\n        pass\n")
    with pytest.raises(GateError, match="preserve"):
        register_native_runtime(task, descriptor)
    assert not base_sdk[1]


def test_source_change_blocks_before_module_execution(tmp_path):
    task, _ = native_task(tmp_path)
    (tmp_path / "native_runtime.py").write_text('raise RuntimeError("must not execute")')
    with pytest.raises(GateError, match="digest"):
        prepare_native_runtime(task)


def test_validated_job_cannot_silently_select_another_adapter(tmp_path):
    task, _ = native_task(tmp_path)
    with pytest.raises(GateError, match="validated job"):
        register_native_runtime(task, None)


def test_runtime_module_cannot_escape_task_directory(tmp_path):
    task, descriptor = native_task(tmp_path)
    descriptor["module"] = "../outside.py"
    (tmp_path / "native-runtime.json").write_text(json.dumps(descriptor))
    with pytest.raises(GateError, match="escapes"):
        prepare_native_runtime(task)
