"""Explicit task-local scene extension that preserves the pinned YAM protocol."""

import importlib.util
import json
import re
import sys
from pathlib import Path

from .gates import GateError, file_hash, inside

NAME = "rcauto_yam_native"
PRESERVED_METHODS = ("__init__", "reset", "step", "_observation", "_task_step")


def prepare_native_runtime(task):
    """Validate the descriptor before any Dream reservation or module execution."""
    task = Path(task).resolve()
    descriptor = task.parent / "native-runtime.json"
    if not descriptor.exists():
        return None
    value = json.loads(descriptor.read_text())
    if not isinstance(value, dict) or set(value) != {"module", "class_name", "sha256"}:
        raise GateError("invalid native runtime descriptor fields")
    relative = value["module"]
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise GateError("native runtime module must be task-relative")
    module = inside(task.parent, relative)
    if module.suffix != ".py" or not module.is_file():
        raise GateError("native runtime module must be an existing Python file")
    if not isinstance(value["class_name"], str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_]*", value["class_name"]
    ):
        raise GateError("invalid native runtime class name")
    if file_hash(module) != value["sha256"]:
        raise GateError("native runtime source digest mismatch")
    return value


def register_native_runtime(task, expected):
    """Register an optional scene builder via Inspect Robots' public extension API.

    This enforces interface inheritance, not physical qualification or sandboxing.
    Source review, native physics, demos and portability gates still apply.
    """
    task = Path(task).resolve()
    actual = prepare_native_runtime(task)
    if actual != expected:
        raise GateError("native runtime differs from validated job")
    if actual is None:
        return "yam_arena"
    from inspect_robots.registry import embodiment
    from yam_arena.inspect_embodiment import YamArenaEmbodiment

    path = inside(task.parent, actual["module"])
    name = "_rcauto_native_yam_" + actual["sha256"][:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GateError("cannot load native runtime source")
    # Local helper modules must remain importable when scene construction starts
    # later in reset(). Do not generate bytecode inside the frozen release.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(task.parent))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    cls = getattr(module, actual["class_name"], None)
    if not isinstance(cls, type) or not issubclass(cls, YamArenaEmbodiment):
        raise GateError("native runtime must extend the pinned YamArenaEmbodiment")
    for method in PRESERVED_METHODS:
        if getattr(cls, method) is not getattr(YamArenaEmbodiment, method):
            raise GateError("native runtime must preserve YAM " + method)
    if cls._ensure_sim is YamArenaEmbodiment._ensure_sim:
        raise GateError("native runtime has no task-local scene constructor")
    embodiment(NAME)(cls)
    return NAME
