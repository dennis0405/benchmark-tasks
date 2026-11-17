"""Representative DRG 80 opening-stop workcell on the YAM station.

The native workcell must publish one measured sample after every physics substep.
These public TaskFile declarations provide the reproducible room layout and fallback
visuals; scoring is intentionally inert until the native read-only observation bridge
has supplied fresh post-physics samples.
"""

from pathlib import Path
from runpy import run_path

from yam_arena.task_file import Box, Cylinder

SCENE = "station"
SUCCESS_HOLD_S = 1.0
SCORE_OVER = "end"
TABLE_Z = 0.7475

OBJECTS = [
    Box("fixture_base", pos=(0.36, 0.0, TABLE_Z + 0.005), size=(0.100, 0.0716, 0.010), color=(0.18, 0.22, 0.26)),
    Box("fixture_left", pos=(0.318, 0.0, TABLE_Z + 0.045), size=(0.016, 0.0716, 0.070), color=(0.18, 0.22, 0.26)),
    Box("fixture_right", pos=(0.402, 0.0, TABLE_Z + 0.045), size=(0.016, 0.0716, 0.070), color=(0.18, 0.22, 0.26)),
    Box("left_jaw", pos=(0.330, 0.022, TABLE_Z + 0.120), size=(0.018, 0.024, 0.062), color=(0.35, 0.37, 0.40)),
    Box("right_jaw", pos=(0.390, 0.022, TABLE_Z + 0.120), size=(0.018, 0.024, 0.062), color=(0.35, 0.37, 0.40)),
    Cylinder("adjustment_screw", pos=(0.36, -0.018, TABLE_Z + 0.105), radius=0.0055, height=0.032, color=(0.40, 0.42, 0.45)),
    Cylinder("nut73", pos=(0.36, -0.018, TABLE_Z + 0.095), radius=0.009, height=0.004, color=(0.55, 0.56, 0.58)),
    Cylinder("nut74", pos=(0.36, -0.018, TABLE_Z + 0.102), radius=0.009, height=0.004, color=(0.55, 0.56, 0.58)),
    Box("driver", pos=(0.24, 0.17, TABLE_Z + 0.027), size=(0.026, 0.145, 0.026), mass=0.18, color=(0.10, 0.16, 0.65)),
    Box("wrench", pos=(0.17, -0.17, TABLE_Z + 0.008), size=(0.160, 0.028, 0.010), mass=0.14, color=(0.45, 0.47, 0.50)),
    Box("driver_rack", pos=(0.24, 0.17, TABLE_Z + 0.006), size=(0.060, 0.160, 0.012), color=(0.12, 0.12, 0.12)),
    Box("wrench_rack", pos=(0.18, -0.17, TABLE_Z + 0.006), size=(0.180, 0.060, 0.012), color=(0.12, 0.12, 0.12)),
]

# The common YAM loader imports task.py by filename, including from outside this directory.
_grading = run_path(str(Path(__file__).with_name("stop_lock_score.py")))
_grader = None


def reset_native_observations(physics_hz):
    """The scene calls this after each physical reset, before the first sample."""
    global _grader
    _grader = _grading["StopLockScore"](physics_hz)


def publish_native_observation(observation):
    """Consume only measured fields, in native substep order."""
    if _grader is None:
        raise RuntimeError("reset_native_observations must precede the first sample")
    try:
        fields = _grading["Sample"].__dataclass_fields__
        sample = _grading["Sample"](**{name: observation[name] for name in fields})
        return _grader.observe(sample, observation["physics_tick"])
    except (KeyError, ValueError, TypeError, OverflowError):
        _grader.damaged = True
        _grader.score = 0.0
        raise


def score(_objects):
    return _grader.score if _grader is not None else 0.0


def success(objects):
    return score(objects) == 1.0
