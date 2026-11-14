"""Public YAM geometry for the ODRI HELICOIL tang-finishing task.

The public interface can only express primitive scene objects.  It therefore
shows the same open shell, hollow bosses/inserts, separate mandrel shaft and
handle, and separate hammer head and handle used by component_mechanism.py.
Native joints/contact accounting live in that shared task-local layer; this
file never turns a waypoint or object pose into success.
"""

import importlib.util
from pathlib import Path

from yam_arena.task_file import Box, Cylinder, TABLE_TOP

# TaskFile.load() deliberately does not add this directory to sys.path.  Load
# the task-local, pure-Python geometry by absolute file location so this file
# works from a clean cwd without a private PYTHONPATH.
_geometry_path = Path(__file__).resolve().with_name("procedural_geometry.py")
_geometry_spec = importlib.util.spec_from_file_location(
    f"{__name__}_procedural_geometry", _geometry_path
)
if _geometry_spec is None or _geometry_spec.loader is None:
    raise ImportError(f"cannot load task geometry: {_geometry_path}")
_geometry = importlib.util.module_from_spec(_geometry_spec)
_geometry_spec.loader.exec_module(_geometry)
INSERT_XY = _geometry.INSERT_XY

_scorer_path = Path(__file__).resolve().with_name("outcome_scorer.py")
_scorer_spec = importlib.util.spec_from_file_location(
    f"{__name__}_outcome_scorer", _scorer_path
)
if _scorer_spec is None or _scorer_spec.loader is None:
    raise ImportError(f"cannot load task scorer: {_scorer_path}")
_scorer = importlib.util.module_from_spec(_scorer_spec)
_scorer_spec.loader.exec_module(_scorer)

# Populated only by the post-App native adapter after its ordered POST update.
# A plain public TaskFile load and an unadapted YAM scene therefore score zero.
_native_snapshot = None


def publish_native_snapshot(snapshot):
    """Adapter entry point; publishing cannot create or alter physical state."""
    global _native_snapshot
    _native_snapshot = snapshot

SCENE = "station"
Z = TABLE_TOP[SCENE]
SUCCESS_HOLD_S = 2.0
SCORE_OVER = "end"
OBJECTS = []

# Open shell: floor plus four walls, not a solid mandrel-obstructing block.
OBJECTS += [
    Box("shell_floor", (.340, 0, Z + .003), (.20, .42, .27), size=(.180, .090, .006)),
    Box("shell_wall_l", (.340, .044, Z + .012), (.20, .42, .27), size=(.180, .003, .024)),
    Box("shell_wall_r", (.340, -.044, Z + .012), (.20, .42, .27), size=(.180, .003, .024)),
    Box("shell_wall_f", (.429, 0, Z + .012), (.20, .42, .27), size=(.003, .084, .024)),
    Box("shell_wall_b", (.251, 0, Z + .012), (.20, .42, .27), size=(.003, .084, .024)),
]

for i, (x, y) in enumerate(INSERT_XY):
    # Four bars leave a genuine central bore in each boss/insert.
    for suffix, dx, dy, sx, sy in (("n",0,.006,.014,.002),("s",0,-.006,.014,.002),
                                    ("e",.006,0,.002,.010),("w",-.006,0,.002,.010)):
        OBJECTS.append(Box(f"boss_{i}_{suffix}",(x+dx,y+dy,Z+.015),(.30,.48,.34),size=(sx,sy,.018)))
    for suffix, dx, dy, sx, sy in (("n",0,.0021,.0048,.0006),("s",0,-.0021,.0048,.0006),
                                    ("e",.0021,0,.0006,.0036),("w",-.0021,0,.0006,.0036)):
        OBJECTS.append(Box(f"insert_{i}_{suffix}",(x+dx,y+dy,Z+.025),(.18,.18,.20),size=(sx,sy,.00375)))
    OBJECTS.append(Box(f"tang_{i}",(x,y,Z+.0272),(.78,.58,.18),mass=.000035,size=(.0032,.00065,.00045)))

OBJECTS += [
    Cylinder("mandrel_shaft", (.255,-.135,Z+.036), (.22,.38,.78), mass=.025, radius=.001, height=.070),
    Cylinder("mandrel_handle", (.255,-.135,Z+.091), (.12,.27,.70), mass=.030, radius=.007, height=.040),
    Box("hammer_head", (.350,.145,Z+.024), (.82,.32,.08), mass=.105, size=(.030,.018,.016)),
    Cylinder("hammer_handle", (.350,.145,Z+.083), (.82,.32,.08), mass=.075, radius=.006, height=.105),
    Box("tray_floor", (.475,-.112,Z+.003), (.18,.23,.30), size=(.080,.070,.006)),
    Box("tray_l", (.475,-.078,Z+.020), (.18,.23,.30), size=(.080,.006,.040)),
    Box("tray_r", (.475,-.146,Z+.020), (.18,.23,.30), size=(.080,.006,.040)),
    Box("tray_f", (.514,-.112,Z+.020), (.18,.23,.30), size=(.006,.064,.040)),
    Box("tray_b", (.436,-.112,Z+.020), (.18,.23,.30), size=(.006,.064,.040)),
]

def score(objects):
    del objects
    return _scorer.score_snapshot(_native_snapshot)

def success(objects):
    del objects
    return _scorer.success_snapshot(_native_snapshot)

def demo():
    return {}
