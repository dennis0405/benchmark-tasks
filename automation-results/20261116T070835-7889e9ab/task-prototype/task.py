"""Public task entry for the loaded-screen YAM prototype.

The generic ``yam_arena.task_file`` schema cannot express this compound rigid
body.  ``build`` is intentionally an explicit adapter used by
the native preflight runners; it does not claim generic TaskFile loader
compatibility.
"""

NAME = "bimanual_tab_sieve_separation_prototype"
INSTRUCTION = "Grip both screen tabs, separate the ceramic media into the receiver while retaining both steel washers, then return and release the screen."


def build():
    from full_screen_scene import build_loaded_environment
    return build_loaded_environment()
