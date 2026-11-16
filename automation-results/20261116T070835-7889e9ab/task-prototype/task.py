"""Public task entry for the full-screen YAM component preflight.

The generic ``yam_arena.task_file`` schema cannot express this compound rigid
body.  ``build`` is intentionally an explicit adapter used by
``run_full_screen_yam_preflight.py``; it does not claim generic TaskFile loader
compatibility.
"""

NAME = "full_screen_bimanual_lift_preflight"
INSTRUCTION = "Grip both screen tabs, lift the empty screen 30 mm, hold it for 3 seconds, then return it to the supports."


def build():
    from full_screen_scene import build_environment
    return build_environment()
