"""Real runtime diagnostics against upstream examples; never counts as a produced task."""

import json
import shlex
from pathlib import Path

from .store import now


def validate_native_baseline(brev, config):
    root = config["remote_root"] + "/pipeline-checks/baseline-" + now().replace(":", "-")
    brev.ssh("mkdir -p " + shlex.quote(root))
    executable = str(Path(config["runtime"]["python"]).parent / "inspect-robots")
    task = config["runtime"]["yam"] + "/examples/stack_cubes.py"
    argv = [
        executable,
        "run",
        "--embodiment",
        "yam_arena",
        "--policy",
        "yam_scripted",
        "-E",
        "task=" + task,
        "-E",
        "object_jitter=0",
        "-E",
        "control_hz=20",
        "-E",
        "cam_width=320",
        "-E",
        "cam_height=176",
        "--instruction",
        "put the red cube on the blue block",
        "--scorer",
        "task_score",
        "--max-steps",
        "600",
        "--epochs",
        "1",
        "--grader",
        "none",
        "--no-prompt",
        "--store-frames",
        "--log-dir",
        root,
    ]
    brev.ssh(
        "ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y "
        + shlex.join(argv)
        + " >"
        + shlex.quote(root + "/run.log")
        + " 2>&1",
        timeout=900,
    )
    code = (
        "import json,pathlib;root=pathlib.Path(" + repr(root) + ");"
        "p=next(root.glob('adhoc_*.json'));d=json.loads(p.read_text());"
        "assert d['status']=='success';assert d['results']['total_trials']==1;"
        "assert d['results']['metrics']['task_score']==1.0;"
        "print(json.dumps({'status':d['status'],'score':d['results']['metrics']['task_score'],"
        "'policy':d['eval']['policy'],'dream_calls':0,'log':str(p)}))"
    )
    return json.loads(brev.ssh("python3 -c " + shlex.quote(code)))
