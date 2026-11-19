"""Portable entrypoint exported by the Robocurve workflow skill."""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime'))
sys.argv[1:1] = ['--task', str(ROOT / 'task.py')]
runpy.run_path(str(ROOT / 'runtime' / 'run_task.py'), run_name='__main__')
