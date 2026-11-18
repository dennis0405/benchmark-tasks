"""Optional pinned-image rebuild; existing images need no rebuilding."""
import argparse
from pathlib import Path
import subprocess
import sys
RUNTIME=Path(__file__).resolve().parent
ROOT=RUNTIME.parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--image',default='yam-inspect-isaac:6.1.0')
a=p.parse_args()
subprocess.run([sys.executable,str(RUNTIME/'prepare.py')],check=True)
subprocess.run(['docker','build','-f',str(RUNTIME/'Dockerfile'),'-t',a.image,'--build-context',f'arena={ROOT/".runtime/IsaacLab-Arena"}','--build-context',f'yamstack={ROOT/".runtime/yam-arena"}',str(ROOT)],check=True)
