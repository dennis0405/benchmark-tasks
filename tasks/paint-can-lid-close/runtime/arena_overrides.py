"""Apply the Arena release's own dependency overrides to the Docker add-ons."""
from pathlib import Path
import tomllib

metadata = tomllib.loads(Path('/opt/IsaacLab-Arena/pyproject.toml').read_text())
overrides = metadata['tool']['uv']['override-dependencies']
overrides.append('opencv-python-headless ; sys_platform == "never"')
Path('/opt/arena-overrides.txt').write_text('\n'.join(overrides) + '\n')
