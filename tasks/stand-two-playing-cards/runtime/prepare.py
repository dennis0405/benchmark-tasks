"""Download the pinned benchmark source trees without optional large datasets."""
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / '.runtime'
VERSIONS = {
    'isaac_sim': '6.1.0',
    'isaaclab_arena_ref': 'release/0.3.1',
    'isaaclab_arena_commit': 'c7edf3a9d5d8276565208f72422f8f8a0806d6cf',
    'isaaclab_commit': '91a0906f93802eb859192dcb0bca879f891e92d6',
    'yam_arena_commit': '0059d02a1a91bc6834c1d0846022891df0c78730',
    'dream_commit': 'b5e936a201cec66c54c297e522c920d98445e339',
}


def download(repo, commit, target):
    stamp = target / 'SOURCE_COMMIT'
    if stamp.exists() and stamp.read_text().strip() == commit:
        return
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f'{target} already exists without the expected SOURCE_COMMIT; inspect it before rebuilding.')
    print(f'Downloading {repo}@{commit}', flush=True)
    data = urlopen(f'https://codeload.github.com/{repo}/tar.gz/{commit}', timeout=120).read()
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            member.name = str(Path(*parts[1:]))
            archive.extract(member, target, filter='data')
    stamp.write_text(commit + '\n')


if __name__ == '__main__':
    RUNTIME.mkdir(exist_ok=True)
    download('isaac-sim/IsaacLab-Arena', VERSIONS['isaaclab_arena_commit'], RUNTIME / 'IsaacLab-Arena')
    download('isaac-sim/IsaacLab', VERSIONS['isaaclab_commit'], RUNTIME / 'IsaacLab-Arena/submodules/IsaacLab')
    download('kairosiann/yam-isaac-lab-arena', VERSIONS['yam_arena_commit'], RUNTIME / 'yam-arena')
    (RUNTIME / 'versions.json').write_text(json.dumps(VERSIONS, indent=2) + '\n')
