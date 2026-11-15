#!/usr/bin/env bash
# Restore the previously tested native runtime on a NEW dedicated x86_64 Brev VM.
# No credentials, evaluation calls, SDK patches, or instance lifecycle operations.
set -euo pipefail

BASE=/data/robocurve-automation
ARENA_REF=c7edf3a9d5d8276565208f72422f8f8a0806d6cf
LAB_REF=91a0906f93802eb859192dcb0bca879f891e92d6
YAM_REF=0059d02a1a91bc6834c1d0846022891df0c78730
DREAM_REF=b5e936a201cec66c54c297e522c920d98445e339
[[ $(uname -m) == x86_64 ]] || { echo 'x86_64 required'; exit 1; }
mountpoint -q /data || { echo '/data is not a separate mounted data volume'; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 update -qq
sudo env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y \
  --no-install-recommends git curl ca-certificates python3-venv python3-pip ffmpeg \
  jq cmake build-essential pkg-config libegl1 libgl1 libvulkan1 libx11-6 libxext6 \
  libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 libglib2.0-0 libsm6 libice6 \
  poppler-utils ripgrep
mkdir -p "$BASE"/{sources,envs,cache,tmp,checkpoints} "$HOME/.local/bin"
python3 -m venv "$BASE/envs/bootstrap"
"$BASE/envs/bootstrap/bin/pip" install uv==0.13.8
ln -sfn "$BASE/envs/bootstrap/bin/uv" "$HOME/.local/bin/uv"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$BASE/cache/uv"
export UV_PYTHON_INSTALL_DIR="$BASE/envs/python"
export TMPDIR="$BASE/tmp"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

checkout() {
  local repo=$1 dir=$2 ref=$3 sparse=${4:-}
  if [[ -e "$dir/.git" ]]; then
    [[ $(git -C "$dir" rev-parse HEAD) == "$ref" ]] || {
      echo "Refusing to replace a different checkout: $dir"; exit 1;
    }
    return
  fi
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$repo"
  git -C "$dir" fetch --filter=blob:none --depth=1 origin "$ref"
  if [[ "$sparse" == lab ]]; then
    git -C "$dir" sparse-checkout init --cone
    git -C "$dir" sparse-checkout set source scripts apps
  fi
  git -C "$dir" checkout --detach FETCH_HEAD
  [[ $(git -C "$dir" rev-parse HEAD) == "$ref" ]]
}

ARENA="$BASE/sources/IsaacLab-Arena"
LAB="$ARENA/submodules/IsaacLab"
YAM="$BASE/sources/yam-isaac-lab-arena"
checkout https://github.com/isaac-sim/IsaacLab-Arena.git "$ARENA" "$ARENA_REF"
# Initialize only the required source checkout. The optional GR00T client is
# absent from the pinned default dependency groups and is not downloaded.
git -C "$ARENA" config submodule.submodules/Isaac-GR00T.update none
checkout https://github.com/isaac-sim/IsaacLab.git "$LAB" "$LAB_REF" lab
checkout https://github.com/kairosiann/yam-isaac-lab-arena.git "$YAM" "$YAM_REF"

cd "$ARENA"
uv sync --frozen --inexact
uv pip install --python "$ARENA/.venv/bin/python" --override "$YAM/overrides.txt" \
  -e "$YAM" inspect-robots==1.0.0 inspect-robots-yam==0.38.0 \
  inspect-robots-agent==0.30.0 dreamlabs-client==1.1.3 \
  "inspect-robots-dream[client] @ git+https://github.com/kairosiann/inspect-robots-dream@$DREAM_REF"
printf 'source %q\nexport OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y\n' \
  "$ARENA/.venv/bin/activate" > "$ARENA/yam.env"
"$ARENA/.venv/bin/python" - <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
names = ['isaacsim','isaaclab','isaaclab-arena','yam-arena','inspect-robots',
         'inspect-robots-yam','inspect-robots-dream','dreamlabs-client','torch']
versions = {name: m.version(name) for name in names}
for name, expected in {'isaacsim':'6.1.0.0','isaaclab':'17.0.2',
                       'inspect-robots':'1.0.0','inspect-robots-yam':'0.38.0',
                       'dreamlabs-client':'1.1.3'}.items():
    assert versions[name] == expected, (name,versions[name],expected)
Path('/data/robocurve-automation/checkpoints/restored-package-versions.json').write_text(
    json.dumps(versions,indent=2))
print(json.dumps(versions))
PY
echo 'Pinned runtime installed. Native smoke, authentication and task tests remain.'
