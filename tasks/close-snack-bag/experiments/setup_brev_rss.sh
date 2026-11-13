set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq python3.10-venv
mkdir -p /data/robocurve/{tasks/close-snack-bag,runs,envs}
python3 -m venv /data/robocurve/envs/bootstrap
/data/robocurve/envs/bootstrap/bin/pip -q install uv
git clone --quiet https://github.com/robosimstudio/robosimstudio.git /data/robocurve/robosimstudio
git -C /data/robocurve/robosimstudio checkout --quiet 4e52e1a76437ef1296028afea744437f6e9469c0
cd /data/robocurve/robosimstudio
UV_PROJECT_ENVIRONMENT=/data/robocurve/envs/rss UV_CACHE_DIR=/data/robocurve/cache /data/robocurve/envs/bootstrap/bin/uv sync --python 3.12 --extra superdex --no-group rtx
/data/robocurve/envs/bootstrap/bin/uv pip install --python /data/robocurve/envs/rss/bin/python 'https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl'
/data/robocurve/envs/rss/bin/python -c 'import warp as wp; wp.init(); assert wp.is_cuda_available(); print(wp.get_cuda_device())'
