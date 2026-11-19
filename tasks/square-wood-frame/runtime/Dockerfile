FROM nvcr.io/nvidia/isaac-sim:6.1.0
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 TERM=xterm

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs build-essential cmake ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Sources are prepared from the exact revisions recorded in runtime/versions.json.
COPY --from=arena /submodules/IsaacLab /opt/IsaacLab
ENV ISAACLAB_PATH=/opt/IsaacLab
RUN rm -f /opt/IsaacLab/.git && ln -s /isaac-sim /opt/IsaacLab/_isaac_sim \
    && /opt/IsaacLab/isaaclab.sh --install mimic,teleop

COPY --from=arena / /opt/IsaacLab-Arena
COPY --from=yamstack / /opt/yam-arena
COPY runtime/arena_overrides.py /tmp/arena_overrides.py
RUN /isaac-sim/python.sh /tmp/arena_overrides.py \
    && /isaac-sim/python.sh -m pip install uv \
    && /isaac-sim/python.sh -m uv pip install --python /isaac-sim/kit/python/bin/python3 \
        --override /opt/arena-overrides.txt \
        -e /opt/IsaacLab-Arena -e /opt/yam-arena \
        'inspect-robots-dream[client] @ git+https://github.com/kairosiann/inspect-robots-dream@b5e936a201cec66c54c297e522c920d98445e339' \
        'dreamlabs-client==1.0.14' \
        inspect-robots-agent

# Arena's older coverage override omits the Tracer API required by Lab's Numba.
# Lab itself specifies coverage >= 7.6.1. Keep the generated override consistent.
RUN sed -i 's/coverage==7.4.4/coverage==7.6.1/' /opt/arena-overrides.txt \
    && /isaac-sim/python.sh -m pip install --no-cache-dir --no-deps coverage==7.6.1 \
    && /isaac-sim/python.sh -c "import torch, numba; print(torch.__version__, numba.__version__)"

# The base image deduplicates these WebRTC libraries into the Teleop package.
# Updating Teleop removes those targets. Restore NvStreamBase from the base image,
# but use Torch's newer CUDA runtime so both streaming and Torch can load together.
COPY --from=nvcr.io/nvidia/isaac-sim:6.1.0 \
    /isaac-sim/exts/isaacsim.replicator.teleop/pip_prebundle/isaacteleop/cloudxr/native/libNvStreamBase.so \
    /opt/isaac-livestream-runtime/
RUN cp /isaac-sim/kit/python/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
        /opt/isaac-livestream-runtime/ \
    && for path in /isaac-sim/extscache/omni.kit.livestream.webrtc-*/bin; do \
        cp --remove-destination /opt/isaac-livestream-runtime/* "$path/"; \
    done

# Arena's published object library is under 6.0; its 6.1 URLs currently return 404.
# This selects data files only. The simulator remains Isaac Sim 6.1.0.
ENV ISAACSIM_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0

WORKDIR /task
USER isaac-sim
ENTRYPOINT ["/isaac-sim/python.sh"]
CMD ["/task/run.py", "--livestream", "0", "--viz", "none", "--episodes", "1"]
