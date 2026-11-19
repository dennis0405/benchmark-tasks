#!/bin/sh
set -eu
runtime_root=/tmp/yam-web-viewer
mkdir -p "$runtime_root"
cp -R /app/dist "$runtime_root/dist"
if [ -d /outputs ]; then ln -s /outputs "$runtime_root/dist/runs"; fi
asset=$(grep -Rl -m1 'signalingServer:' "$runtime_root/dist/assets" | head -1)
[ -n "$asset" ]
sed -i -E \
  -e "s/signalingServer:\"[^\"]+\"/signalingServer:\"${ISAACSIM_HOST}\"/g" \
  -e "s/signalingPort:[0-9]+/signalingPort:${ISAACSIM_SIGNAL_PORT}/g" \
  -e "s/mediaServer:\"[^\"]+\"/mediaServer:\"${ISAACSIM_HOST}\"/g" \
  -e "s/mediaPort:[0-9]+/mediaPort:${ISAACSIM_STREAM_PORT}/g" \
  "$asset"
exec /app/node_modules/.bin/vite preview "$runtime_root" --host 0.0.0.0 --port "$WEB_VIEWER_PORT"
