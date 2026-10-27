#!/bin/sh
set -eu
D=/srv/workspace/open-webui
IMG=ghcr.io/open-webui/open-webui@sha256:1565038903151ced924b5a3226f73f40f16963df4b910441f6391f10144afdaa
set -- -v "$D/data:/app/backend/data" -v "$D/branding/favicon.png:/app/build/favicon.png:ro"
for f in favicon.png favicon.svg favicon.ico favicon-96x96.png apple-touch-icon.png logo.png splash.png splash-dark.png web-app-manifest-192x192.png web-app-manifest-512x512.png site.webmanifest; do
  set -- "$@" -v "$D/branding/$f:/app/build/static/$f:ro"
done
set -- "$@" -v "$D/branding/loader.js:/app/build/static/loader.js:ro"
docker rm -f sparkpipe-chatui >/dev/null 2>&1 || true
docker run -d --name sparkpipe-chatui --restart unless-stopped -p 127.0.0.1:8090:8080 \
  --log-opt max-size=10m --log-opt max-file=3 "$@" \
  --env-file "$D/open_webui.env" --env-file "$D/secrets.env" "$IMG"
