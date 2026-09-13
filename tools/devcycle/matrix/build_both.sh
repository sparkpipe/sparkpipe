#!/usr/bin/env bash
set -uo pipefail
cd /tmp/sparkpipe-devcycle
bash tools/devcycle/build_remote.sh matrix-b4 ef8fa302ad8f545ee8bfad20c63329d79b76f72c 4 > /tmp/matrix-b4-build.log 2>&1
rc4=$?
echo "b4 rc=$rc4" >> /tmp/matrix-builds.status
if [ $rc4 -eq 0 ]; then
  bash tools/devcycle/build_remote.sh matrix-b16 ef8fa302ad8f545ee8bfad20c63329d79b76f72c 16 > /tmp/matrix-b16-build.log 2>&1
  rc16=$?
  echo "b16 rc=$rc16" >> /tmp/matrix-builds.status
fi
echo DONE >> /tmp/matrix-builds.status
