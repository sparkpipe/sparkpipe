#!/usr/bin/env bash
# restore_live.sh IDX — relaunch the live DSV4 Flash serving daemon exactly as
# snapshotted before the benchmark bounce (binary name, cwd, env, ids).
set -euo pipefail
IDX="$1"
ROOT=/tmp/dsv4-integrated-lean-3d962820-runtime
cd "$ROOT"
export LD_LIBRARY_PATH="$ROOT/lib":${LD_LIBRARY_PATH:-}
export SPARKPIPE_RELEASE_GENERATION=20260815000000
export SPARKPIPE_RELEASE_GIT_COMMIT=ef8fa302ad8f545ee8bfad20c63329d79b76f72c
export SPARKPIPE_RELEASE_ID="dsv4flash-serving-rank$IDX"
setsid -f bin/lean_residentd \
  --deployment config/model_resident.json --rank-index "$IDX" \
  >/tmp/dsv4flash-serving-rank$IDX.restore.log 2>&1 </dev/null
echo "restored rank$IDX"
