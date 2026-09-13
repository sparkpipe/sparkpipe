#!/usr/bin/env bash
# start_renamed.sh IDX — launch the lean serving daemon from a neutral binary
# path so coarse 'pkill -f residentd' sweeps from other benches cannot see it.
# Same binary bits, same config, same everything else.
set -euo pipefail
IDX="$1"
ROOT=/tmp/dsv4-integrated-lean-3d962820-runtime
BIN=/tmp/dsv4-flash-mx/residentd_mx
mkdir -p /tmp/dsv4-flash-mx
if [ ! -x "$BIN" ] || ! cmp -s "$ROOT/bin/lean_residentd" "$BIN"; then
  cp "$ROOT/bin/lean_residentd" "$BIN"
fi
cd "$ROOT"
export LD_LIBRARY_PATH="$ROOT/lib":${LD_LIBRARY_PATH:-}
export SPARKPIPE_RELEASE_GENERATION=20260815000000
export SPARKPIPE_RELEASE_GIT_COMMIT=ef8fa302ad8f545ee8bfad20c63329d79b76f72c
export SPARKPIPE_RELEASE_ID="dsv4flash-serving-rank$IDX"
setsid -f "$BIN" \
  --deployment config/model_resident.json --rank-index "$IDX" \
  >/tmp/mx-serving-rank$IDX.log 2>&1 </dev/null
echo "mx-started rank$IDX"
