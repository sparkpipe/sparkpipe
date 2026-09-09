#!/usr/bin/env bash
# dsv4flash: daemon-side lazy receipts on the PLACED rank1 pack + the
# family-produced v2 manifest — supervised weightd (lane socket, same
# queue-owned cgroup per WEIGHTD_SUPERVISED_STARTUP) + weightd_smoke as the
# consumer. GPU job (the daemon's arena + copies are CUDA).
set -euo pipefail
cd /home/spark2/lane-dsv4flash-build
export PATH="/usr/local/cuda/bin:$PATH"

echo "== build sparkpipe_weightd + weightd_smoke"
make -j4 build/sparkpipe_weightd NVCC=/usr/local/cuda/bin/nvcc
cc -O2 -Wall -Wextra -Werror -I. -Iinclude \
    tools/weightd_smoke.c build/libsparkpipe_runtime.a \
    build/libsparkpipe_core.a -pthread -o build/weightd_smoke

SOCKET=/tmp/dsv4flash_lane_weightd.sock
rm -f "$SOCKET"
echo "== supervised weightd (lane socket, same cgroup)"
./build/sparkpipe_weightd --socket "$SOCKET" &
DAEMON_PID=$!
for i in $(seq 1 20); do
    [ -S "$SOCKET" ] && break
    sleep 0.5
done
[ -S "$SOCKET" ] || { echo "DAEMON-SOCKET-MISSING"; kill -TERM $DAEMON_PID; exit 1; }

trap 'kill -TERM $DAEMON_PID 2>/dev/null || true' EXIT

echo "== weightd_smoke: rank1 pack, my v2 manifest, 384 MiB pool, 64 touches"
export SPARK_WEIGHTD_SOCKET="$SOCKET"
./build/weightd_smoke \
    /home/spark2/sparkdata/dsv4flash.tp16/packs/dsv4flash.tp16.rank1.spstage \
    dsv4flash 7872f01b1d1fe23eabc4c98b48bffcef5a386062 384 64

echo "LAZY-SMOKE-PASS"
