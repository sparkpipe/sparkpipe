#!/usr/bin/env bash
# Shared-socket family wrapper for the qwen38_27b developer lane (lane 1,
# TP4 on spark0-spark3). Runs INSIDE an admitted spark_queue job: prepare a
# private deployment under $SPARK_QUEUE_RUNTIME_ROOT, optionally warm the
# smoke expert set through the SAME shared socket (the < 5 s cold-launch
# preload), then launch the verified sparkpipe_model_residentd. Never
# starts a private weightd (multidev quickstart law).
#
# Required environment
#   SPARK_QUEUE_RUNTIME_ROOT / SPARK_QUEUE_RANK / SPARK_QUEUE_SIZE /
#   SPARK_QUEUE_ATTEMPT        (supplied by the queue)
#   QWEN38_27B_LANE_FIRMWARE_ROOT  verified lane build
#                               (tools/qwen38_27b_lane_build_release.sh)
#
# Optional environment (defaults: the tracked shared weightd socket
# /run/sparkpipe-weightd-shared/weightd.sock, lane-1 hosts/ports/mesh
# control 23016 / collective 53016 / transport 64016, weightd mesh lane 1,
# physical mesh ranks 0,1,2,3, node-local nvfp4a16 TP4 rank packs,
# preload ON, measurement mode ON):
#   QWEN38_27B_LANE_PRELOAD=0|1   batch-preload the smoke expert set (.wset
#                                 derived from the staged .experts manifest)
#   QWEN38_27B_LANE_MEASURE=0|1   1 = run residentd as a child, print
#                                 TIME-TO-LAUNCH at its ready line (A/B
#                                 measurement); 0 = plain exec shape
#
# The controller's run-family-job.sh calls this script as the job command;
# every participant stays inside the queue cgroup.
set -euo pipefail

: "${SPARK_QUEUE_RUNTIME_ROOT:?run inside a spark_queue job}"
: "${SPARK_QUEUE_RANK:?run inside a spark_queue job}"
: "${SPARK_QUEUE_SIZE:?run inside a spark_queue job}"
: "${SPARK_QUEUE_ATTEMPT:?run inside a spark_queue job}"
: "${QWEN38_27B_LANE_FIRMWARE_ROOT:?point at a verified lane build}"
: "${QWEN38_27B_LANE_SHARED_SOCKET:=/run/sparkpipe-weightd-shared/weightd.sock}"
export QWEN38_27B_LANE_SHARED_SOCKET

if [ "${SPARK_QUEUE_SIZE}" -ne 4 ]; then
    echo "qwen38_27b lane: TP4 lane expects SPARK_QUEUE_SIZE=4 (got ${SPARK_QUEUE_SIZE})" >&2
    exit 2
fi

T_BEGIN=$(python3 -c 'import time; print(f"{time.monotonic():.6f}")')
echo "QWEN38_27B_LANE_T_BEGIN $(date +%s.%N)"

here=$(cd -- "$(dirname -- "$0")" && pwd)
staged=$(python3 "${here}/qwen38_27b_lane_deployment.py")
runtime_root=$(printf '%s' "${staged}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["runtime_root"])')
pack_name=$(printf '%s' "${staged}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["pack_name"])')
pack_sha=$(printf '%s' "${staged}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["pack_sha256"])')
model_revision=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_revision"])' \
    "${runtime_root}/config/stage.json")
T_STAGED=$(python3 -c 'import time; print(f"{time.monotonic():.6f}")')
echo "QWEN38_27B_LANE_T_STAGED $(date +%s.%N) stage_s=$(python3 -c "print(f'{$T_STAGED-$T_BEGIN:.3f}')")"

if [ ! -S "${QWEN38_27B_LANE_SHARED_SOCKET}" ]; then
    echo "qwen38_27b lane: shared weightd socket is not present: ${QWEN38_27B_LANE_SHARED_SOCKET} (operator-level; do not start daemons by hand)" >&2
    exit 2
fi

# Batch-preload the smoke expert set (milestone 3): derive the .wset from
# the STAGED .experts manifest so the warmed set is exactly the pack's
# expert tier, then warm through the shared socket the resident attaches to.
if [ "${QWEN38_27B_LANE_PRELOAD:-1}" = "1" ]; then
    wset="${runtime_root}/packs/${pack_name}.wset"
    python3 - "${runtime_root}/packs/${pack_name}.experts" "${wset}" <<'PYEOF'
import struct, sys
from pathlib import Path
source, target = Path(sys.argv[1]), Path(sys.argv[2])
raw = source.read_bytes()
magic, version, count, zero = struct.unpack_from("<4I", raw, 0)
if magic != 0x58504557 or version != 2 or zero != 0:
    raise SystemExit("staged .experts manifest is not a v2 weightd manifest")
pairs = {}
for index in range(count):
    layer, expert = struct.unpack_from("<2I", raw, 16 + index * 48)
    pairs[(layer, expert)] = None
if not pairs:
    raise SystemExit("staged .experts manifest holds no expert groups")
target.write_bytes(b"".join(struct.pack("<2I", layer, expert)
                            for (layer, expert) in sorted(pairs)))
print(f"wset {target.name} keys={len(pairs)}")
PYEOF
    # Pool budget on the 2 MiB chunk basis (x2.15 rule): the dense expert
    # tier is 2,406,482,688 B raw per rank -> ~5.0 GiB chunked; 6 GiB covers
    # it with margin.
    SPARK_WEIGHTD_EXPERT_POOL_BYTES="${QWEN38_27B_LANE_EXPERT_POOL_BYTES:-6442450944}" \
    "${QWEN38_27B_LANE_FIRMWARE_ROOT}/bin/weightd_warm" \
        "${QWEN38_27B_LANE_SHARED_SOCKET}" \
        "${runtime_root}/packs/${pack_name}" \
        "${pack_sha}" "${model_revision}" "${SPARK_QUEUE_SIZE}" \
        --wset "${wset}" 300 >"${SPARK_QUEUE_RUNTIME_ROOT}/warm.log" 2>&1
    grep -q "WSET-WARM keys=" "${SPARK_QUEUE_RUNTIME_ROOT}/warm.log" || {
        echo "qwen38_27b lane: working-set warm failed:" >&2
        tail -5 "${SPARK_QUEUE_RUNTIME_ROOT}/warm.log" >&2
        exit 2
    }
    T_WARM=$(python3 -c 'import time; print(f"{time.monotonic():.6f}")')
    echo "QWEN38_27B_LANE_T_WARMED $(date +%s.%N) warm_s=$(python3 -c "print(f'{$T_WARM-$T_STAGED:.3f})'") $(grep -o 'WSET-WARM keys=[0-9]*' "${SPARK_QUEUE_RUNTIME_ROOT}/warm.log" | tail -1)"
else
    echo "QWEN38_27B_LANE_PRELOAD off (cold arm)"
fi

export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_SOCKET="${QWEN38_27B_LANE_SHARED_SOCKET}"
export SPARK_WEIGHTD_LANE="${QWEN38_27B_LANE_WEIGHTD_LANE:-1}"
export SPARK_TP_MESH_RANKS="${QWEN38_27B_LANE_MESH_RANKS:-0,1,2,3}"
export LD_LIBRARY_PATH="${runtime_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "qwen38_27b lane: rank=${SPARK_QUEUE_RANK} socket=${SPARK_WEIGHTD_SOCKET}" \
     "lane=${SPARK_WEIGHTD_LANE} mesh=${SPARK_TP_MESH_RANKS} runtime=${runtime_root}"

if [ "${QWEN38_27B_LANE_MEASURE:-1}" = "1" ]; then
    # A/B measurement mode: the resident stays a child of this job cgroup;
    # mark time-to-launch at its ready line, then follow it.
    log="${SPARK_QUEUE_RUNTIME_ROOT}/residentd.log"
    "${QWEN38_27B_LANE_FIRMWARE_ROOT}/bin/sparkpipe_model_residentd" \
        --deployment "${SPARK_QUEUE_RUNTIME_ROOT}/deployment.json" \
        --rank-index "${SPARK_QUEUE_RANK}" >"${log}" 2>&1 &
    child=$!
    for _ in $(seq 1 2400); do
        grep -q "model_residentd ready" "${log}" 2>/dev/null && break
        kill -0 "${child}" 2>/dev/null || { echo "residentd exited before ready:" >&2; tail -20 "${log}" >&2; exit 2; }
        sleep 0.1
    done
    grep -q "model_residentd ready" "${log}" || { echo "residentd not ready in 240s" >&2; tail -20 "${log}" >&2; exit 2; }
    T_READY=$(python3 -c 'import time; print(f"{time.monotonic():.6f}")')
    echo "QWEN38_27B_LANE_TIME_TO_LAUNCH $(date +%s.%N) ttl_s=$(python3 -c "print(f'{$T_READY-$T_BEGIN:.3f})'") $(grep -o 'model_residentd ready.*' "${log}" | head -1)"
    wait "${child}"
else
    exec "${QWEN38_27B_LANE_FIRMWARE_ROOT}/bin/sparkpipe_model_residentd" \
        --deployment "${SPARK_QUEUE_RUNTIME_ROOT}/deployment.json" \
        --rank-index "${SPARK_QUEUE_RANK}"
fi
