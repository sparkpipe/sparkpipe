#!/usr/bin/env bash
# first_decode_pro.sh — the DSV4 Pro ring test runbook (run from the pro
# worktree once the ring reservation is granted). One command: designate
# Pro fleet-wide, run the first decode, capture the receipt.
set -euo pipefail

SC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SC/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="/tmp/dsv4pro-first-${STAMP}.json"

echo "== preflight =="
"${SC}/preflight_pro.sh"

echo "== fleet_swap dsv4-pro =="
"${ROOT}/tools/fleet_swap.sh" dsv4-pro
sleep 30

echo "== rank liveness (expect 1 pro residentd per host, no stragglers) =="
rank_up=0
for host in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do
    count=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" "pgrep -cf 'sparkpipe_model_residentd --deployment config/model_resident.json --rank-index' 2>/dev/null || echo 0")
    if [[ "${count}" == "1" ]]; then
        rank_up=$((rank_up + 1))
    else
        echo "rank liveness: ${host} residentd_count=${count}"
    fi
done
echo "rank liveness: ${rank_up}/16 up"
if [[ "${rank_up}" != "16" ]]; then
    echo "NOTE: not all 16 ranks are up; check /tmp/fleet-swap-dsv4-pro-*.log on the affected hosts" >&2
fi

echo "== first decode (this may take minutes: 16-rank start + pipeline fill) =="
python3 "${ROOT}/tools/model_stream_decode_benchmark.py" --output "${OUT}" \
    ssh -o BatchMode=yes spark0 \
        /home/spark0/sparkdata/dsv4_pro.tp4pp4/bin/sparkpipe_model_batch \
        --deployment /home/spark0/sparkdata/dsv4_pro.tp4pp4/config/model_resident.json \
        --runtime-root /home/spark0/sparkdata/dsv4_pro.tp4pp4 \
        --batch /tmp/dsv4pro-o128-batch.json

echo "== receipt =="
jq -r '{decode_tokens_per_second, token_count, ttft_seconds, total_seconds, inter_token_p95_seconds}' "${OUT}"
echo "token_hash:"
jq -r '[.events[] | select(.event.event == "token") | .event.token_id] | join(",") + "\n"' "${OUT}" | shasum -a 256
echo "receipt: ${OUT}"

echo "== restore (after the measured window) =="
echo "tools/fleet_swap.sh <previous-big-model>   # restores the pre-swap snapshot"
