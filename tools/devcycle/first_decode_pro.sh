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

echo "== fleet_swap dsv4pro =="
"${ROOT}/tools/fleet_swap.sh" dsv4pro
sleep 30

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
