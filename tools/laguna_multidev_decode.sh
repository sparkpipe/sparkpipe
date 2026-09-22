#!/bin/bash
# laguna_multidev_decode.sh — small-B* decode driver against a RUNNING
# lane-8 attach attempt (multidev M3/M4). Runs on spark0 as a separate
# queue job beside the attach job: it waits for the coordinator's
# residentd readiness inside the attach attempt's log, launches the
# common sparkpipe_model_api against the attach attempt's PRIVATE
# deployment (read-only; the residentds own the control endpoints), and
# drives the canonical T1 prompt set (qualification/t1_reference/laguna/
# prompts.json - the same prompts the M2 census measured its working set
# from) at B1, then B2 when B1 is exact.
#
# The API listener binds one number inside the lane-8 session block's
# spare tail (23740; reserve it with --ports on THIS job).
#
# Environment
#   LAGUNA_DECODE_ATTEMPT   the ATTACH job's 32-hex attempt id (whose
#                           /tmp/sparkqueue-<attempt> namespace holds the
#                           private deployment + residentd logs)
#   LAGUNA_DECODE_API_PORT  default 23740
# Receipt lines: DECODE-REQ and DECODE-DONE with token ids + wall ms.
set -euo pipefail

ATTEMPT="${LAGUNA_DECODE_ATTEMPT:?set LAGUNA_DECODE_ATTEMPT to the attach job attempt id}"
API_PORT="${LAGUNA_DECODE_API_PORT:-23740}"
ROOT="/tmp/sparkqueue-$ATTEMPT"
CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="$(hostname)"
[ "$HOST" = "spark0" ] || { echo "decode driver runs on spark0 (the coordinator)" >&2; exit 2; }
[ -d "$ROOT" ] || { echo "attach attempt root missing: $ROOT" >&2; exit 2; }
[ -f "$ROOT/deployment.json" ] || { echo "attach deployment missing" >&2; exit 2; }

fail() { echo "laguna-decode: $*" >&2; exit 1; }

# 1. The common API against the attach attempt's private deployment
#    (this job's own cgroup; TERM-stopped on exit). The API waits for
#    the engine ranks itself (model_api.c engine_ranks_not_ready) and
#    serves GET /health once up - poll it, no log parsing.
API_LOG="$ROOT/api.log"
make -s -C "$CHECKOUT" build/sparkpipe_model_api
"$CHECKOUT/build/sparkpipe_model_api" \
  --deployment "$ROOT/deployment.json" \
  --runtime-root "$ROOT" \
  --port "$API_PORT" >"$API_LOG" 2>&1 &
API_PID=$!
trap 'kill -TERM "$API_PID" 2>/dev/null || true' EXIT
HEALTHY=""
for _ in $(seq 1 240); do
  if curl -sf --max-time 5 "http://127.0.0.1:$API_PORT/health" >"$ROOT/health.json" 2>/dev/null; then
    HEALTHY=1
    break
  fi
  kill -0 "$API_PID" 2>/dev/null || { tail -5 "$API_LOG" >&2; fail "api exited early"; }
  sleep 5
done
[ -n "$HEALTHY" ] || { tail -5 "$API_LOG" >&2; fail "api not healthy in 20 min (engine ranks not ready?)"; }
echo "DECODE-API health=$(cat "$ROOT/health.json")"

# 3. Canonical prompts at B1 (single-sequence completions, the exact T1
#    fixture prompts and token budgets).
decode() {
  local name="$1" ids="$2" new_tokens="$3" began ended
  began=$(date +%s%N)
  body="$(curl -sf --max-time 120 -X POST "http://127.0.0.1:$API_PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt_token_ids\": [$ids], \"max_tokens\": $new_tokens}")" || {
    echo "DECODE-REQ name=$name status=ERROR" >&2; return 1; }
  ended=$(date +%s%N)
  echo "DECODE-REQ name=$name wall_ms=$(( (ended - began) / 1000000 )) body=${body:0:400}"
}

python3 - "$CHECKOUT" <<'PYIDS' > /tmp/laguna-decode-ids.$$
import json, sys
prompts = json.load(open(sys.argv[1] + "/qualification/t1_reference/laguna/prompts.json"))["prompts"]
for prompt in prompts:
    print(prompt["name"] + "\t" + ",".join(str(t) for t in prompt["prompt_token_ids"]) + "\t" + str(prompt["new_tokens"]))
PYIDS
while IFS=$'\t' read -r name ids new_tokens; do
  [ -n "$name" ] || continue
  decode "$name" "$ids" "$new_tokens"
done < "/tmp/laguna-decode-ids.$$"
rm -f "/tmp/laguna-decode-ids.$$"
echo "DECODE-DONE attempt=$ATTEMPT api_port=$API_PORT"
