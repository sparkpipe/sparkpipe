#!/usr/bin/env bash
# qwen4_flash post-wave B1 cells: run AFTER the 16-rank wave reports
# "ALL 16 RANKS READY" (tools/qwen4_flash_fleet16_launch.sh). One client
# per residentd: the api must be DOWN during batch cells - this script
# TERMs it first (launcher --api-term, no-op if absent).
#
#   smoke   - B1 smoke batch, token-stream hash + decode rate (the receipt)
#   cell32k - B1 exact-32K cell (32640+128 = 32768 = declared max positions)
#   both    - smoke then cell32k
#
# Usage: qwen4_flash_wave_cells.sh --table <launch_table.json> both
set -euo pipefail

table_path=""
mode="${1:-}"
[[ "$mode" == "--table" ]] && { table_path="$2"; shift 2; mode="${1:-both}"; }
[[ -n "$table_path" ]] || { echo "usage: $0 --table <launch_table.json> [smoke|cell32k|both]" >&2; exit 2; }
case "$mode" in smoke|cell32k|both) ;; *) echo "unknown mode $mode" >&2; exit 2 ;; esac

coordinator_host=$(python3 -c "import json;print(json.load(open('$table_path'))[0]['host'])")
launcher="$(cd "$(dirname "$0")" && pwd)/qwen4_flash_fleet16_launch.sh"
[[ -x "$launcher" ]] || launcher="tools/qwen4_flash_fleet16_launch.sh"
deploy="/home/$coordinator_host/sparkdata/qwen4_flash.tp4/deploy_v4"
receipt="$deploy/wave_cells-$(date -u +%Y%m%dT%H%M%SZ).receipt.md"

token_csv_hash() {
  python3 -c '
import hashlib, json, sys
ev = json.load(open(sys.argv[1])).get("events", [])
ids = ",".join(str(e["event"]["token_id"]) for e in ev if e.get("event", {}).get("event") == "token")
print(hashlib.sha256((ids + "\n").encode()).hexdigest())
' "$1"
}
decode_rate() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("decode_tokens_per_second"))' "$1"; }

run_cell() {  # name batch_json
  local name="$1" batch="$deploy/$2" out ts hash rate
  [[ -s "$batch" ]] || { echo "BATCH MISSING: $batch" >&2; exit 5; }
  # one-client rule: api down during batch cells
  "$launcher" --table "$table_path" --api-term >/dev/null 2>&1 || true
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  out="$deploy/cells-$name-$ts.json"
  echo "== $name start $(date -u +%FT%TZ) batch=$2" | tee -a "$receipt"
  ssh -o BatchMode=yes "$coordinator_host" \
    "$deploy/bin/sparkpipe_model_batch --deployment '$deploy/deployment.json' --runtime-root '$deploy' --batch '$batch'" \
    > "$out" 2> "$out.err" || { echo "== $name BATCH FAILED (see $out.err)" | tee -a "$receipt"; tail -5 "$out.err" >&2; exit 6; }
  hash=$(token_csv_hash "$out")
  rate=$(decode_rate "$out")
  echo "== $name PASS hash=$hash decode=$rate tok/s out=$name-$ts.json" | tee -a "$receipt"
  curl -s --max-time 5 http://127.0.0.1:8765/api/summary > "$deploy/cells-$name-$ts.telemetry.json" 2>/dev/null || true
}

echo "# qwen4_flash wave cells receipt $(date -u +%FT%TZ)" > "$receipt"
echo "table=$table_path mode=$mode" >> "$receipt"
if [[ "$mode" == "smoke" || "$mode" == "both" ]]; then
  run_cell smoke q4f_smoke_batch.json
  # determinism gate: second smoke pass must hash identically (exactness
  # before trusting any timing from this receipt)
  run_cell smoke2 q4f_smoke_batch.json
  h1=$(grep " PASS smoke " "$receipt" | tail -2 | head -1 | sed 's/.*hash=\([0-9a-f]*\).*/\1/')
  h2=$(grep " PASS smoke2 " "$receipt" | tail -1 | sed 's/.*hash=\([0-9a-f]*\).*/\1/')
  if [[ -n "$h1" && "$h1" == "$h2" ]]; then
    echo "== DETERMINISM PASS smoke==smoke2 hash=${h1:0:16}" | tee -a "$receipt"
  else
    echo "== DETERMINISM FAIL smoke=$h1 smoke2=$h2 - RED STOP, timings void" | tee -a "$receipt"
    exit 8
  fi
fi
[[ "$mode" == "cell32k" || "$mode" == "both" ]] && run_cell cell32k qwen4_flash_b1_32k_batch.json
echo "receipt: $receipt"
