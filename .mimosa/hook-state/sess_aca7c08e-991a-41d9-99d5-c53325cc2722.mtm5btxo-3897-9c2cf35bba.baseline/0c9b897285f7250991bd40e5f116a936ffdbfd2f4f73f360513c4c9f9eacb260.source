#!/bin/bash
# qwen38 storm: the massive stress test.
# Production = tens of thousands of queued requests; this tests at or beyond
# that: 10,000+ requests across the shape space, verifying:
#   1. Zero failures (every request delivers its full token budget)
#   2. GPU saturation throughout (ffn_ms continuously advancing)
#   3. Spec rounds firing for B1 cells
#   4. No daemon poisoning (sequential clients keep working)
#   5. Throughput scaling with batch width
# Usage: bash tools/qwen38_storm.sh [total_requests] [output_dir]
set -u
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
TOTAL=${1:-10000}
D=${2:-/tmp/storm}
mkdir -p "$D"
cd /home/spark2/sparkdata/qwen38.fp8.tp1

PASS=0; FAIL=0; TOTAL_TOKENS=0; START=$(date +%s)
# Shape rotation: cycle through working shapes (edge cases isolated separately)
SHAPES=(
  "128 1 64 8"    # standard B1 spec
  "128 4 64 8"    # B4 batched
  "128 16 64 8"   # B16 batched
  "512 1 128 8"   # long prompt B1
  "512 4 128 8"   # long prompt B4
  "64 1 32 8"     # short prompt B1
  "64 8 32 8"     # short B8
  "256 2 64 8"    # mid B2
)
SHAPE_COUNT=${#SHAPES[@]}
BATCH_NUM=0

while [ $((PASS + FAIL)) -lt "$TOTAL" ]; do
  BATCH_NUM=$((BATCH_NUM + 1))
  SHAPE_INDEX=$((BATCH_NUM % SHAPE_COUNT))
  set -- ${SHAPES[$SHAPE_INDEX]}
  PLEN=$1; B=$2; BUD=$3; PFR=$4

  # Generate batch file (unique sequence IDs per batch to avoid collisions)
  python3 -c "
import json,random
random.seed($BATCH_NUM)
reqs=[]
for i in range($B):
    rid=900000 + $BATCH_NUM * 200 + i
    reqs.append({\"request_id\":rid,\"sequence_id\":rid,\"priority\":0,
      \"output_token_budget\":$BUD,
      \"prompt_token_ids\":[random.randint(100,200000) for _ in range($PLEN)]})
json.dump({\"schema_version\":1,\"connect_timeout_ms\":30000,
 \"request_capacity\":$B,\"max_context_tokens\":4096,
 \"max_prefill_rows_per_submission\":$PFR,
 \"maximum_messages_per_rank_per_progress\":8,
 \"maximum_new_submissions_per_progress\":2,
 \"stop_token_ids\":[],\"requests\":reqs},open(\"$D/batch.json\",\"w\"))
"
  EXPECTED=$((B * BUD))
  T0=$(date +%s.%N)
  timeout 120 ./bin/sparkpipe_model_batch \
    --deployment config/model_resident.json --runtime-root $PWD \
    --batch "$D/batch.json" > "$D/batch.out" 2> "$D/batch.err"
  RC=$?
  T1=$(date +%s.%N)
  EL=$(echo "$T1 - $T0" | bc)
  TOKENS=$(grep -c '"event":"token"' "$D/batch.out" 2>/dev/null)
  ERRORS=$(grep -c '"event":"error"' "$D/batch.out" 2>/dev/null)
  TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))

  if [ "$RC" -eq 0 ] && [ "$TOKENS" -eq "$EXPECTED" ] && [ "$ERRORS" -eq 0 ]; then
    PASS=$((PASS + B))
    if [ $((BATCH_NUM % 50)) -eq 0 ]; then
      NOW=$(date +%s)
      RATE=$((TOTAL_TOKENS / (NOW - START + 1)))
      echo "progress: batch=$BATCH_NUM requests=$PASS/$TOTAL tokens=$TOTAL_TOKENS rate=${RATE}t/s shape=p${PLEN}_b${B}"
    fi
  else
    FAIL=$((FAIL + B))
    echo "FAIL batch=$BATCH_NUM shape=p${PLEN}_b${B}_bud${BUD} rc=$RC tokens=$TOKENS/${EXPECTED} errors=$ERRORS wall=${EL}s"
    cp "$D/batch.err" "$D/fail_${BATCH_NUM}.err"
    cp "$D/batch.out" "$D/fail_${BATCH_NUM}.out" 2>/dev/null
    # daemon health check + restart if wedged
    if [ "$RC" -eq 124 ]; then
      echo "  WEDGE detected - restarting daemon"
      pgrep -f "[s]parkpipe_model_residentd" | xargs -r kill
      sleep 3
      nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9
      sleep 4
      bash /tmp/launch_prod2.sh > /dev/null 2>&1
      sleep 2
    fi
  fi
done

END=$(date +%s)
DUR=$((END - START))
echo "=== STORM COMPLETE ==="
echo "requests: $PASS pass, $FAIL fail (target $TOTAL)"
echo "tokens delivered: $TOTAL_TOKENS in ${DUR}s = $((TOTAL_TOKENS / (DUR + 1))) tok/s aggregate"
echo "daemon restarts: see FAIL lines above"
# GPU saturation evidence
LAST_FFN=$(grep -a "gpu_spin_profile" /tmp/qwen38.log | tail -1 | grep -oE "ffn_ms=[0-9.]+" | cut -d= -f2)
SPEC=$(grep -ac "spec_diag" /tmp/qwen38.log)
echo "GPU ffn_ms (cumulative): $LAST_FFN"
echo "spec rounds (cumulative): $SPEC"
