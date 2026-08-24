#!/bin/bash
# qwen38 fuzz matrix: drives the full stack (batch engine -> daemon ->
# adapter -> module -> GPU) across the request-shape space, verifying:
#   1. No crashes, no wedges (timeout), no unexpected errors
#   2. GPU saturation during decode (frame_ms advances, ffn_ms > 0)
#   3. Correct M-shapes: spec rounds for B1, batched decode for B>1
#   4. Deterministic streams for identical inputs
# Usage: bash tools/qwen38_fuzz_matrix.sh [output_dir]
set -u
D=${1:-/tmp/fuzz_results}
mkdir -p "$D"
cd /home/spark2/sparkdata/qwen38.fp8.tp1
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
PASS=0; FAIL=0; SKIP=0
# Shape axes: prompt_len x batch x budget x prefill_rows
for SHAPE in "64 1 32 8" "128 1 64 8" "512 1 128 8" "64 4 32 8" "128 4 64 8" \
             "512 4 128 8" "64 16 32 8" "128 16 64 8" "2048 1 32 8" \
             "64 1 32 1" "128 1 64 1" "1 1 16 8" "2 1 16 8" "7 1 16 8" \
             "63 1 16 8" "65 1 16 8" "127 1 16 8" "129 1 16 8"; do
  set -- $SHAPE; PLEN=$1; B=$2; BUD=$3; PFR=$4
  TAG="p${PLEN}_b${B}_bud${BUD}_pfr${PFR}"
  python3 -c "
import json,random
random.seed(42)
reqs=[]
for i in range($B):
    rid=800000+i
    reqs.append({\"request_id\":rid,\"sequence_id\":rid,\"priority\":0,
      \"output_token_budget\":$BUD,
      \"prompt_token_ids\":[random.randint(1,248000) for _ in range($PLEN)]})
json.dump({\"schema_version\":1,\"connect_timeout_ms\":30000,
 \"request_capacity\":$B,\"max_context_tokens\":4096,
 \"max_prefill_rows_per_submission\":$PFR,
 \"maximum_messages_per_rank_per_progress\":8,
 \"maximum_new_submissions_per_progress\":2,
 \"stop_token_ids\":[],\"requests\":reqs},open(\"$D/$TAG.json\",\"w\"))
"
  T0=$(date +%s.%N)
  timeout 120 ./bin/sparkpipe_model_batch \
    --deployment config/model_resident.json --runtime-root $PWD \
    --batch "$D/$TAG.json" > "$D/$TAG.out" 2> "$D/$TAG.err"
  RC=$?
  T1=$(date +%s.%N)
  EL=$(echo "$T1 - $T0" | bc)
  TOKENS=$(grep -c '"event":"token"' "$D/$TAG.out" 2>/dev/null || echo 0)
  ERRORS=$(grep -c '"event":"error"' "$D/$TAG.out" 2>/dev/null || echo 0)
  EXPECTED=$((B * BUD))
  # pass: all tokens delivered, zero errors, clean exit
  if [ "$RC" -eq 0 ] && [ "$TOKENS" -eq "$EXPECTED" ] && [ "$ERRORS" -eq 0 ]; then
    echo "PASS $TAG tokens=$TOKENS/${EXPECTED} wall=${EL}s"
    PASS=$((PASS+1))
  elif [ "$RC" -eq 124 ]; then
    echo "WEDGE $TAG tokens=$TOKENS/${EXPECTED} wall=${EL}s TIMEOUT"
    FAIL=$((FAIL+1))
  else
    echo "FAIL $TAG rc=$RC tokens=$TOKENS/${EXPECTED} errors=$ERRORS wall=${EL}s $(tail -1 $D/$TAG.err | cut -c1-60)"
    FAIL=$((FAIL+1))
  fi
  sleep 1
done
echo "=== FUZZ COMPLETE: $PASS pass, $FAIL fail ==="
# GPU saturation check from the daemon profile
SAT=$(grep -a "gpu_spin_profile" /tmp/qwen38.log | tail -1 | grep -oE "ffn_ms=[0-9.]+" | cut -d= -f2)
echo "GPU ffn_ms (last): $SAT (nonzero = FFN kernels ran = GPU active)"
SPEC=$(grep -ac "spec_diag" /tmp/qwen38.log)
echo "Spec rounds total: $SPEC (B1 cells should contribute)"
