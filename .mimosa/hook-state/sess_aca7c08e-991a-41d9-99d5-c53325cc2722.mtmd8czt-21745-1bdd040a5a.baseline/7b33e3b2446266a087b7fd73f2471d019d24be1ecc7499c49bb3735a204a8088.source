#!/usr/bin/env bash
# glm53_m5_cell_task.sh — glm5_next M5 first perf cell (exact-32K B1 decode),
# queue-staged on spark0, holding all 16 nodes. Shape per MODEL_DEV_GUIDE +
# the q27b known-good (cache-drop IN THE TASK CMD — the GB10 page cache ate
# two q27b serve tasks) and the g5dsa wave-task template.
#
# Sequence: drop caches fleet-wide -> clean serve wave (registrars -> clean
# slate -> 16 residentds same-second -> ready -> api) -> health gate ->
# generate the M5 batch (tools/glm5_next_m5_batch.py: 17 COMPSEC prompts
# cycled to EXACTLY 32768-256 prompt ids, budget 256, no stop tokens) ->
# 8 sequential batch runs (B1 decode ~6-8s each; the loop keeps the fleet
# busy across the 5s telemetry poll — bursty single runs hide) -> receipt
# pair (dashboard summary + rank-local nvidia-smi DURING a run) -> preserve
# logs. Perf numbers must record context/batch/topology/precision with them.
#
# Stage first (controller): the lane tree at ~/g5m5-src on spark0 with
# bin/sparkpipe_registrar staged (tools/registrar_stage.sh) and the
# runtime roots unchanged (/home/<h>/sparkdata/glm5_next.tp16).
set -uo pipefail
SRC="${G5M5_SRC:-$HOME/g5m5-src}"
RUNS="${G5M5_RUNS:-8}"
API=http://localhost:8433
cd "$SRC" || { echo "NO-SRC $SRC"; exit 1; }

echo "== cache-drop on all 16 (GB10 page-cache trap; q27b serve-6/7) =="
for h in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 \
         spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" \
    "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && free -g | head -2" \
    || { echo "CACHE-DROP-FAIL $h"; exit 1; }
done

echo "== clean serve wave (no probes) =="
bash tools/glm5_next_wave.sh full || { echo "WAVE-FAIL"; exit 1; }

echo "== api health gate =="
health=""
for i in $(seq 1 60); do
  health=$(curl -s --max-time 3 "$API/health" || true)
  echo "health[$i]: $health"
  case "$health" in *'"status":"ok"'*) break ;; esac
  sleep 5
done
case "$health" in *'"status":"ok"'*) ;; *) echo "API-NEVER-READY"; exit 1 ;; esac

echo "== M5 batch: exact-32K B1 (32512 prompt + 256 budget) =="
# rows must be <= deployment runtime_limits.max_input_rows (16 in the
# current deployment); the daemon is SINGLE-CLIENT — the api holds the
# control session, so requests go through the api with prompt_token_ids.
python3 tools/glm5_next_m5_batch.py \
  --fixture qualification/ds4_eval/quality-fixtures-glm5.3-flash.json \
  --rows 16 \
  --out /tmp/glm53-m5-exact32k-b1.json || { echo "BATCH-GEN-FAIL"; exit 1; }
python3 - <<'PYEOF'
import json
b=json.load(open("/tmp/glm53-m5-exact32k-b1.json"))
r=b["requests"][0]
json.dump({"prompt_token_ids":r["prompt_token_ids"],
           "max_tokens":r["output_token_budget"],"temperature":0},
          open("/tmp/glm53-m5-api-req.json","w"))
print("api request staged:",len(r["prompt_token_ids"]),"prompt ids")
PYEOF

rr=/home/spark0/sparkdata/glm5_next.tp16
for n in $(seq 1 "$RUNS"); do
  echo "== run $n/$RUNS =="
  ( for delay in 30 90 180 300; do sleep "$delay"; \
    curl -s --max-time 5 http://127.0.0.1:8765/api/summary \
      >> "/tmp/glm53-m5-summary-run$n.json"; echo >> "/tmp/glm53-m5-summary-run$n.json"; \
    ssh -o BatchMode=yes sparkc "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv" \
      >> "/tmp/glm53-m5-smi-runc-run$n.txt"; done ) &
  probe_pid=$!
  curl -s --max-time 3600 -X POST http://localhost:8433/v1/completions \
    -H "Content-Type: application/json" -d @/tmp/glm53-m5-api-req.json \
    -o "/tmp/glm53-m5-run$n.out" -w "WALL %e s\n" > "/tmp/glm53-m5-run$n.err"
  wait "$probe_pid" 2>/dev/null
  python3 - "/tmp/glm53-m5-run$n.out" <<'EOF'
import json,hashlib,sys
try:
    r=json.load(open(sys.argv[1]))
    toks=r.get("tokens",[])
    print(f"tokens={len(toks)} status={r.get('status')} sha256={hashlib.sha256(str(toks).encode()).hexdigest()[:16]}")
except Exception as e:
    print("PARSE-FAIL",e,open(sys.argv[1]).read()[:120])
EOF
done

echo "== receipts =="
ls -la /tmp/glm53-m5-*.json /tmp/glm53-m5-*.txt 2>/dev/null
echo "context: exact-32K (32768 positions) batch=B1 topology=TP16 host-rdma precision=fp8-experts"
echo TASK-DONE
# appended: leave the fleet clean for the next exclusive window
# (TERM-only, cwd-scoped — glm5_next_wave.sh stop)
echo "== wave stop (clean fleet handoff) =="
bash tools/glm5_next_wave.sh stop || echo "WARN: wave stop reported failure"
echo CELL-DONE
