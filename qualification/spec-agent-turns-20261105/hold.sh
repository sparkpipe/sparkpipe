#!/usr/bin/env bash
set -u
HUB=spec@100.123.97.61
W=/srv/workspace/sparkpipe/spec-agentturn
MODEL=${MODEL:-kimi-k3}
UNIT=${UNIT:-kv-k3-api}
APIDIR=${APIDIR:-k3-kv-api}
PORT=${PORT:-8459}
LOG=${LOG:-k3-kv-api.log}
ARMS=${ARMS:-base:1caf5401e cand:fcef5a3cc base:1caf5401e cand:fcef5a3cc}
KINDS=${KINDS:-agent,edit,prose}
BATCHES=${BATCHES:-1,2,4,8}
TOKENS=${TOKENS:-256}
UTC=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$W/hold-$UTC
hub() { ssh -o BatchMode=yes -o ConnectTimeout=8 "$HUB" "$@" < /dev/null; }
swap() { hub "token=\$(cat /srv/workspace/sparkpipe/swap-controller/token) && printf 'Authorization: Bearer %s\\n' \"\$token\" | curl -sf -H @- $*"; }
status() { swap localhost:8440/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["state"], d["active"], d["error"])'; }
stamp() { echo "== $(date -u +%FT%TZ) $*"; }
api_command() { python3 -c 'import json,sys; m=[m for m in json.load(open(sys.argv[1]))["models"] if m["id"]==sys.argv[2]][0]; print(m[sys.argv[3]])' deployment/hub/model_swap.json "$MODEL" "$1"; }
exec > >(tee -a "hold-$UTC.log") 2>&1
stamp "spec agent-turn hold $UTC model $MODEL arms [$ARMS] kinds $KINDS batches $BATCHES tokens $TOKENS"
[ "$(status | cut -d' ' -f1)" = idle ] || { echo "CONTROLLER-BUSY $(status)"; exit 2; }
if hub "pgrep -f '[n]ll_score.py|[n]ll_compare.py|[r]elease_capture.py|[a]pi_serving_perf' > /dev/null"; then echo "ANOTHER-HARNESS-RUNNING"; exit 2; fi
PREVIOUS=$(status | cut -d' ' -f2)
API_START=$(api_command api_start)
API_STOP=$(api_command api_stop)
HEALTH=$(api_command health)
restore() {
  trap - EXIT
  stamp "restore: production API for $MODEL"
  hub "$API_STOP"
  hub "$API_START"
  for i in $(seq 1 60); do hub "curl -sf $HEALTH > /dev/null" && break; sleep 2; done
  if [ "$PREVIOUS" != "$MODEL" ] && [ "$PREVIOUS" != None ]; then
    swap -X POST -d "{\"model\":\"$PREVIOUS\"}" localhost:8440/activate > /dev/null
    for i in $(seq 1 120); do [ "$(status)" = "idle $PREVIOUS None" ] && break; sleep 5; done
  fi
  stamp "hold end: $(status) -> $OUT"
}
trap restore EXIT
hub "mkdir -p $OUT"
swap -X POST -d "{\"model\":\"$MODEL\"}" localhost:8440/activate > /dev/null
for i in $(seq 1 180); do [ "$(status)" = "idle $MODEL None" ] && break; sleep 5; done
[ "$(status)" = "idle $MODEL None" ] || { echo "ACTIVATE-FAILED $(status)"; exit 3; }
stamp "active: $(status)"
for spec in $ARMS; do
  arm=${spec%%:*}; commit=${spec#*:}
  [ "$(status)" = "idle $MODEL None" ] || { echo "CONTROLLER-CHANGED $(status)"; exit 4; }
  hub "$API_STOP"
  hub "systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit=$UNIT -p MemoryMax=2G -p MemorySwapMax=0 --working-directory=\$HOME/$APIDIR bash -c 'exec $W/api-$commit/build/sparkpipe_model_api --deployment model_resident.json --runtime-root \$HOME/$APIDIR/runtime --port $PORT >> $OUT/api-$arm-$commit.log 2>&1' > /dev/null"
  for i in $(seq 1 90); do hub "curl -sf $HEALTH > /dev/null" && break; sleep 2; done
  hub "curl -sf $HEALTH" > /dev/null || { echo "ARM-API-DOWN $arm $commit"; exit 5; }
  stamp "arm $arm $commit api $(hub "sha256sum $W/api-$commit/build/sparkpipe_model_api | cut -c1-16")"
  hub "cd \$HOME/sparkpipe-chat && timeout 7200 python3 $W/batch_speculation_gate.py --engine http://127.0.0.1:$PORT --chat-dir \$HOME/sparkpipe-chat --model $MODEL --sources $W/src/runtime --kinds $KINDS --batches $BATCHES --repeats 1 --max-tokens $TOKENS --label $arm-$commit --output $OUT/gate.jsonl 2>&1 | grep -E '^GATE|^\{' | tail -40"
  hub "grep -a -E 'batch_speculation|batch engine: model drafts' $OUT/api-$arm-$commit.log | tail -2 | cut -c1-2000"
done
