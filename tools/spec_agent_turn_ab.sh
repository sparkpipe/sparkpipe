#!/usr/bin/env bash
set -u
HUB=${HUB:-spec@100.123.97.61}
W=${W:-/srv/workspace/sparkpipe/spec-agent-turn-ab}
MODEL=${MODEL:-kimi-k3}
UNIT=${UNIT:-kv-k3-api}
APIDIR=${APIDIR:-k3-kv-api}
PORT=${PORT:-8459}
LOG=${LOG:-k3-kv-api.log}
BASE=${BASE:?BASE is the baseline commit, for example the merge base of the PR}
CAND=${CAND:?CAND is the candidate commit}
ARMS=${ARMS:-base:$BASE:$BASE cand:$CAND:$CAND base:$BASE:$BASE cand:$CAND:$CAND}
DRAFTD_SOURCE=${DRAFTD_SOURCE:-/srv/workspace/sparkpipe/draftd-5903345aa}
DRAFTD_ENV="DRAFTD_UNIT=sparkpipe-draftd DRAFTD_DRAFTER=/srv/workspace/sparkpipe/drafter-weights/kimi-k3-dflash2-lightseek-e77935fb DRAFTD_TARGET_SHARD=/srv/workspace/sparkpipe/drafter-weights/kimi-k3/model-00094-of-000096.safetensors DRAFTD_TAP_HOST=10.10.250.2 DRAFTD_TAP_PORT=7701 DRAFTD_DRAFT_PORT=7702 DRAFTD_WINDOW=4096 DRAFTD_CONTEXT_BYTES=8589934592"
KINDS=${KINDS:-agent,edit,prose}
BATCHES=${BATCHES:-1,2,4,8}
TOKENS=${TOKENS:-256}
UTC=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$W/hold-$UTC
hub() { ssh -o BatchMode=yes -o ConnectTimeout=8 "$HUB" "$@" < /dev/null; }
stage() {
  local commit
  for commit in "$BASE" "$CAND"; do
    git archive --format=tar "$commit" | ssh -o BatchMode=yes "$HUB" "rm -rf $W/api-$commit && mkdir -p $W/api-$commit && tar -C $W/api-$commit -xf - && echo $commit > $W/api-$commit/SOURCE_COMMIT && cd $W/api-$commit && export PATH=/usr/local/cuda/bin:\$PATH && nice make -j8 build/sparkpipe_model_api > ../api-$commit.out 2>&1; echo API-RC=\$? $commit; sha256sum build/sparkpipe_model_api | cut -c1-16" || return 1
    git archive --format=tar "$commit" tools/block_draftd.py tools/context_retention.py tools/hub_draftd.sh tools/spec_offline | ssh -o BatchMode=yes "$HUB" "rm -rf $W/draftd-$commit $W/draftd-$commit.tmp && mkdir -p $W/draftd-$commit.tmp && tar -C $W/draftd-$commit.tmp -xf - && mv $W/draftd-$commit.tmp/tools $W/draftd-$commit && rmdir $W/draftd-$commit.tmp && echo $commit > $W/draftd-$commit/SOURCE_COMMIT" || return 1
  done
  git archive --format=tar "$CAND" runtime | ssh -o BatchMode=yes "$HUB" "rm -rf $W/src && mkdir -p $W/src && tar -C $W/src -xf -" || return 1
  git show "$CAND:tools/batch_speculation_gate.py" | ssh -o BatchMode=yes "$HUB" "cat > $W/batch_speculation_gate.py"
}
case "${1:-}" in
stage) stage; exit $? ;;
ab) ;;
*) echo "usage: BASE=<commit> CAND=<commit> tools/spec_agent_turn_ab.sh stage|ab" >&2; exit 2 ;;
esac
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
draftd() { hub "env $DRAFTD_ENV DRAFTD_LOG=$1 DRAFTD_RESTART=$2 bash $3/hub_draftd.sh stop > /dev/null; env $DRAFTD_ENV DRAFTD_LOG=$1 DRAFTD_RESTART=$2 bash $3/hub_draftd.sh start $3"; }
restore() {
  trap - EXIT
  stamp "restore: production API and draft server for $MODEL"
  hub "$API_STOP"
  draftd /srv/workspace/sparkpipe/logs/draftd.log on-failure "$DRAFTD_SOURCE"
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
offset=0
for spec in $ARMS; do
  arm=${spec%%:*}; rest=${spec#*:}; commit=${rest%%:*}; drafter=${rest#*:}
  [ "$(status)" = "idle $MODEL None" ] || { echo "CONTROLLER-CHANGED $(status)"; exit 4; }
  hub "$API_STOP"
  draftd "$OUT/draftd-$arm-$drafter.log" no "$W/draftd-$drafter" || { echo "ARM-DRAFTD-DOWN $arm $drafter"; exit 5; }
  hub "systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit=$UNIT -p MemoryMax=2G -p MemorySwapMax=0 --working-directory=\$HOME/$APIDIR bash -c 'exec $W/api-$commit/build/sparkpipe_model_api --deployment model_resident.json --runtime-root \$HOME/$APIDIR/runtime --port $PORT >> $OUT/api-$arm-$commit.log 2>&1' > /dev/null"
  for i in $(seq 1 90); do hub "curl -sf $HEALTH > /dev/null" && break; sleep 2; done
  hub "curl -sf $HEALTH" > /dev/null || { echo "ARM-API-DOWN $arm $commit"; exit 5; }
  stamp "arm $arm $commit api $(hub "sha256sum $W/api-$commit/build/sparkpipe_model_api | cut -c1-16") draftd $drafter $(hub "sha256sum $W/draftd-$drafter/block_draftd.py | cut -c1-16")"
  offset=$((offset + 13))
  hub "for i in \$(seq 1 $offset); do curl -sf -m 120 -H 'Content-Type: application/json' -d '{\"prompt\":\"Say ok.\",\"max_tokens\":1,\"temperature\":0}' http://127.0.0.1:$PORT/v1/completions > /dev/null; done"
  hub "cd \$HOME/sparkpipe-chat && timeout 7200 python3 $W/batch_speculation_gate.py --engine http://127.0.0.1:$PORT --chat-dir \$HOME/sparkpipe-chat --model $MODEL --sources $W/src/runtime --kinds $KINDS --batches $BATCHES --repeats 1 --max-tokens $TOKENS --label $arm-$commit --output $OUT/gate.jsonl --streams $OUT/streams.jsonl 2>&1 | grep -E '^GATE|^\{' | tail -40"
  hub "grep -a -E 'batch_speculation|batch engine: model drafts' $OUT/api-$arm-$commit.log | tail -2 | cut -c1-2000"
done
api_arm() {
  hub "$API_STOP"
  hub "systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit=$UNIT -p MemoryMax=2G -p MemorySwapMax=0 --working-directory=\$HOME/$APIDIR bash -c 'exec $W/api-$1/build/sparkpipe_model_api --deployment model_resident.json --runtime-root \$HOME/$APIDIR/runtime --port $PORT >> $OUT/api-$2.log 2>&1' > /dev/null"
  for i in $(seq 1 90); do hub "curl -sf $HEALTH > /dev/null" && break; sleep 2; done
  hub "curl -sf $HEALTH" > /dev/null
}
RESTART_ARMS=${RESTART_ARMS:-base:$BASE cand:$CAND base:$BASE cand:$CAND}
RESTART_API=${RESTART_API:-$CAND}
for spec in $RESTART_ARMS; do
  arm=${spec%%:*}; drafter=${spec#*:}
  [ "$(status)" = "idle $MODEL None" ] || { echo "CONTROLLER-CHANGED $(status)"; exit 4; }
  hub "$API_STOP"
  draftd "$OUT/restart-draftd-$arm-$drafter.log" no "$W/draftd-$drafter" || { echo "RESTART-DRAFTD-DOWN $arm $drafter"; exit 6; }
  for session in 1 2; do
    api_arm "$RESTART_API" "restart-$arm-$drafter-$session" || { echo "RESTART-API-DOWN $arm $session"; exit 6; }
    hub "for i in \$(seq 1 $((session * 5))); do curl -sf -m 120 -H 'Content-Type: application/json' -d '{\"prompt\":\"Say ok.\",\"max_tokens\":1,\"temperature\":0}' http://127.0.0.1:$PORT/v1/completions > /dev/null; done"
    stamp "restart arm $arm draftd $drafter session $session"
    hub "cd \$HOME/sparkpipe-chat && timeout 3600 python3 $W/batch_speculation_gate.py --engine http://127.0.0.1:$PORT --chat-dir \$HOME/sparkpipe-chat --model $MODEL --sources $W/src/runtime --kinds agent --batches 1 --repeats 1 --max-tokens $TOKENS --label restart-$arm-$drafter-$session --output $OUT/restart.jsonl 2>&1 | grep -E '^GATE' | tail -2"
  done
  hub "grep -c TOKEN-MISMATCH $OUT/restart-draftd-$arm-$drafter.log; grep DRAFTD-STATS $OUT/restart-draftd-$arm-$drafter.log | tail -1"
done
