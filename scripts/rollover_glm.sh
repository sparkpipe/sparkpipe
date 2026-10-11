#!/usr/bin/env bash
set -u
. "$(dirname "$0")/rollover_lib.sh"
TAG="$1"
case "$TAG" in
  full)
    SWAP=glm-5.3; M=glm53_full; REV=935644c05e76fc198714f4cca449fd8b970ff6d7; PACKGLOB='packs/*.glm52sp'; WARMARGS="--family glm52 --shared 78 256 1800"
    GOLDREF=qualification/glm53_full/golden/port_graphs_lp.json
    SMOKES_MIN=6; GOLDCMD="python3 tools/glm53_full_golden.py capture --prompts qualification/glm53_full/golden/prompts.json --port 8475 --max-tokens 128 --logprobs 4 --label rollover-$V --out" ;;
  flash)
    SWAP=glm-5.3-flash; M=glm53_flash; REV=84c6a6aa9497188e15a635ba793b0f95a79b1033; PACKGLOB='packs/*.sp'; WARMARGS="--shared"
    GOLDREF=qualification/glm53_flash/golden/port-8d27916a4-warm.json
    SMOKES_MIN=5; GOLDCMD="python3 tools/glm53_flash_golden.py --port 8476 --label rollover-$V --output" ;;
  *) echo "usage: rollover_glm.sh full|flash"; exit 2 ;;
esac
. "$W/scripts/rollover_env_$M.sh"
PORT=$(python3 -c "import sys; sys.path.insert(0, '$W/tools'); import stage_lane; print(stage_lane.MODELS['$M']['api_port'])")
ROOT="\$HOME/stage-lanes/$M/lane$STAGE_LANE/root"
API="$M-kv-api"
UNIT="kv-${M//_/-}-api"
Q="qualification/$M/rollover-$V"
L="$D/$TAG"
mkdir -p "$L"
cd "$L"
UTC=$(date -u +%Y%m%dT%H%M%SZ)
echo "$UTC" > backup_utc
WARM="cd $ROOT && SPARK_WEIGHTD_EXPERT_POOL_BYTES=$STAGE_EXPERT_POOL_BYTES ./bin/weightd_warm $STAGE_WEIGHTD_SOCKET \$(readlink -f $PACKGLOB) \$(cut -c1-64 packs/pack.sha256) $REV 16 $WARMARGS > warm.log 2>&1"

compsec() {
  hub "cd ~/rollover/tree && mkdir -p $Q && rm -rf $Q/compsec17-$1 && timeout 7200 python3 tools/compsec17_openai.py --endpoint http://127.0.0.1:4000 --model $SWAP --reasoning-effort minimal --out $Q/compsec17-$1" > "compsec_$1.log" 2>&1
  echo "compsec $1 rc=$?: $(grep -o 'RESULT: [0-9]*/[0-9]*' "compsec_$1.log") fails: $(grep -v ' PASS ' "compsec_$1.log" | grep -o 'compsec-[0-9]*' | tr '\n' ' ')"
}

restart_services() {
  hub "systemctl --user restart sparkpipe-swap sparkpipe-chat; sleep 8; for i in \$(seq 1 30); do curl -s localhost:8433/v1/models > /dev/null && curl -s localhost:8440/status > /dev/null && break; sleep 2; done"
  echo "services: controller $(controller); chat models $(hub "curl -s localhost:8433/v1/models" | python3 -c "import json,sys; print(len(json.load(sys.stdin)['data']))")"
}

revert() {
  stamp "REVERT $SWAP to the legacy entry"
  activate qwen3.8-max
  hub "cp ~/sparkpipe-swap/model_swap.json ~/sparkpipe-swap/model_swap.json.port-$UTC; cp ~/sparkpipe-chat/config.json ~/sparkpipe-chat/config.json.port-$UTC; cp ~/sparkpipe-swap/model_swap.json.prev-$UTC ~/sparkpipe-swap/model_swap.json; cp ~/sparkpipe-chat/config.json.prev-$UTC ~/sparkpipe-chat/config.json"
  restart_services
}

finish() {
  restore_max
  fan "$ROOT/lane.sh reclaim > /dev/null 2>&1 && echo ok" | grep -v ': ok$'
  nodes_up || exit 9
  stamp "done $SWAP ($(controller))"
}

stamp "$SWAP precheck"
[ "$(controller)" = "idle qwen3.8-max None" ] || { echo "PRECHECK-FAILED: $(controller)"; exit 1; }
nodes_up || exit 9

stamp "legacy build"
if activate "$SWAP"; then compsec old; else echo "LEGACY-ACTIVATE-FAILED (no old COMPSEC)"; fi
restore_max || exit 1

stamp "port setup + api-setup"
bash "$W/tools/stage_lane.sh" setup > setup.log 2>&1 || { tail -3 setup.log; echo SETUP-FAILED; finish; exit 1; }
bash "$W/tools/stage_lane.sh" api-setup > apisetup.log 2>&1 || { tail -3 apisetup.log; echo API-SETUP-FAILED; finish; exit 1; }
hub "rm -f ~/$API/runtime/prefix_index.spi"

stamp "hub configs (.prev-$UTC)"
hub "cp -p ~/sparkpipe-swap/model_swap.json ~/sparkpipe-swap/model_swap.json.prev-$UTC && cp -p ~/sparkpipe-chat/config.json ~/sparkpipe-chat/config.json.prev-$UTC && cp -p ~/sparkpipe-chat/litellm.yaml ~/sparkpipe-chat/litellm.yaml.prev-$UTC"
scp -q rtx5090:sparkpipe-swap/model_swap.json swap_live.json && scp -q rtx5090:sparkpipe-chat/config.json chat_live.json || { echo FETCH-FAILED; finish; exit 1; }
python3 - "$SWAP" "$M.swap_entry.json" "$WARM" "$PORT" <<'PY' || { echo CONFIG-EDIT-FAILED; finish; exit 1; }
import json, sys
swap_id, entry_path, warm, port = sys.argv[1:5]
entry = json.load(open(entry_path))
entry["id"] = swap_id
entry["node_warm"] = warm
swap = json.load(open("swap_live.json"))
index = [model["id"] for model in swap["models"]].index(swap_id)
swap["models"][index] = entry
open("swap_new.json", "w").write(json.dumps(swap, indent=1) + "\n")
chat = json.load(open("chat_live.json"))
model = [model for model in chat["models"] if model["id"] == swap_id][0]
model["engine"] = f"http://127.0.0.1:{port}"
open("chat_new.json", "w").write(json.dumps(chat, indent=1, ensure_ascii=False) + "\n")
print("swap entry", json.dumps(entry)[:160], "...; chat engine", model["engine"], "context", model["context_tokens"])
PY
scp -q swap_new.json rtx5090:sparkpipe-swap/model_swap.json && scp -q chat_new.json rtx5090:sparkpipe-chat/config.json || { echo PUSH-FAILED; revert; finish; exit 1; }
restart_services

stamp "port build"
if ! activate "$SWAP"; then echo "PORT-ACTIVATE-FAILED"; revert; finish; exit 1; fi
hub "grep swap_memory ~/sparkpipe-swap/swap.log | tail -1"
{
  hub "pid=\$(systemctl --user show $UNIT -p MainPID --value); sha256sum /proc/\$pid/exe ~/$API/runtime/lib/*.so $APIB/sparkpipe_model_api $APIB/lib${M}_tp16_serving_adapter.so | cut -c1-16,65-; curl -s localhost:$PORT/health; echo"
  fan "cd $ROOT && echo \$(sha256sum lib/lib${M}_tp16_serving_adapter.so bin/sparkpipe_model_residentd bin/weightd_warm | cut -c1-16 | tr '\n' ' ') warm=\$(grep -c WARM warm.log) resident=\$(grep -a -c STAGE-RESIDENT residentd.log) \$(tail -c 300 warm.log | tr '\n' ' ' | grep -o -i 'refused\|failed' | head -1)"
} > identity_new.txt
echo "identity: $(grep -o '"connected_ranks":[0-9]*' identity_new.txt) fw adapter $(fw_sha lib${M}_tp16_serving_adapter.so); node variants: $(grep '^spark' identity_new.txt | cut -d' ' -f2- | sed 's/warm=[0-9]*/warm=N/' | sort | uniq -c | sed 's/^ *//' | tr '\n' ';')"
grep '^spark' identity_new.txt | awk '{print $5, $6}' | sort | uniq -c | tr '\n' ';'; echo
hub "cd ~/rollover/tree && mkdir -p $Q && timeout 2400 $GOLDCMD $Q/golden-new.json > /dev/null 2>&1; python3 scripts/rollover_golden_compare.py $GOLDREF $Q/golden-new.json 0.5 $Q/golden-compare-new.json"
rc=$?
[ "$TAG" = flash ] && hub "cd ~/rollover/tree && python3 tools/glm53_flash_golden_compare.py $GOLDREF $Q/golden-new.json 2>&1 | tail -4"
GATE=""
if [ $rc = 2 ]; then
  echo "GOLDEN rc=2: quality gate decides (COMPSEC >= legacy - 1, smokes >= legacy $SMOKES_MIN/6)"; GATE=1
elif [ $rc != 0 ]; then echo "GOLDEN-FAILED rc=$rc"; revert; finish; exit 1; fi
stamp "smokes"
hub "cd ~/rollover/tree && timeout 3000 python3 tools/ui_smoke.py --endpoint http://127.0.0.1:4000 --model $SWAP --reasoning both --out $Q/ui-smoke.json" > smoke.log 2>&1
tail -8 smoke.log | cut -c1-200
compsec new
old_score=$(grep -o 'RESULT: [0-9]*' compsec_old.log 2>/dev/null | grep -o '[0-9]*$')
new_score=$(grep -o 'RESULT: [0-9]*' compsec_new.log | grep -o '[0-9]*$')
smokes=$(grep -o "UI-SMOKE $SWAP: [0-9]*" smoke.log | grep -o '[0-9]*$')
echo "quality: COMPSEC legacy ${old_score:-none} -> port ${new_score:-none}; smokes ${smokes:-none}/6 (legacy $SMOKES_MIN/6)"
if [ -n "$GATE" ] && { [ "${new_score:-0}" -lt $(( ${old_score:-17} - 1 )) ] || [ "${smokes:-0}" -lt "$SMOKES_MIN" ]; }; then
  echo "QUALITY-GATE-FAILED"; revert; finish; exit 1
fi
fan "grep -a -i -E 'error|fail|fault|abort' $ROOT/residentd.log | tail -2" > ranklogs.txt 2>&1
echo "rank log error lines: $(grep -c . ranklogs.txt)"
finish
