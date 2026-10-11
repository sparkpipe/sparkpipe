#!/usr/bin/env bash
set -u
. "$(dirname "$0")/rollover_lib.sh"
TAG="$1"
REF3=qualification/qwen38_27b/reference/reference3_fp8_numpy.json
case "$TAG" in
  q27) SWAP=qwen3.8-27b; M=qwen38_27b; OLD=qwen27b_lp_2fff.json; REF=$REF3; EFF="--reasoning-effort minimal"; PARK=qwen3.8-max ;;
  qflash) SWAP=qwen3.8-flash; M=qwen38_flash; OLD=flash_lp_new.json; REF=$REF3; EFF="--reasoning-effort minimal"; PARK=qwen3.8-max ;;
  qmax) SWAP=qwen3.8-max; M=qwen38_max; OLD=qmax_lp_2fff.json; REF=$REF3; EFF="--max-tokens 2048"; PARK=qwen3.8-flash ;;
  d41) SWAP=deepseek-v4.1-flash; M=dsv41_flash; OLD=d41_lp_2fff.json; REF=qualification/dsv41_flash/reference_tokens.json; EFF="--reasoning-effort minimal"; PARK=qwen3.8-max ;;
  *) echo "usage: rollover_stage.sh q27|qflash|qmax|d41"; exit 2 ;;
esac
. "$W/scripts/rollover_env_$M.sh"
PORT=$(python3 -c "import sys; sys.path.insert(0, '$W/tools'); import stage_lane; print(stage_lane.MODELS['$M']['api_port'])")
PARKM=$( [ "$PARK" = qwen3.8-max ] && echo qwen38_max/lane12 || echo qwen38_flash/lane13 )
LANEDIR="\$HOME/stage-lanes/$M/lane$STAGE_LANE"
API="$M-kv-api"
UNIT="kv-${M//_/-}-api"
Q="qualification/$M/rollover-$V"
L="$D/$TAG"
mkdir -p "$L"
cd "$L"
UTC=$(date -u +%Y%m%dT%H%M%SZ)
echo "$UTC" > backup_utc

identity() {
  {
    hub "pid=\$(systemctl --user show $UNIT -p MainPID --value); sha256sum /proc/\$pid/exe ~/$API/bin/sparkpipe_model_api ~/$API/runtime/lib/*.so | cut -c1-16,65-; curl -s localhost:$PORT/health; echo"
    fan "cd $LANEDIR/root && echo \$(sha256sum lib/lib${M}_tp16_serving_adapter.so bin/sparkpipe_model_residentd bin/weightd_warm lib/hidden_transport.so | cut -c1-16 | tr '\n' ' ') \$(ls packs | sed 's/rank[0-9]*/rankN/' | tr '\n' ' ')"
  } > "identity_$1.txt"
  echo "identity $1: $(grep -o '"connected_ranks":[0-9]*' "identity_$1.txt") node variants: $(grep '^spark' "identity_$1.txt" | cut -d' ' -f2- | sort | uniq -c | sed 's/^ *//' | tr '\n' ';')"
}

capture() {
  hub "cd ~/rollover/tree && mkdir -p $Q && timeout 1200 python3 scripts/rollover_lp_capture.py $REF $PORT $Q/golden-$1.json > /dev/null && python3 scripts/rollover_golden_compare.py ~/rollover/old/$OLD $Q/golden-$1.json 0.5 $Q/golden-compare-$1.json"
}

compsec() {
  hub "cd ~/rollover/tree && mkdir -p $Q && rm -rf $Q/compsec17-$1 && timeout 7200 python3 tools/compsec17_openai.py --endpoint http://127.0.0.1:4000 --model $SWAP $EFF --out $Q/compsec17-$1" > "compsec_$1.log" 2>&1
  echo "compsec $1 rc=$?: $(grep -o 'RESULT: [0-9]*/[0-9]*' "compsec_$1.log") fails: $(grep -v ' PASS ' "compsec_$1.log" | grep -o 'compsec-[0-9]*' | tr '\n' ' ')"
}

rollback() {
  stamp "ROLLBACK $SWAP to root.prev-$UTC"
  activate "$PARK"
  fan "cd $LANEDIR && rm -rf root.failed-$UTC && mv root root.failed-$UTC && cp -a root.prev-$UTC root && echo ok" | grep -v ': ok$'
  hub "rm -rf ~/$API.failed-$UTC && mv ~/$API ~/$API.failed-$UTC && cp -a ~/$API.prev-$UTC ~/$API && rm -f ~/$API/runtime/prefix_index.spi && echo hub restored"
  activate "$SWAP" && identity rolledback && capture rolledback
}

finish() {
  if [ "$SWAP" = qwen3.8-max ]; then
    activate qwen3.8-max || restore_max
    fan "\$HOME/stage-lanes/$PARKM/root/lane.sh reclaim > /dev/null 2>&1 && echo ok" | grep -v ': ok$'
  else
    restore_max
    fan "$LANEDIR/root/lane.sh reclaim > /dev/null 2>&1 && echo ok" | grep -v ': ok$'
  fi
  nodes_up || exit 9
  stamp "done $SWAP ($(controller))"
}

stamp "$SWAP precheck"
[ "$(controller)" = "idle qwen3.8-max None" ] || { echo "PRECHECK-FAILED: $(controller)"; exit 1; }
nodes_up || exit 9

if [ -z "${ROLLOVER_SKIP_OLD:-}" ]; then
  stamp "old build"
  activate "$SWAP" || { echo "OLD-ACTIVATE-FAILED"; restore_max; exit 1; }
  identity old
  capture old
  compsec old
fi

stamp "park on $PARK, backup root.prev-$UTC"
activate "$PARK" || { echo "PARK-FAILED"; restore_max; exit 1; }
fan "$LANEDIR/root/lane.sh reclaim > /dev/null 2>&1; cd $LANEDIR && mkdir root.prev-$UTC && rsync -a '--exclude=/kvsnapshot/*' '--exclude=/kvcache/*' root/ root.prev-$UTC/ && echo ok" | grep -v ': ok$'
hub "cp -a ~/$API ~/$API.prev-$UTC && echo hub backup ok"

stamp "setup + api-setup"
bash "$W/tools/stage_lane.sh" setup > setup.log 2>&1 || { tail -3 setup.log; echo SETUP-FAILED; rollback; finish; exit 1; }
bash "$W/tools/stage_lane.sh" api-setup > apisetup.log 2>&1 || { tail -3 apisetup.log; echo API-SETUP-FAILED; rollback; finish; exit 1; }
hub "rm -f ~/$API/runtime/prefix_index.spi"
hub "python3 -c \"import json;d=json.load(open('/home/spec/sparkpipe-swap/model_swap.json'));print(json.dumps([m for m in d['models'] if m['id']=='$SWAP'][0],sort_keys=True))\"" > swap_live.json
python3 -c "import json;a=json.load(open('swap_live.json'));b=json.load(open('$M.swap_entry.json'));print('swap entry vs generated:', 'IDENTICAL' if a==b else 'DIFFERS '+str(sorted(k for k in set(a)|set(b) if a.get(k)!=b.get(k))))"

stamp "new build"
if ! activate "$SWAP"; then
  echo "NEW-ACTIVATE-FAILED"; rollback; finish; exit 1
fi
identity new
echo "fw: adapter $(fw_sha lib${M}_tp16_serving_adapter.so) residentd $(fw_sha sparkpipe_model_residentd) weightd_warm $(fw_sha weightd_warm) transport $(fw_sha libhidden_transport_spark_host_rdma_verbs.so); hub build: $(hub "sha256sum $APIB/sparkpipe_model_api $APIB/lib${M}_tp16_serving_adapter.so | cut -c1-16 | tr '\n' ' '")"
capture new
rc=$?
if [ $rc != 0 ]; then
  echo "GOLDEN-FAILED rc=$rc (2 = large divergence)"; rollback; finish; exit 1
fi
stamp "smokes"
hub "cd ~/rollover/tree && timeout 3000 python3 tools/ui_smoke.py --endpoint http://127.0.0.1:4000 --model $SWAP --reasoning both --out $Q/ui-smoke.json" > smoke.log 2>&1
tail -8 smoke.log | cut -c1-200
compsec new
identity after
fan "f=$LANEDIR/root/residentd.log; grep -a -i -E 'error|fail|fault|abort' \$f | tail -2" > ranklogs.txt 2>&1
echo "rank log error lines: $(grep -c . ranklogs.txt)"
finish
