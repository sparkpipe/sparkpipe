#!/usr/bin/env bash
set -u
. "$(dirname "$0")/rollover_lib.sh"
. "$W/scripts/rollover_env_k3.sh"
SWAP=kimi-k3
PORT=8459
UNIT=kv-k3-api
Q="qualification/k3/rollover-$V"
ROOTREL=k3-lanes/lane8/root
LABELS=$(python3 -c "import sys; sys.path.insert(0, '$W/tools'); import stage_lane; print(' '.join(sorted(({m['kv_label'] for m in stage_lane.MODELS.values()} | {'k3_stage', 'glm52_stage', 'glm5_next_stage'}) - {'qwen38_max_stage'})))")
L="$D/k3"
mkdir -p "$L"
cd "$L"
UTC=$(date -u +%Y%m%dT%H%M%SZ)
echo "$UTC" > backup_utc

identity() {
  {
    hub "pid=\$(systemctl --user show $UNIT -p MainPID --value); sha256sum /proc/\$pid/exe ~/k3-kv-api/bin/sparkpipe_model_api ~/k3-kv-api/runtime/lib/*.so ~/k3-kv-api/runtime/tokenizer/tokenizer.compiled | cut -c1-16,65-; curl -s localhost:$PORT/health; echo"
    fan "cd ~/$ROOTREL && echo \$(sha256sum lib/libk3_serving_adapter.so bin/sparkpipe_model_residentd bin/weightd_warm lib/hidden_transport.so | cut -c1-16 | tr '\n' ' ') \$(ls packs | sed 's/rank[0-9]*/rankN/' | tr '\n' ' ') \$(grep -c WARM warm.log 2>/dev/null)"
  } > "identity_$1.txt"
  echo "identity $1: $(grep -o '"connected_ranks":[0-9]*' "identity_$1.txt") node variants: $(grep '^spark' "identity_$1.txt" | cut -d' ' -f2- | sort | uniq -c | sed 's/^ *//' | tr '\n' ';')"
}

capture() {
  hub "cd ~/rollover/tree && mkdir -p $Q && timeout 1800 python3 scripts/rollover_k3_golden.py $PORT $Q/golden-$1.json && python3 scripts/rollover_golden_compare.py ~/rollover/old/golden_t2fff.json $Q/golden-$1.json 0.5 $Q/golden-compare-$1.json"
}

compsec() {
  hub "cd ~/rollover/tree && mkdir -p $Q && rm -rf $Q/compsec17-$1 && timeout 7200 python3 tools/compsec17_openai.py --endpoint http://127.0.0.1:4000 --model $SWAP --reasoning-effort minimal --out $Q/compsec17-$1" > "compsec_$1.log" 2>&1
  echo "compsec $1 rc=$?: $(grep -o 'RESULT: [0-9]*/[0-9]*' "compsec_$1.log") fails: $(grep -v ' PASS ' "compsec_$1.log" | grep -o 'compsec-[0-9]*' | tr '\n' ' ')"
}

hygiene() {
  fan "W=\$HOME/$ROOTREL/bin/weightd_warm; \$W /tmp/spark_weightd.sock --drop-kv $LABELS > /tmp/rollover_dropkv.log 2>&1; a=\$?; \$W /tmp/spark_weightd.sock --reclaim > /tmp/rollover_reclaim.log 2>&1; b=\$?; echo dropkv=\$a reclaim=\$b avail=\$(awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo)" > "hygiene_$1.txt"
  echo "hygiene $1: $(cut -d' ' -f2,3 "hygiene_$1.txt" | sort | uniq -c | tr '\n' ';') min avail $(grep -o 'avail=[0-9.]*' "hygiene_$1.txt" | cut -d= -f2 | sort -n | head -1) GiB (+ qwen3.8-max ~90.2 when it stops)"
}

rollback() {
  stamp "ROLLBACK $SWAP to root.prev-$UTC"
  activate qwen3.8-max
  fan "cd ~/k3-lanes/lane8 && rm -rf root.failed-$UTC && mv root root.failed-$UTC && cp -a root.prev-$UTC root && echo ok" | grep -v ': ok$'
  hub "rm -rf ~/k3-kv-api.failed-$UTC && mv ~/k3-kv-api ~/k3-kv-api.failed-$UTC && cp -a ~/k3-kv-api.prev-$UTC ~/k3-kv-api && rm -f ~/k3-kv-api/runtime/prefix_index.spi && echo hub restored"
  hygiene rollback
  activate "$SWAP" && identity rolledback && capture rolledback
}

finish() {
  restore_max
  fan "cd ~/$ROOTREL && ./bin/weightd_warm /tmp/spark_weightd.sock --reclaim-pack \$(cut -c1-64 packs/pack.sha256) > /dev/null 2>&1; ./bin/weightd_warm /tmp/spark_weightd.sock --drop-kv k3_stage > /dev/null 2>&1; echo ok" | grep -v ': ok$'
  nodes_up || exit 9
  stamp "done $SWAP ($(controller))"
}

stamp "$SWAP precheck"
[ "$(controller)" = "idle qwen3.8-max None" ] || { echo "PRECHECK-FAILED: $(controller)"; exit 1; }
nodes_up || exit 9
hygiene pre-old

if [ -z "${ROLLOVER_SKIP_OLD:-}" ]; then
  stamp "old build"
  if activate "$SWAP"; then
    identity old
    capture old
    compsec old
  else
    echo "OLD-ACTIVATE-FAILED (no old baseline)"
  fi
fi

stamp "park on qwen3.8-max, backup root.prev-$UTC"
restore_max || exit 1
fan "cd ~/k3-lanes/lane8 && mkdir root.prev-$UTC && rsync -a '--exclude=/kvsnapshot/*' '--exclude=/kvcache/*' root/ root.prev-$UTC/ && echo ok" | grep -v ': ok$'
hub "cp -a ~/k3-kv-api ~/k3-kv-api.prev-$UTC && echo hub backup ok"

stamp "setup + api"
bash "$W/tools/k3_lane.sh" setup > setup.log 2>&1 || { tail -3 setup.log; echo SETUP-FAILED; rollback; finish; exit 1; }
R="$L/render-api"
rm -rf "$R"
python3 "$W/tools/k3_multidev_lane.py" --lane 8 --sequences "$K3_SEQUENCES" --kv-pages "$K3_KV_PAGES" --rows "$K3_ROWS" \
  --kv-physical-pages "$K3_KV_PHYSICAL_PAGES" --kv-logical-pages "$K3_KV_LOGICAL_PAGES" --kv-backing-bytes "$K3_KV_BACKING_BYTES" \
  --topology tp16 --pipeline-transport host-rdma --kv-snapshot-bytes 8589934592 --runtime-root "/home/{host}/k3-lanes/lane8/root" \
  --state-budget-bytes "$K3_STATE_BUDGET_BYTES" --weightd-socket /tmp/spark_weightd.sock --tokenizer-sha256 "$K3_TOKENIZER_SHA256" --output-dir "$R" > /dev/null || { echo RENDER-FAILED; rollback; finish; exit 1; }
scp -q "$R/deployment.json" rtx5090:k3-kv-api/model_resident.json.new || { echo API-UPDATE-FAILED; rollback; finish; exit 1; }
hub "set -e; cd ~/k3-kv-api; B=$K3_API_BUILD; rm -f runtime/prefix_index.spi; install -m 755 \$B/sparkpipe_model_api bin/sparkpipe_model_api.new && mv bin/sparkpipe_model_api.new bin/sparkpipe_model_api; install -m 755 \$B/sparkpipe_tokenize_prompt bin/sparkpipe_tokenize_prompt.new && mv bin/sparkpipe_tokenize_prompt.new bin/sparkpipe_tokenize_prompt; install -m 755 \$B/libk3_tp16_serving_adapter.so runtime/lib/libk3_serving_adapter.so.new && mv runtime/lib/libk3_serving_adapter.so.new runtime/lib/libk3_serving_adapter.so; mv model_resident.json.new model_resident.json; echo $V > SOURCE_COMMIT; cmp -s model_resident.json ~/k3-kv-api.prev-$UTC/model_resident.json && echo api deployment unchanged" || { echo API-UPDATE-FAILED; rollback; finish; exit 1; }
hygiene pre-new

stamp "new build"
if ! activate "$SWAP"; then
  echo "NEW-ACTIVATE-FAILED"; rollback; finish; exit 1
fi
identity new
echo "fw: adapter $(fw_sha libk3_tp16_serving_adapter.so) residentd $(fw_sha sparkpipe_model_residentd) weightd_warm $(fw_sha weightd_warm) transport $(fw_sha libhidden_transport_spark_host_rdma_verbs.so); hub build: $(hub "sha256sum $APIB/sparkpipe_model_api $APIB/libk3_tp16_serving_adapter.so | cut -c1-16 | tr '\n' ' '")"
capture new
rc=$?
if [ $rc = 2 ] && [ -n "${ROLLOVER_QUALITY_GATE:-}" ]; then
  echo "GOLDEN rc=2: flips reported above; quality gate decides (COMPSEC >= ${ROLLOVER_QUALITY_GATE}/17 and smokes 6/6)"
elif [ $rc != 0 ]; then
  echo "GOLDEN-FAILED rc=$rc (2 = large divergence)"; rollback; finish; exit 1
fi
stamp "smokes"
hub "cd ~/rollover/tree && timeout 3000 python3 tools/ui_smoke.py --endpoint http://127.0.0.1:4000 --model $SWAP --reasoning both --out $Q/ui-smoke.json" > smoke.log 2>&1
tail -8 smoke.log | cut -c1-200
compsec new
if [ -n "${ROLLOVER_QUALITY_GATE:-}" ]; then
  score=$(grep -o 'RESULT: [0-9]*' compsec_new.log | grep -o '[0-9]*$')
  if [ "${score:-0}" -lt "$ROLLOVER_QUALITY_GATE" ] || ! grep -q "UI-SMOKE $SWAP: 6/6" smoke.log; then
    echo "QUALITY-GATE-FAILED (COMPSEC ${score:-none}, $(grep UI-SMOKE smoke.log))"; rollback; finish; exit 1
  fi
  echo "QUALITY-GATE-PASSED (COMPSEC $score, $(grep UI-SMOKE smoke.log))"
fi
identity after
fan "f=\$(ls -t ~/$ROOTREL/logs/*.log | head -1); grep -a -i -E 'error|fail|fault|abort' \$f | tail -2" > ranklogs.txt 2>&1
echo "rank log error lines: $(grep -c . ranklogs.txt)"
finish
