#!/usr/bin/env bash
set -u
. /Users/mac/sparkpipe-coord/lanes/driver-sprint/post-reset/pr_lib.sh
hold_lock "$@"
UTC=$(date -u +%Y%m%dT%H%M%SZ); H=perf/holdbw-$UTC; ST=/private/tmp/claude-502/-Users-mac-sparkpipe/df54bdc1-84bf-4584-9653-5211cf17258e/scratchpad/batchsweep/holdbw-$UTC; mkdir -p $ST
exec > >(tee -a $ST/hold.log) 2>&1
stamp "batch sweep hold BW $UTC (B1 o256 x3, 8 streams x512 x2, GLM Flash also 16 streams x512 x2; all-streams-decoding window metric 9e5d21d60; K3 without draft_server, lookup on; GLM production) precheck $(controller)"
nodes_up || exit 9
scp -q /Users/mac/wf/long-ctx/tools/api_serving_perf.py rtx5090:perf/api_serving_perf_window.py || exit 1
hub "mkdir -p ~/$H; sha256sum ~/perf/api_serving_perf_window.py | cut -c1-16"
counters() { hub "grep -a engine_measurements /srv/workspace/sparkpipe/logs/$1 | tail -1 | grep -o '\"completed\":[0-9]*\|\"verify_lanes\":[0-9]*\|\"draft_tokens\":[0-9]*\|\"accepted_draft_tokens\":[0-9]*' | tr '\n' ' '"; }
finish() {
  trap - EXIT
  hub "pkill -f '[a]pi_serving_perf_window.py' ; true"
  hub "cp -p ~/k3-kv-api/model_resident.json.prevbw-$UTC ~/k3-kv-api/model_resident.json && grep -c draft_server ~/k3-kv-api/model_resident.json"
  activate glm-5.3 || true
  activate kimi-k3 || activate kimi-k3
  stamp "batch sweep hold BW end ($(controller)) -> $ST"
}
trap finish EXIT
BOOT0=$(bootids); T0=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
hub "cd ~/k3-kv-api && cp -p model_resident.json model_resident.json.prevbw-$UTC && python3 -c \"
import json
d = json.load(open('model_resident.json'))
d.pop('draft_server')
json.dump(d, open('model_resident.json.tmp', 'w'), indent=2)
\" && mv model_resident.json.tmp model_resident.json && echo draft_server removed for the hold"
for spec in kimi-k3:8459:k3-kv-api.log:streams8x512 glm-5.3:8475:glm53_full-kv-api.log:streams8x512 glm-5.3-flash:8476:glm53_flash-kv-api.log:streams8x512,streams16x512; do
  m=${spec%%:*}; rest=${spec#*:}; port=${rest%%:*}; rest=${rest#*:}; log=${rest%%:*}; streams=${rest#*:}
  [ "$(bootids)" = "$BOOT0" ] || { echo "NODE REBOOTED before $m: abort"; exit 9; }
  activate $m || activate $m || { echo "ACTIVATE-FAILED $m"; exit 3; }
  echo "COUNTERS-BEFORE $m $(counters $log)"
  hub "cd ~/perf && timeout 900 python3 api_serving_perf_window.py --endpoint http://127.0.0.1:$port --cases o32 --repeats 1 --label warmup --output ~/$H/$m.jsonl > /dev/null 2>&1; timeout 900 python3 api_serving_perf_window.py --endpoint http://127.0.0.1:$port --cases o256 --repeats 3 --label bw --output ~/$H/$m.jsonl 2>&1 | tail -1" | sed "s/^/B1 $m /"
  for s in ${streams//,/ } ${streams//,/ }; do
    hub "cd ~/perf && timeout 1800 python3 api_serving_perf_window.py --endpoint http://127.0.0.1:$port --cases $s --label bw --output ~/$H/$m.jsonl 2>&1 | tail -1" | sed "s/^/$s $m /"
  done
  echo "COUNTERS-AFTER $m $(counters $log)"
done
scp -q "rtx5090:$H/*.jsonl" $ST/
echo "AER $(fan "journalctl -k --since '$T0' --no-pager 2>/dev/null | grep -ciE 'AER|CmpltTO|mlx5.*(error|fatal)|SMMU'" | awk '$2 != 0' | tr '\n' ' ')"
[ "$(bootids)" = "$BOOT0" ] || echo "NODE REBOOTED"
