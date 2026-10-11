#!/usr/bin/env bash
set -u
W="$(cd "$(dirname "$0")/.." && pwd)"
CUR=${1:?current config dir}
OUT=$(mktemp -d)
for m in qwen38_27b qwen38_flash qwen38_max dsv41_flash; do
  ( . "$W/scripts/rollover_env_$m.sh"; bash "$W/tools/stage_lane.sh" render "$OUT/$m" > /dev/null )
  echo "$m deployment: $(diff -q <(python3 -m json.tool --sort-keys "$CUR/$m/deployment.json") <(python3 -m json.tool --sort-keys "$OUT/$m/deployment.json") > /dev/null && echo IDENTICAL || echo DIFFERS)  adapter.spark1: $(diff -q <(python3 -m json.tool --sort-keys "$CUR/$m/adapter.spark1.json") <(python3 -m json.tool --sort-keys "$OUT/$m/adapter.spark1.json") > /dev/null && echo IDENTICAL || echo DIFFERS)"
done
( . "$W/scripts/rollover_env_k3.sh"; bash "$W/tools/k3_lane.sh" render "$OUT/k3" > /dev/null
  python3 "$W/tools/k3_multidev_lane.py" --lane 8 --sequences $K3_SEQUENCES --kv-pages $K3_KV_PAGES --rows $K3_ROWS --kv-physical-pages $K3_KV_PHYSICAL_PAGES --kv-logical-pages $K3_KV_LOGICAL_PAGES --kv-backing-bytes $K3_KV_BACKING_BYTES --topology tp16 --pipeline-transport host-rdma --kv-snapshot-bytes 8589934592 --runtime-root "/home/{host}/k3-lanes/lane8/root" --state-budget-bytes $K3_STATE_BUDGET_BYTES --weightd-socket /tmp/spark_weightd.sock --tokenizer-sha256 $K3_TOKENIZER_SHA256 --output-dir "$OUT/k3api" > /dev/null )
for p in "deployment.json k3/deployment.json" "adapter.spark1.json k3/adapter.spark1.json" "api_model_resident.json k3api/deployment.json"; do set -- $p
  echo "k3 $1: $(diff -q <(python3 -m json.tool --sort-keys "$CUR/k3/$1") <(python3 -m json.tool --sort-keys "$OUT/$2") > /dev/null && echo IDENTICAL || { echo DIFFERS; diff <(python3 -m json.tool --sort-keys "$CUR/k3/$1") <(python3 -m json.tool --sort-keys "$OUT/$2") | head -10; })"
done
rm -rf "$OUT"
