#!/bin/bash
# DSV4 Flash M1 CPU-side build (no GPU steps; queue not required).
# Runs ON spark5 in /home/spark5/lane-dsv4flash-m1/src (export of 879fc5a).
# Produces: host tools, b1 serving adapter, host-rdma transport, the b1
# module archive (variants = compile-only), and the runtime skeleton
# (bin/, lib/ minus model_driver.so, config/ with per-node home paths).
set -euo pipefail
BASE=/home/spark5/lane-dsv4flash-m1
SRC=$BASE/src
RT=$BASE/rt
BIS=/home/spark5/lane-dsv4bisect
STAGE=$BASE/staging

cd "$SRC"
echo "== [1/4] host tools + adapters + transport (CPU)"
make -j20 build/sparkpipe_model_compile build/sparkpipe_model_batch \
    build/sparkpipe_model_residentd build/sparkpipe_module_publish \
    build/libdsv4_tp4_b1_serving_adapter.so \
    build/libhidden_transport_spark_host_rdma_verbs.so

echo "== [2/4] dsv4 module b1 archive (compile-only, no GPU)"
make -C modules/dsv4_resident_decode_stage variants \
    MODULE_BATCH_VARIANT_BUCKETS=1 CUDA_ARCH=sm_121a

echo "== [3/4] assemble runtime skeleton"
rm -rf "$RT"; mkdir -p "$RT/bin" "$RT/lib" "$RT/config" "$RT/packs" "$RT/kv"
cp build/sparkpipe_model_batch build/sparkpipe_model_residentd "$RT/bin/"
cp build/libdsv4_tp4_b1_serving_adapter.so "$RT/lib/model_serving_adapter.so"
cp build/libhidden_transport_spark_host_rdma_verbs.so "$RT/lib/hidden_transport.so"

python3 - "$STAGE/model_resident.o128.json" "$RT/config/model_resident.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
cfg = json.load(open(src))
for node in cfg["nodes"]:
    h = node["transport_host"]
    root = f"/home/{h}/lane-dsv4flash-m1/rt"
    node["runtime_root"] = root
    node["kv_backing_directory"] = f"{root}/kv"
json.dump(cfg, open(dst, "w"), indent=1)
PY
cp "$STAGE/dsv4_flash_tp4_stage.o128.json" "$RT/config/dsv4_flash_tp4_stage.json"

echo "== [4/4] record artifact identities"
{
  echo "source_export: 879fc5a (origin/main 2026-08-30)"
  sha256sum "$RT"/bin/* "$RT"/lib/* "$RT"/config/* \
    build/modules/dsv4_resident_decode_stage/libdsv4_resident_decode_stage_b1.a
} | tee "$BASE/results/m1_cpu_build_receipt.txt"
echo "CPU-BUILD-DONE"
