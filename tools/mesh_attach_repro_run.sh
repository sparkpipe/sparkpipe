#!/bin/bash
# attach-then-register repro runner (lane 0). Queue cmd:
#   bash tools/mesh_attach_repro_run.sh
set -euo pipefail
CHECKOUT="$(pwd)"
make build/mesh_register_attach_repro
export SPARK_WEIGHTD_EXPERT_POOL_BYTES="${SPARK_WEIGHTD_EXPERT_POOL_BYTES:-25769803776}"
export SPARK_WEIGHTD_SPINE_BUDGET_BYTES="${SPARK_WEIGHTD_SPINE_BUDGET_BYTES:-4294967296}"
HOST="$(hostname)"
exec build/mesh_register_attach_repro /run/sparkpipe-weightd-shared/weightd.sock \
  "/home/$HOST/sparkdata/glm53flash.fp8.tp16/packs/glm53flash.fp8.tp16.rank${HOST#spark}.sp" \
  318dd18ad18dc748181f22687be2a482c53f76d13c23c3056895128fa550ac80 \
  84c6a6aa9497188e15a635ba793b0f95a79b1033 16
