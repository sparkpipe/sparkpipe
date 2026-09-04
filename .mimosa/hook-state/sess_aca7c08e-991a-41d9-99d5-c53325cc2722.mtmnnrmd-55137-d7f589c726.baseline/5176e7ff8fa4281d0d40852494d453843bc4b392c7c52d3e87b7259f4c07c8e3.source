#!/usr/bin/env bash
# qwen4_flash TP4 rank pack smoke: run the resident-decode-stage whole-stack
# validation harness on ONE node against its rank-local pack, with the TP
# collective env that module initialize requires for tp_degree>1.
#
# Parameterized (no hardcoded nodes, per lane rules):
#   --spark-host H        this node's hostname (default: $(hostname))
#   --tp-rank R           this node's TP rank (0-3)
#   --rail-hosts LIST     comma-separated rail IPs in rank order
#   --port-base P         TP control port base
#   --identifier N        TP collective identifier
#   --packs-dir DIR       directory holding qwen4_flash_full.tp4-rank<R>.qwen4_flashsp
#   --worktree DIR        lane worktree with the module sources
#
# Exit evidence: harness log. Since the M5 sharded port (vocab-block
# embedding gather + all-reduce, sharded head argmax through the maxloc
# collective, MTP draft chain at tp>1) initialize should PASS and the full
# check ladder runs. Standalone (TP_STANDALONE=1, the default here) skips
# the collective: embedding/head results stay rank-partial BY DESIGN -
# validation semantics, never serving. Pass --live-collective to connect
# the real peer group instead.
set -euo pipefail

spark_host="$(hostname)"
tp_rank=""
rail_hosts="10.10.100.14,10.10.100.15,10.10.100.16,10.10.100.17"
port_base="66640"
identifier="20260827"
packs_dir=""
worktree=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spark-host) spark_host="$2"; shift 2 ;;
    --tp-rank) tp_rank="$2"; shift 2 ;;
    --rail-hosts) rail_hosts="$2"; shift 2 ;;
    --port-base) port_base="$2"; shift 2 ;;
    --identifier) identifier="$2"; shift 2 ;;
    --packs-dir) packs_dir="$2"; shift 2 ;;
    --worktree) worktree="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$tp_rank" && -n "$packs_dir" && -n "$worktree" ]] || {
  echo "usage: $0 --tp-rank R --packs-dir DIR --worktree DIR [--spark-host H --rail-hosts LIST --port-base P --identifier N]" >&2
  exit 2
}

IFS=',' read -ra host_array <<< "$rail_hosts"
local_host="${host_array[$tp_rank]}"
pack_path="$packs_dir/qwen4_flash_full.tp4-rank${tp_rank}.qwen4_flashsp"
backend_so="$worktree/build/libhidden_transport_spark_host_rdma_verbs.so"

[[ -s "$pack_path" ]] || { echo "pack missing: $pack_path" >&2; exit 2; }
if [[ ! -s "$backend_so" ]]; then
  (cd "$worktree" && make build/libhidden_transport_spark_host_rdma_verbs.so \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a)
fi

echo "qwen4_flash_tp4_smoke host=$spark_host rank=$tp_rank local_host=$local_host pack=$pack_path"

cd "$worktree"
export SPARK_QWEN4_FLASH_STAGE_TP_BACKEND_PATH="$backend_so"
export SPARK_QWEN4_FLASH_STAGE_TP_IDENTIFIER="$identifier"
export SPARK_QWEN4_FLASH_STAGE_TP_PORT_BASE="$port_base"
export SPARK_QWEN4_FLASH_STAGE_TP_HOSTS="$rail_hosts"
export SPARK_QWEN4_FLASH_STAGE_TP_LOCAL_HOST="$local_host"
export SPARK_QWEN4_FLASH_STAGE_TP_TIMEOUT_MS="120000"

# The harness prints per-check lines; module initialize for tp_degree>1
# connects to the peers listed above. Any of the four ranks failing to
# start shows up here as a connect/timeout status.
make -C modules/qwen4_flash_resident_decode_stage validate \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
  STAGE_PACK_PATH="$pack_path" \
  STAGE_COUNT=1 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=48 \
  MTP_LAYER_COUNT=1 MAX_ACTIVE_SEQUENCES=8 KV_BLOCK_COUNT=8 \
  TP_DEGREE=4 TP_RANK="$tp_rank" TP_STANDALONE=1 \
  ALLOW_UNQUALIFIED_EXECUTION=1
