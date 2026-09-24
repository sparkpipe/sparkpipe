#!/usr/bin/env bash
# qwen38max_multidev_build_artifacts.sh — build the lane-2 wrapper's
# PREBUILT artifact set on THIS node (multidev M3 build arm).
#
# Stages (single literal argument; queue cmd stays BARE):
#   compile  root targets + module archive/adapter, install the four
#            static artifacts. CPU resources suffice (nvcc needs no GPU
#            device) — budget host memory generously.
#   publish  module publish (whole-stack smoke GPU validation on this
#            node's placed pack) + driver compile, then add the driver
#            artifact to the existing set. GPU-owned; needs the compile
#            stage's build tree in the SAME checkout directory.
#   all      both, one unit (fits only with host memory >= ~24 GiB after
#            the device carve-out — build-r6 OOM-killed at 7 GiB host).
#
# Produces the coherent artifact set the wrapper's QMAX_PREBUILT_DIR
# consumes, in a persistent node-local directory:
#
#   /home/<host>/sparkdata/qwen38max.tp16/build-latest/
#     sparkpipe_model_residentd  model_serving_adapter.so
#     model_driver.so            hidden_transport.so   weightd_warm
#
# Same build chain as the wrapper's inline path (the family's proven
# qwen38_tp4_build.sh flow): root targets, module archive+adapter with
# the pinned serving revision, module publish (whole-stack smoke tier on
# this node's OWN placed pack), then sparkpipe_model_compile for the
# driver — the driver compile resolves the module from
# build/module_library, so the publish MUST precede it. Atomic: builds
# into build-partial.$$ and renames on success.
set -euo pipefail

WORLD=16
EXPERT_CODEC="fp8"
MODEL_REVISION="d2dc35658bcf77e66643428cb52e774cc3b5bd29"
CONTRACT="model_contracts/qwen38_authoritative.json"
OUT_REL="sparkdata/qwen38max.tp16/build-latest"

fail() { echo "qwen38max-build-artifacts: $*" >&2; exit 1; }

STAGE="${1:-all}"
case "$STAGE" in compile|publish|all) ;; *) fail "usage: $0 [compile|publish|all]" ;; esac

ATTEMPT="${SPARK_QUEUE_ATTEMPT:?run through the authoritative spark queue}"
case "$ATTEMPT" in
  *[!0-9a-f]*|""|?????????????????????????????????*) fail "bad attempt id" ;;
esac
[ "${#ATTEMPT}" -eq 32 ] || fail "bad attempt id length"
# Single-node targeted form or the 16-node --per-node fleet form (each
# node builds its own artifacts).
case "${SPARK_QUEUE_SIZE:-1}" in 1|16) ;; *) fail "size must be 1 or 16" ;; esac

HOST="$(hostname)"
case "$HOST" in spark[0-9a-f]) ;; *) fail "unexpected hostname '$HOST'" ;; esac
# This node's OWN placed pack (one pack per node: rank i lives on
# spark{hex(i)} — tools/qwen38max_multidev_pack_emit.sh placement). Pack
# names carry the HEX rank suffix (qwenmax.nvfp4.tp16.ranka..rankf for
# ranks 10-15 — tools/qwen38max_patch_rank.sh printf %x convention;
# decimal rank10..15 would miss on 6/16 nodes). The publish validates the
# module against the same pack this node's driver will attach.
NODE_RANK="$((16#${HOST#spark}))"
PACK="/home/$HOST/sparkdata/qwenmax.nvfp4.tp16/packs/qwenmax.nvfp4.tp16.rank${HOST#spark}.sp"

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="/home/$HOST/$OUT_REL"
# The partial MUST NOT be spelled through "$OUT/..": the atomic swap
# removes $OUT first, and a path containing a removed component fails
# ENOENT even though the partial exists (build-r7c, all 16 nodes: mv
# "cannot stat" with the partial present — a latent bug every earlier
# round died too early to reach). Same filesystem, no traversal.
PARTIAL="$(dirname "$OUT")/build-partial.$$"
rm -rf "$PARTIAL"
# Sweep stale EMPTY partials from the pre-fix runs (their traps no-oped
# on the unresolvable path); non-empty partials belong to live runs.
# MUST precede our own mkdir or it deletes our still-empty partial
# (build-r8c2, all 16 nodes).
find "$(dirname "$OUT")" -maxdepth 1 -type d -empty \
  -name 'build-partial.*' -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$PARTIAL"
trap 'rm -rf "$PARTIAL"' EXIT

[ "$STAGE" = compile ] || \
  [ -r "$PACK" ] || fail "placed pack for this node missing: $PACK (operator-placed NVMe set; no pack, no publish)"

PATH="/usr/local/cuda/bin:$PATH"
export PATH
CONTRACT_SHA256="$(sha256sum "$CHECKOUT/$CONTRACT" | awk '{print $1}')"
START="$(date +%s)"

if [ "$STAGE" != publish ]; then
  # ---- compile stage (CPU-sufficient; the fat nvcc TUs need host RAM) ----
  make -C "$CHECKOUT" -j4 \
    build/sparkpipe_model_residentd \
    build/sparkpipe_model_compile \
    build/sparkpipe_model_batch \
    build/weightd_warm \
    hidden_transport_spark_host_rdma_verbs
  make -C "$CHECKOUT/modules/qwen38_max_resident_decode_stage" -j4 \
    CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
    EXPERT_CODEC="$EXPERT_CODEC" \
    MODEL_REVISION="$MODEL_REVISION" \
    CONTRACT_SHA256="$CONTRACT_SHA256" \
    archive adapter
  ADAPTER="$CHECKOUT/build/modules/qwen38_max_resident_decode_stage/$EXPERT_CODEC/libqwen38_max_serving_adapter_$EXPERT_CODEC.so"
  [ -f "$ADAPTER" ] || fail "adapter not built: $ADAPTER"
  install -m 0755 "$CHECKOUT/build/sparkpipe_model_residentd" "$PARTIAL/"
  install -m 0755 "$CHECKOUT/build/sparkpipe_model_batch" "$PARTIAL/"
  install -m 0755 "$CHECKOUT/build/weightd_warm" "$PARTIAL/"
  install -m 0644 "$ADAPTER" "$PARTIAL/model_serving_adapter.so"
  install -m 0644 \
    "$CHECKOUT/build/libhidden_transport_spark_host_rdma_verbs.so" \
    "$PARTIAL/hidden_transport.so"
  rm -rf "$OUT" && mv "$PARTIAL" "$OUT"
  END="$(date +%s)"
  for artifact in sparkpipe_model_residentd sparkpipe_model_batch model_serving_adapter.so \
      hidden_transport.so weightd_warm; do
    [ -f "$OUT/$artifact" ] || fail "missing artifact: $OUT/$artifact"
  done
  echo "COMPILE-DONE host=$HOST seconds=$((END - START))"
  [ "$STAGE" = compile ] && exit 0
fi

# ---- publish stage (GPU-owned: retained-receipt whole-stack validation) ----
ADAPTER="$CHECKOUT/build/modules/qwen38_max_resident_decode_stage/$EXPERT_CODEC/libqwen38_max_serving_adapter_$EXPERT_CODEC.so"
[ -f "$ADAPTER" ] || fail "compile stage artifacts missing in this checkout: $ADAPTER (run the compile stage in the SAME cwd first)"
[ -x "$CHECKOUT/build/sparkpipe_model_compile" ] || fail "sparkpipe_model_compile missing: run the compile stage first"
# Publish the validated module into build/module_library: the driver
# compile below resolves the module by exact identity from the library
# and fails with MODULE_NOT_VALIDATED without this. Whole-stack smoke
# tier (STAGE_COUNT=1, all 92 layers, TP1, MAS=8) on this node's own
# placed pack — the validator's admitted single-node tier (qwen38_27b
# publish precedent: tp4-rank0 pack + standalone whole-stack).
#
# The shared runtime refuses standalone full-pack loads by law (build-r9p:
# "stage-module direct pack load refused: weightd attach is mandatory" —
# full-pack loads by residentds kill shared nodes), so the validation
# attaches through THIS node's shared daemon exactly like a residentd:
# socket + attach switch + the pack's sidecar digest, with the pool and
# spine budgets from this rank's chunk-basis numbers (the arena rides the
# daemon's tracked 29,184 MiB, not this unit's device carve-out).
WEIGHTD_SOCKET="${QMAX_WEIGHTD_SOCKET:-/run/sparkpipe-weightd-shared/weightd.sock}"
[ -S "$WEIGHTD_SOCKET" ] || fail "shared weightd socket not live: $WEIGHTD_SOCKET"
# The placed set carries .experts + .receipt.json sidecars (no .sha256
# files); the receipt's output_sha256 is the pack digest (the attach
# identity check on the daemon side fail-closes if it does not match).
[ -r "$PACK.receipt.json" ] || fail "pack receipt missing: $PACK.receipt.json"
PACK_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_sha256"])' "$PACK.receipt.json")"
[ "${#PACK_SHA}" -eq 64 ] || fail "bad output_sha256 in $PACK.receipt.json"
BUDGET_LINE="$(python3 "$CHECKOUT/tools/qwen38max_multidev_lane.py" \
  --budgets "$CHECKOUT/model-families/qwen38_max/smoke_experts.json" \
  --rank "$NODE_RANK" \
  --pack "$PACK")"
POOL_BYTES="${QMAX_EXPERT_POOL_BYTES:-${BUDGET_LINE%% *}}"
SPINE_BYTES="${QMAX_SPINE_BUDGET_BYTES:-${BUDGET_LINE##* }}"
export SPARK_WEIGHTD_SOCKET="$WEIGHTD_SOCKET"
export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_PACK_SHA256="$PACK_SHA"
export SPARK_WEIGHTD_EXPERT_POOL_BYTES="$POOL_BYTES"
export SPARK_WEIGHTD_SPINE_BUDGET_BYTES="$SPINE_BYTES"
# -j1: two concurrent nvcc passes peak ~10 GiB host and OOM a queue
# unit whose host slice is the admission-fit 8 GiB (r22p2: oom-kill at
# 3 s on every node); the fat TU alone fits comfortably.
make -C "$CHECKOUT/modules/qwen38_max_resident_decode_stage" -j1 \
  CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
  EXPERT_CODEC="$EXPERT_CODEC" \
  MODEL_REVISION="$MODEL_REVISION" \
  CONTRACT_SHA256="$CONTRACT_SHA256" \
  STAGE_PACK_PATH="$PACK" \
  STAGE_COUNT=1 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=92 \
  MTP_LAYER_COUNT=0 MAX_ACTIVE_SEQUENCES=8 KV_BLOCK_COUNT=8 \
  ALLOW_UNQUALIFIED_EXECUTION=1 TP_STANDALONE=1 \
  publish
"$CHECKOUT/build/sparkpipe_model_compile" \
  --model "$CHECKOUT/examples/model_descriptions/qwen38_max_resident_decode_stage_firmware.json" \
  --stage qwen38_max_resident_decode_stage \
  --library "$CHECKOUT/build/module_library" \
  --output "$PARTIAL/driver" \
  --cc /usr/bin/cc \
  --include "$CHECKOUT/include" \
  --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
  --cc-arg -lcuda \
  --cc-arg -lcudart \
  --cc-arg -lstdc++ \
  --cc-arg -lm \
  --cc-arg -ldl \
  --cc-arg -pthread
# The compile stage owns the directory atomically; add the driver with a
# same-filesystem atomic rename so a torn install can never be observed.
install -m 0644 "$PARTIAL/driver/model_driver.so" "$OUT/model_driver.so.$$"
mv "$OUT/model_driver.so.$$" "$OUT/model_driver.so"
[ -f "$OUT/sparkpipe_model_residentd" ] || fail "compile-stage set missing at $OUT (run compile first)"
END="$(date +%s)"
echo "qwen38max-build-artifacts: $HOST -> $OUT in $((END - START))s"
for artifact in sparkpipe_model_residentd model_serving_adapter.so \
    model_driver.so hidden_transport.so weightd_warm; do
  [ -f "$OUT/$artifact" ] || fail "missing artifact: $OUT/$artifact"
done
echo "BUILD-DONE host=$HOST seconds=$((END - START))"
