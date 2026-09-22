#!/usr/bin/env bash
# qwen38max_multidev_run_family.sh — the lane-2 qwen38_max family wrapper for
# the shared developer lanes (TP16 over spark0..sparkf), following the GLM
# pattern in docs/MULTIDEV_QUICKSTART.md and the shared contract extracted
# in tools/devcycle/templates/run-family-job.sh.template (PR #1084). Runs
# INSIDE one per-node queue job on the synced checkout (never nested ssh,
# never a private weightd on a multi-node job). It validates the queue
# contract, prepares a private deployment under $SPARK_QUEUE_RUNTIME_ROOT,
# stages exactly one pack digest sidecar for the shared weightd attach,
# then execs the common residentd:
#
#   SPARK_WEIGHTD_ATTACH=1
#   SPARK_WEIGHTD_SOCKET=$shared  (the operator-tracked fleet daemon)
#   SPARK_WEIGHTD_LANE=2          (one of the eight common mesh lanes)
#   SPARK_TP_MESH_RANKS=0,...,15  (identity: logical rank = --nodes index)
#   exec sparkpipe_model_residentd --deployment \
#        "$SPARK_QUEUE_RUNTIME_ROOT/deployment.json" \
#        --rank-index "$SPARK_QUEUE_RANK"
#
# The queue cmd that invokes this file stays DOLLAR-FREE (systemd expands
# $-syntax in cmd files — the lane-6 M1 finding): all resolution logic
# lives here, in the family script.
#
# Required environment
#   SPARK_QUEUE_ATTEMPT        queue attempt id (32 hex chars)
#   SPARK_QUEUE_RUNTIME_ROOT   private per-attempt runtime root
#                              (/tmp/sparkqueue-$SPARK_QUEUE_ATTEMPT)
#   SPARK_QUEUE_PORTS          queue-reserved port ranges (FIRST:LAST,...)
#   SPARK_QUEUE_RANK           index in --nodes (0..15)
#   SPARK_QUEUE_SIZE           must be 16
#   QMAX_WEIGHTD_SOCKET        shared weightd socket path (supervised
#                              daemon; this wrapper never starts one).
#                              Defaults to the fleet shared unit:
#                              /run/sparkpipe-weightd-shared/weightd.sock
#   QMAX_EXPERT_POOL_BYTES     bounded routed-expert pool (arena-side
#                              sizing: model-families/qwen38_max/
#                              smoke_experts.json — 1,501 experts x
#                              28,311,576 B / 16 ranks)
#   QMAX_SPINE_BUDGET_BYTES    bounded full-resolution spine budget
#                              (arena-side: 150,721,556,224 B / 16 ranks)
# Optional environment
#   QMAX_PREBUILT_DIR          reuse built artifacts instead of compiling
#                              (from a prior queue build job on this node)
#   QMAX_KV_BACKING_BYTES      KV backing cap for the private root
#                              (default 8 GiB, must be finite)
#   QMAX_WORKING_SET           when non-empty: materialize the smoke-expert
#                              .wset from the committed manifest and warm
#                              it through the shared socket before launch
#                              (milestone-3 cold-launch path, < 5 s goal)
set -euo pipefail

# ----------------------------- FAMILY PARAMETERS -----------------------------

FAMILY="qwen38_max"
LANE=2
WORLD=16
MESH_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
EXPERT_CODEC="fp8"
# The module's serving pin (modules/qwen38_max_resident_decode_stage/Makefile
# QWEN38_MODEL_REVISION); the adapter refuses a drifted build.
MODEL_REVISION="d2dc35658bcf77e66643428cb52e774cc3b5bd29"
CONTRACT="model_contracts/qwen38_authoritative.json"

# Lane port math (tools/devcycle/lane_assignments.json): lane L owns
#   control 23000+16L..+15, collective 53000+16L..+15 (u16-valid, #1094),
#   transport 64000+16L..+15
CONTROL_BASE=$((23000 + 16 * LANE))
COLLECTIVE_BASE=$((53000 + 16 * LANE))
TRANSPORT_BASE=$((64000 + 16 * LANE))
# The collective block is reserved for this family's admission-time TCP
# collective shape if one ever binds (the module's embedded transport rides
# the weightd mesh today — see the port ledger in
# tools/qwen38max_multidev_lane.py). No extra session block: the device
# collective rides the weightd mesh under the shared socket.

fail() { echo "qwen38max-$FAMILY-lane$LANE: $*" >&2; exit 1; }

# ------------------------------ QUEUE CONTRACT -------------------------------

ATTEMPT="${SPARK_QUEUE_ATTEMPT:?run through the authoritative spark queue}"
case "$ATTEMPT" in
  *[!0-9a-f]*|""|?????????????????????????????????*) fail "bad attempt id" ;;
esac
[ "${#ATTEMPT}" -eq 32 ] || fail "bad attempt id length"

ROOT="${SPARK_QUEUE_RUNTIME_ROOT:?missing queue runtime root}"
[ "$ROOT" = "/tmp/sparkqueue-$ATTEMPT" ] ||
  fail "unexpected job namespace: $ROOT"

[ "${SPARK_QUEUE_SIZE:-}" = "$WORLD" ] ||
  fail "SPARK_QUEUE_SIZE must be $WORLD (got '${SPARK_QUEUE_SIZE:-}')"
RANK="${SPARK_QUEUE_RANK:?SPARK_QUEUE_RANK is required}"
[ "$RANK" -ge 0 ] && [ "$RANK" -lt "$WORLD" ] ||
  fail "SPARK_QUEUE_RANK must be 0..$((WORLD - 1))"

reserved_ok() {  # every emitted listener must fall inside a queue-reserved range
  local port="$1" entry first last
  [ -n "${SPARK_QUEUE_PORTS:-}" ] || fail "queue reserved no ports"
  for entry in ${SPARK_QUEUE_PORTS//,/ }; do
    first="${entry%%:*}"; last="${entry##*:}"
    if [ "$port" -ge "$first" ] && [ "$port" -le "$last" ]; then return 0; fi
  done
  fail "listener port $port is not inside a queue-reserved range"
}

for r in $(seq 0 $((WORLD - 1))); do
  reserved_ok $((CONTROL_BASE + r))
  reserved_ok $((COLLECTIVE_BASE + r))
  reserved_ok $((TRANSPORT_BASE + r))
done

# ------------------------------- WEIGHTD MODE --------------------------------

# shared-socket only: the lane-2 charter runs smoke and small B* under the
# shared lanes; a private daemon is never started by this wrapper.
SOCKET="${QMAX_WEIGHTD_SOCKET:-${SPARK_WEIGHTD_SOCKET:-/run/sparkpipe-weightd-shared/weightd.sock}}"
[ -S "$SOCKET" ] || fail "shared weightd socket $SOCKET is not a live socket; \
the operator must establish the shared daemon (never start one by hand)"

# Arena-side budget defaults derive from the committed census manifest
# (PR #1085) - per-node tp-sharded bytes; the QMAX_* envs only override.
# This keeps the queue cmd BARE (no env prefixes: systemd pre-expansion).
MANIFEST_JSON="$CHECKOUT/model-families/qwen38_max/smoke_experts.json"
[ -f "$MANIFEST_JSON" ] ||
  fail "smoke_experts.json missing (the census manifest is the sizing record)"
BUDGETS="$(python3 "$CHECKOUT/tools/qwen38max_multidev_lane.py" --budgets "$MANIFEST_JSON")"
DEFAULT_POOL="${BUDGETS%% *}"
DEFAULT_SPINE="${BUDGETS##* }"
: "${QMAX_EXPERT_POOL_BYTES:=$DEFAULT_POOL}"
: "${QMAX_SPINE_BUDGET_BYTES:=$DEFAULT_SPINE}"
for value in QMAX_EXPERT_POOL_BYTES QMAX_SPINE_BUDGET_BYTES QMAX_KV_BACKING_BYTES; do
  eval "text=\${$value:-}"
  # shellcheck disable=SC2154  # assigned by the eval above
  case "$text" in
    '')
      [ "$value" = "QMAX_KV_BACKING_BYTES" ] || fail "$value is required"
      ;;
    *[!0-9]*) fail "$value must be a positive decimal byte count" ;;
    *) [ "$text" -gt 0 ] || fail "$value must be positive" ;;
  esac
done

HOST="spark$(printf '%x' "$RANK")"
[ "$(hostname)" = "$HOST" ] ||
  fail "node order mismatch: rank $RANK expects $HOST but this job runs on \
$(hostname); --nodes order must stay identical to the mesh map ($MESH_RANKS)"

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOYED_PACK="/home/$HOST/sparkdata/qwenmax.nvfp4.tp16/packs/qwenmax.nvfp4.tp16.rank$RANK.sp"
[ -f "$DEPLOYED_PACK" ] || fail "deployed rank pack missing: $DEPLOYED_PACK \
(operator-placed set; grep the fleet pack inventory before any warm read)"

# --------------------------- PRIVATE RUNTIME PREP ----------------------------

mkdir -p "$ROOT/bin" "$ROOT/lib" "$ROOT/config" "$ROOT/packs" "$ROOT/kvcache"

# 1. Coherent artifacts: build the exact synced source (GPU job: the module
#    archive and driver need nvcc), or reuse a prior build on this node.
if [ -n "${QMAX_PREBUILT_DIR:-}" ]; then
  for artifact in sparkpipe_model_residentd model_serving_adapter.so \
      model_driver.so hidden_transport.so weightd_warm; do
    [ -f "$QMAX_PREBUILT_DIR/$artifact" ] ||
      fail "QMAX_PREBUILT_DIR missing artifact: $QMAX_PREBUILT_DIR/$artifact"
  done
  install -m 0755 "$QMAX_PREBUILT_DIR/sparkpipe_model_residentd" "$ROOT/bin/"
  install -m 0755 "$QMAX_PREBUILT_DIR/weightd_warm" "$ROOT/bin/"
  install -m 0644 "$QMAX_PREBUILT_DIR/model_serving_adapter.so" "$ROOT/lib/"
  install -m 0644 "$QMAX_PREBUILT_DIR/model_driver.so" "$ROOT/lib/"
  install -m 0644 "$QMAX_PREBUILT_DIR/hidden_transport.so" "$ROOT/lib/"
else
  PATH="/usr/local/cuda/bin:$PATH"
  export PATH
  CONTRACT_SHA256="$(sha256sum "$CHECKOUT/$CONTRACT" | awk '{print $1}')"
  make -C "$CHECKOUT" -j1 \
    build/sparkpipe_model_residentd \
    build/sparkpipe_model_compile \
    build/weightd_warm \
    hidden_transport_spark_host_rdma_verbs
  make -C "$CHECKOUT/modules/qwen38_max_resident_decode_stage" -j1 \
    CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
    EXPERT_CODEC="$EXPERT_CODEC" \
    MODEL_REVISION="$MODEL_REVISION" \
    CONTRACT_SHA256="$CONTRACT_SHA256" \
    archive adapter
  ADAPTER="$CHECKOUT/build/modules/qwen38_max_resident_decode_stage/$EXPERT_CODEC/libqwen38_max_serving_adapter_$EXPERT_CODEC.so"
  [ -f "$ADAPTER" ] || fail "adapter not built: $ADAPTER"
  rm -rf "$ROOT/driver"
  "$CHECKOUT/build/sparkpipe_model_compile" \
    --model "$CHECKOUT/examples/model_descriptions/qwen38_max_resident_decode_stage_firmware.json" \
    --stage qwen38_max_resident_decode_stage \
    --library "$CHECKOUT/build/module_library" \
    --output "$ROOT/driver" \
    --cc /usr/bin/cc \
    --include "$CHECKOUT/include" \
    --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
    --cc-arg -lcuda \
    --cc-arg -lcudart \
    --cc-arg -lstdc++ \
    --cc-arg -lm \
    --cc-arg -ldl \
    --cc-arg -pthread
  install -m 0755 "$CHECKOUT/build/sparkpipe_model_residentd" "$ROOT/bin/"
  install -m 0755 "$CHECKOUT/build/weightd_warm" "$ROOT/bin/"
  install -m 0644 "$ADAPTER" "$ROOT/lib/model_serving_adapter.so"
  install -m 0644 "$ROOT/driver/model_driver.so" "$ROOT/lib/model_driver.so"
  install -m 0644 \
    "$CHECKOUT/build/libhidden_transport_spark_host_rdma_verbs.so" \
    "$ROOT/lib/hidden_transport.so"
fi

# 2. Private deployment + this rank's stage configuration (the adapter's
#    EXACT member set; see tools/qwen38max_multidev_lane.py).
GENERATED="$ROOT/generate.$$"
mkdir -p "$GENERATED"
# shellcheck disable=SC2086  # empty when QMAX_KV_BACKING_BYTES is unset
python3 "$CHECKOUT/tools/qwen38max_multidev_lane.py" \
  --runtime-root "$ROOT" \
  --weightd-socket "$SOCKET" \
  --output-dir "$GENERATED" \
  --rank "$RANK" \
  ${QMAX_KV_BACKING_BYTES:+--kv-backing-bytes "$QMAX_KV_BACKING_BYTES"}
install -m 0644 "$GENERATED/deployment.json" "$ROOT/deployment.json"
install -m 0644 "$GENERATED/adapter.json" "$ROOT/config/adapter.json"
rm -rf "$GENERATED"

# 3. Shared pack bytes through private symlinks, exactly one digest
#    sidecar (model_residentd refuses anything else), and the routed
#    expert manifest the lazy attach fails closed without.
PRIVATE_PACK="$ROOT/packs/$(basename "$DEPLOYED_PACK")"
ln -sfn "$DEPLOYED_PACK" "$PRIVATE_PACK"
if [ -f "$DEPLOYED_PACK.experts" ]; then
  ln -sfn "$DEPLOYED_PACK.experts" "$PRIVATE_PACK.experts"
else
  bash "$CHECKOUT/tools/qwen38max_multidev_experts_manifest.sh" "$PRIVATE_PACK"
fi
CACHED_DIGEST="$DEPLOYED_PACK.sha256"
if [ -r "$CACHED_DIGEST" ] && [ "$(wc -c < "$CACHED_DIGEST")" -ge 65 ] && \
   grep -Eq '^[0-9a-f]{64}([ \t].*)?$' "$CACHED_DIGEST"; then
  head -c 64 "$CACHED_DIGEST" > "$ROOT/packs/pack.sha256"
else
  sha256sum "$PRIVATE_PACK" | awk '{print $1}' > "$ROOT/packs/pack.sha256"
  # Identity reconciliation (NVMe-only): when the operator-placed pack
  # carries its emission receipt, the freshly computed digest must match
  # the receipt's output_sha256 (drift = not the verified emission -
  # fail closed rather than attach unverified bytes).
  if [ -r "$DEPLOYED_PACK.receipt.json" ]; then
    RECEIPT_SHA="$(python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("output_sha256") or "")' \
      "$DEPLOYED_PACK.receipt.json")"
    [ "$RECEIPT_SHA" = "$(cat "$ROOT/packs/pack.sha256")" ] ||
      fail "pack digest disagrees with its emission receipt: $DEPLOYED_PACK"
  fi
  # Best-effort cache beside the deployed pack for later attempts.
  ( set -c; umask 022
    head -c 64 "$ROOT/packs/pack.sha256" > "$CACHED_DIGEST.tmp.$$" 2>/dev/null \
      && mv "$CACHED_DIGEST.tmp.$$" "$CACHED_DIGEST" 2>/dev/null ) || true
fi
[ "$(find "$ROOT/packs" -maxdepth 1 -name '*.sha256' | wc -l)" = 1 ] ||
  fail "packs/ must contain exactly one .sha256 sidecar"

# --------------------- COLD-LAUNCH PRELOAD (milestone 3) ---------------------

if [ -n "${QMAX_WORKING_SET:-}" ]; then
  WSET="$ROOT/smoke.wset"
  python3 "$CHECKOUT/tools/qwen38max_multidev_lane.py" --emit-wset "$WSET"
  REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_revision"])' "$ROOT/config/adapter.json")"
  "$ROOT/bin/weightd_warm" "$SOCKET" "$PRIVATE_PACK" \
    "$(cat "$ROOT/packs/pack.sha256")" "$REVISION" "$WORLD" \
    --wset "$WSET" 300 > "$ROOT/warm.log" 2>&1
  grep -q "WSET-WARM keys=" "$ROOT/warm.log" ||
    fail "working set warm failed (see $ROOT/warm.log)"
fi

# ----------------------------- RESIDENT LAUNCH -------------------------------

export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_SOCKET="$SOCKET"
export SPARK_WEIGHTD_LANE="$LANE"
export SPARK_TP_MESH_RANKS="$MESH_RANKS"
export SPARK_WEIGHTD_EXPERT_POOL_BYTES="$QMAX_EXPERT_POOL_BYTES"
export SPARK_WEIGHTD_SPINE_BUDGET_BYTES="$QMAX_SPINE_BUDGET_BYTES"
# Pinned CUDA environment for shared-lane smoke (template hard rule).
export CUDA_MODULE_LOADING=LAZY
export CUDA_MODULE_DATA_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=32
echo "qwen38max-$FAMILY-lane$LANE: rank=$RANK host=$HOST \
pack=$PRIVATE_PACK pool=${QMAX_EXPERT_POOL_BYTES}B spine=${QMAX_SPINE_BUDGET_BYTES}B \
socket=$SOCKET"
exec "$ROOT/bin/sparkpipe_model_residentd" \
  --deployment "$ROOT/deployment.json" \
  --rank-index "$RANK"
