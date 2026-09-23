#!/usr/bin/env bash
# ling_multidev_run_family.sh — the lane-9 ling family wrapper for the
# shared developer lanes (TP16 over spark0..sparkf), following the GLM
# pattern in docs/MULTIDEV_QUICKSTART.md and the shared contract extracted
# in tools/devcycle/templates/run-family-job.sh.template (PR #1084),
# mirroring the lane-2 qwen38max TP16 port (#1099) and the lane-3 k3
# sidecar-hygiene rules (#1131). Runs INSIDE one per-node queue job on the
# synced checkout (never nested ssh, never a private weightd on a
# multi-node job). It validates the queue contract, prepares a private
# deployment under $SPARK_QUEUE_RUNTIME_ROOT, stages exactly one pack
# digest sidecar for the shared weightd attach, then execs the common
# residentd:
#
#   SPARK_WEIGHTD_ATTACH=1
#   SPARK_WEIGHTD_SOCKET=$shared  (the operator-tracked fleet daemon)
#   SPARK_WEIGHTD_LANE=9          (lane 9 of the common mesh lanes)
#   SPARK_TP_MESH_RANKS=0,...,15  (identity: logical rank = --nodes index)
#   exec sparkpipe_model_residentd --deployment \
#        "$SPARK_QUEUE_RUNTIME_ROOT/deployment.json" \
#        --rank-index "$SPARK_QUEUE_RANK"
#
# Required environment
#   SPARK_QUEUE_ATTEMPT        queue attempt id (32 hex chars)
#   SPARK_QUEUE_RUNTIME_ROOT   private per-attempt runtime root
#                              (/tmp/sparkqueue-$SPARK_QUEUE_ATTEMPT)
#   SPARK_QUEUE_PORTS          queue-reserved port ranges (FIRST:LAST,...)
#                              MUST include 23144:23159 23744:23807
#                              53144:53159 64144:64159
#   SPARK_QUEUE_RANK           index in --nodes (0..15)
#   SPARK_QUEUE_SIZE           must be 16
#   LING_EXPERT_POOL_BYTES     bounded routed-expert pool (calculator:
#                              tools/devcycle/lane_budget_calc.py against
#                              model-families/ling/smoke_experts.json)
#   LING_SPINE_BUDGET_BYTES    bounded full-resolution spine budget;
#                              optional - when unset it is derived from
#                              the placed pack's .experts manifest (the
#                              exact allocation the lazy attach checks,
#                              plus margin), and when set it must cover
#                              that manifest-derived floor
# Optional environment
#   LING_EXPERT_CODEC          placed arm: bf16 (default) | fp8
#   LING_WEIGHTD_SOCKET        shared weightd socket path (supervised
#                              daemon; this wrapper never starts one).
#                              Defaults to the fleet shared unit:
#                              /run/sparkpipe-weightd-shared/weightd.sock
#   LING_PREBUILT_DIR          reuse built artifacts instead of compiling
#                              (from a prior queue build job on this node)
#   LING_KV_BACKING_BYTES      KV backing cap for the private root
#                              (default 8 GiB, must be finite)
#   LING_WORKING_SET           cold-launch preload hook (M3): if set,
#                              materialize the committed smoke-expert .wset
#                              and warm it through the SAME socket before
#                              the resident attaches.
#
# Queue cmd files stay dollar-free (systemd expands $-syntax; lane-6 M1
# finding): this wrapper reads its knobs from the environment above, and
# the queue cmd file only sets plain KEY=VALUE assignments.
set -euo pipefail

# ----------------------------- FAMILY PARAMETERS -----------------------------

FAMILY="ling"
LANE=9
WORLD=16
MESH_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
EXPERT_CODEC="${LING_EXPERT_CODEC:-bf16}"
ARM="ling.$EXPERT_CODEC.tp16"
# The module's serving pin (Makefile LING_MODEL_REVISION); the adapter
# refuses a drifted build.
MODEL_REVISION="e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3"
CONTRACT="model_contracts/ling_authoritative.json"
FIRMWARE="examples/model_descriptions/ling_resident_decode_stage_firmware.json"
MODULE="modules/ling_resident_decode_stage"

# Lane port math (tools/devcycle/lane_assignments.json): lane 9 owns
#   control 23144-23159, collective 53144-23159 (u16-valid, #1094),
#   transport 64144-64159, session 23744-23807 (the 23168+64L block).
CONTROL_BASE=$((23000 + 16 * LANE))
COLLECTIVE_BASE=$((53000 + 16 * LANE))
TRANSPORT_BASE=$((64000 + 16 * LANE))
SESSION_BASE=$((23168 + 64 * LANE))
# The session block carries the stage config's topology-only session
# matrices (rows 23744..23759 + hc rows 23760..23775; k3 lane-3 M1
# finding: under the shared socket ApplyTopology records them and nothing
# binds). The wrapper requires the WHOLE 64-number block reserved so any
# future binding path is already lane-scoped.

fail() { echo "ling-$FAMILY-lane$LANE: $*" >&2; exit 1; }

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

case "$EXPERT_CODEC" in
  bf16|fp8) ;;
  *) fail "LING_EXPERT_CODEC must be bf16 or fp8 (placed arms), got '$EXPERT_CODEC'" ;;
esac

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
for p in $(seq "$SESSION_BASE" $((SESSION_BASE + 31))); do
  reserved_ok "$p"
done

# ------------------------------- WEIGHTD MODE --------------------------------

# shared-socket only: the ling lane charter runs smoke and small B* under
# the shared lanes; a private daemon is never started by this wrapper.
SOCKET="${LING_WEIGHTD_SOCKET:-${SPARK_WEIGHTD_SOCKET:-/run/sparkpipe-weightd-shared/weightd.sock}}"
[ -S "$SOCKET" ] || fail "shared weightd socket $SOCKET is not a live socket; \
the operator must establish the shared daemon (never start one by hand)"

for value in LING_EXPERT_POOL_BYTES LING_SPINE_BUDGET_BYTES LING_KV_BACKING_BYTES; do
  eval "text=\${$value:-}"
  # shellcheck disable=SC2154  # assigned by the eval above
  case "$text" in
    '')
      case "$value" in
        LING_EXPERT_POOL_BYTES) fail "$value is required" ;;
      esac
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
# ling rank packs are HEX-named (rankb, not rank11): the stagepack
# canonical rule names the file after the spark node letter.
DEPLOYED_PACK="/home/$HOST/sparkdata/$ARM/packs/$ARM.rank$(printf '%x' "$RANK").sp"
[ -f "$DEPLOYED_PACK" ] || fail "deployed rank pack missing: $DEPLOYED_PACK \
(operator-placed set; grep the fleet pack inventory before any warm read)"

MANIFEST_SPINE="$(python3 "$CHECKOUT/tools/ling_multidev_lane.py" \
  --spine-budget "$DEPLOYED_PACK")" ||
  fail "cannot derive the spine budget from $DEPLOYED_PACK.experts"
if [ -n "${LING_SPINE_BUDGET_BYTES:-}" ]; then
  [ "$LING_SPINE_BUDGET_BYTES" -ge "$MANIFEST_SPINE" ] ||
    fail "LING_SPINE_BUDGET_BYTES=$LING_SPINE_BUDGET_BYTES is below the \
manifest-derived floor $MANIFEST_SPINE (allocation + margin); the lazy \
attach checks (allocation + 255) and fails with capacity_exceeded"
else
  LING_SPINE_BUDGET_BYTES="$MANIFEST_SPINE"
fi
export LING_SPINE_BUDGET_BYTES

# --------------------------- PRIVATE RUNTIME PREP ----------------------------

mkdir -p "$ROOT/bin" "$ROOT/lib" "$ROOT/config" "$ROOT/packs" "$ROOT/kvcache"

# 1. Coherent artifacts: build the exact synced source (GPU job: the module
#    archive and driver need nvcc), or reuse a prior build on this node.
if [ -n "${LING_PREBUILT_DIR:-}" ]; then
  for artifact in sparkpipe_model_residentd model_serving_adapter.so \
      model_driver.so hidden_transport.so weightd_warm; do
    [ -f "$LING_PREBUILT_DIR/$artifact" ] ||
      fail "LING_PREBUILT_DIR missing artifact: $LING_PREBUILT_DIR/$artifact"
  done
  install -m 0755 "$LING_PREBUILT_DIR/sparkpipe_model_residentd" "$ROOT/bin/"
  install -m 0755 "$LING_PREBUILT_DIR/weightd_warm" "$ROOT/bin/"
  install -m 0644 "$LING_PREBUILT_DIR/model_serving_adapter.so" "$ROOT/lib/"
  install -m 0644 "$LING_PREBUILT_DIR/model_driver.so" "$ROOT/lib/"
  install -m 0644 "$LING_PREBUILT_DIR/hidden_transport.so" "$ROOT/lib/"
else
  PATH="/usr/local/cuda/bin:$PATH"
  export PATH
  CONTRACT_SHA256="$(sha256sum "$CHECKOUT/$CONTRACT" | awk '{print $1}')"
  make -C "$CHECKOUT" -j1 \
    build/sparkpipe_model_residentd \
    build/sparkpipe_model_compile \
    build/weightd_warm \
    hidden_transport_spark_host_rdma_verbs
  make -C "$CHECKOUT/$MODULE" -j1 \
    CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
    EXPERT_CODEC="$EXPERT_CODEC" \
    MODEL_REVISION="$MODEL_REVISION" \
    CONTRACT_SHA256="$CONTRACT_SHA256" \
    archive adapter
  ADAPTER="$CHECKOUT/build/modules/ling_resident_decode_stage/$EXPERT_CODEC/libling_serving_adapter_$EXPERT_CODEC.so"
  [ -f "$ADAPTER" ] || fail "adapter not built: $ADAPTER"
  rm -rf "$ROOT/driver"
  "$CHECKOUT/build/sparkpipe_model_compile" \
    --model "$CHECKOUT/$FIRMWARE" \
    --stage ling_resident_decode_stage \
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

# 2. Private deployment + this rank's adapter configuration (the adapter's
#    EXACT member set; see tools/ling_multidev_lane.py).
GENERATED="$ROOT/generate.$$"
mkdir -p "$GENERATED"
# shellcheck disable=SC2086  # empty when LING_KV_BACKING_BYTES is unset
python3 "$CHECKOUT/tools/ling_multidev_lane.py" \
  --runtime-root "$ROOT" \
  --weightd-socket "$SOCKET" \
  --output-dir "$GENERATED" \
  --rank "$RANK" \
  --codec "$EXPERT_CODEC" \
  ${LING_KV_BACKING_BYTES:+--kv-backing-bytes "$LING_KV_BACKING_BYTES"}
install -m 0644 "$GENERATED/deployment.json" "$ROOT/deployment.json"
install -m 0644 "$GENERATED/adapter.json" "$ROOT/config/adapter.json"
rm -rf "$GENERATED"

# 3. Shared pack bytes through private symlinks, exactly one digest
#    sidecar (model_residentd refuses anything else), and the routed
#    expert manifest the lazy attach fails closed without. Existence is
#    not validity (lane-3 stage-3 find): only a v2 manifest with records
#    is acceptable; the fleet set was verified 2026-09-22 and a stale
#    sidecar is a hard fail until the M2 producer regenerates it.
PRIVATE_PACK="$ROOT/packs/$(basename "$DEPLOYED_PACK")"
ln -sfn "$DEPLOYED_PACK" "$PRIVATE_PACK"
if [ -f "$DEPLOYED_PACK.experts" ] && python3 - "$DEPLOYED_PACK.experts" <<'PYVER'
import struct, sys
with open(sys.argv[1], "rb") as handle:
    magic, version, count, _ = struct.unpack("<IIII", handle.read(16))
raise SystemExit(0 if (magic == 0x58504557 and version == 2 and count > 0) else 1)
PYVER
then
  ln -sfn "$DEPLOYED_PACK.experts" "$PRIVATE_PACK.experts"
else
  fail "routed-expert manifest missing or not v2-with-records beside \
$DEPLOYED_PACK - regenerate with the ling manifest producer (M2) before attach"
fi
CACHED_DIGEST="$DEPLOYED_PACK.sha256"
if [ -r "$CACHED_DIGEST" ] && [ "$(wc -c < "$CACHED_DIGEST")" -ge 65 ] && \
   grep -Eq '^[0-9a-f]{64}([ \t].*)?$' "$CACHED_DIGEST"; then
  head -c 64 "$CACHED_DIGEST" > "$ROOT/packs/pack.sha256"
else
  sha256sum "$PRIVATE_PACK" | awk '{print $1}' > "$ROOT/packs/pack.sha256"
  # Best-effort cache beside the deployed pack for later attempts.
  ( set -c; umask 022
    head -c 64 "$ROOT/packs/pack.sha256" > "$CACHED_DIGEST.tmp.$$" 2>/dev/null \
      && mv "$CACHED_DIGEST.tmp.$$" "$CACHED_DIGEST" 2>/dev/null ) || true
fi
[ "$(find "$ROOT/packs" -maxdepth 1 -name '*.sha256' | wc -l)" = 1 ] ||
  fail "packs/ must contain exactly one .sha256 sidecar"

# --------------------- COLD-LAUNCH PRELOAD (milestone 3) ---------------------

if [ -n "${LING_WORKING_SET:-}" ]; then
  WSET="$ROOT/smoke.wset"
  python3 "$CHECKOUT/tools/ling_multidev_lane.py" --emit-wset "$WSET"
  REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_revision"])' "$ROOT/config/adapter.json")"
  # --family ling pins the module tag so the warm keys the SAME arena
  # the residentd attaches to (identity equality; PR #1146). The head
  # exceeds the client's 512-key lease cap on every rank (chunk-union
  # 16: all layers per rank pack), so warm the shard sequence through
  # the same socket/arena.
  python3 "$CHECKOUT/tools/ling_wset_split.py" "$WSET" "$ROOT/smoke" \
    > "$ROOT/smoke.shards"
  : > "$ROOT/warm.log"
  while read -r shard _keys; do
    SPARK_WEIGHTD_EXPERT_POOL_BYTES="$LING_EXPERT_POOL_BYTES" \
      "$ROOT/bin/weightd_warm" "$SOCKET" "$PRIVATE_PACK" \
      "$(cat "$ROOT/packs/pack.sha256")" "$REVISION" "$WORLD" \
      --family ling --wset "$shard" 300 >> "$ROOT/warm.log" 2>&1 ||
      fail "working set warm failed on $shard (see $ROOT/warm.log)"
  done < "$ROOT/smoke.shards"
  [ "$(grep -c "WSET-WARM keys=" "$ROOT/warm.log")" = "$(wc -l < "$ROOT/smoke.shards")" ] ||
    fail "working set warm incomplete (see $ROOT/warm.log)"
fi

# Publish the private runtime root for same-lane follow-on jobs (the
# decode receipt on the coordinator rank needs deployment.json/config):
# a per-attempt pointer plus a stable latest symlink, both under /tmp.
echo "$ROOT" > "/tmp/ling-lane9-root.$ATTEMPT"
ln -sfn "$ROOT" /tmp/ling-lane9-root-latest

# ----------------------------- RESIDENT LAUNCH -------------------------------

export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_SOCKET="$SOCKET"
export SPARK_WEIGHTD_LANE="$LANE"
export SPARK_TP_MESH_RANKS="$MESH_RANKS"
export SPARK_WEIGHTD_EXPERT_POOL_BYTES="$LING_EXPERT_POOL_BYTES"
export SPARK_WEIGHTD_SPINE_BUDGET_BYTES="$LING_SPINE_BUDGET_BYTES"
# Pinned CUDA environment for shared-lane smoke (template hard rule).
export CUDA_MODULE_LOADING=LAZY
export CUDA_MODULE_DATA_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=32
echo "ling-$FAMILY-lane$LANE: rank=$RANK host=$HOST codec=$EXPERT_CODEC \
pack=$PRIVATE_PACK pool=${LING_EXPERT_POOL_BYTES}B spine=${LING_SPINE_BUDGET_BYTES}B \
socket=$SOCKET"
exec "$ROOT/bin/sparkpipe_model_residentd" \
  --deployment "$ROOT/deployment.json" \
  --rank-index "$RANK"
