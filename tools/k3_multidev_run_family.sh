#!/usr/bin/env bash
# k3_multidev_run_family.sh — the lane-3 k3 family wrapper for the shared
# developer lanes (TP4xPP4 over spark0..sparkf), following the GLM pattern
# in docs/MULTIDEV_QUICKSTART.md and the shared contract extracted in
# tools/devcycle/templates/run-family-job.sh.template (PR #1084). Runs
# INSIDE one per-node queue job on the synced checkout (never nested ssh,
# never a private weightd on a multi-node job). It validates the queue
# contract, prepares a private deployment under $SPARK_QUEUE_RUNTIME_ROOT,
# stages exactly one pack digest sidecar for the shared weightd attach,
# then execs the common residentd:
#
#   SPARK_WEIGHTD_ATTACH=1
#   SPARK_WEIGHTD_SOCKET=$shared  (the operator-tracked fleet daemon)
#   SPARK_WEIGHTD_LANE=3          (one of the eight common mesh lanes)
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
#   SPARK_QUEUE_RANK           index in --nodes (0..15)
#   SPARK_QUEUE_SIZE           must be 16
#   K3_WEIGHTD_SOCKET          shared weightd socket path (supervised
#                              daemon; this wrapper never starts one).
#                              Defaults to the fleet shared unit:
#                              /run/sparkpipe-weightd-shared/weightd.sock
#   K3_EXPERT_POOL_BYTES       bounded routed-expert pool (calculator:
#                              tools/devcycle/lane_budget_calc.py)
#   K3_SPINE_BUDGET_BYTES      bounded full-resolution spine budget
# Optional environment
#   K3_PREBUILT_DIR            reuse built artifacts instead of compiling
#                              (from a prior queue build job on this node)
#   K3_KV_BACKING_BYTES        KV backing cap for the private root
#                              (default 8 GiB, must be finite)
set -euo pipefail

# ----------------------------- FAMILY PARAMETERS -----------------------------

FAMILY="k3"
LANE=3
WORLD=16
MESH_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

# Lane port math (tools/devcycle/lane_assignments.json): lane L owns
#   control 23000+16L..+15, collective 53000+16L..+15 (u16-valid, #1094),
#   transport 64000+16L..+15
CONTROL_BASE=$((23000 + 16 * LANE))
COLLECTIVE_BASE=$((53000 + 16 * LANE))
TRANSPORT_BASE=$((64000 + 16 * LANE))
# No extra session block is reserved: under the shared socket the k3 device
# collective rides the weightd mesh and its session table (packed into
# 53052..53063 inside the collective block) binds nothing — topology-only
# values documented in tools/k3_multidev_lane.py. Re-plan if a
# verbs-qualified binding path ever lands (see the M1 port finding).

fail() { echo "k3-$FAMILY-lane$LANE: $*" >&2; exit 1; }

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

# shared-socket only: the k3 lane charter runs smoke and small B* under the
# shared lanes; a private daemon is never started by this wrapper.
SOCKET="${K3_WEIGHTD_SOCKET:-${SPARK_WEIGHTD_SOCKET:-/run/sparkpipe-weightd-shared/weightd.sock}}"
[ -S "$SOCKET" ] || fail "shared weightd socket $SOCKET is not a live socket; \
the operator must establish the shared daemon (never start one by hand)"

for value in K3_EXPERT_POOL_BYTES K3_SPINE_BUDGET_BYTES K3_KV_BACKING_BYTES; do
  eval "text=\${$value:-}"
  # shellcheck disable=SC2154  # assigned by the eval above
  case "$text" in
    '')
      [ "$value" = "K3_KV_BACKING_BYTES" ] || fail "$value is required"
      ;;
    *[!0-9]*) fail "$value must be a positive decimal byte count" ;;
    *) [ "$text" -gt 0 ] || fail "$value must be positive" ;;
  esac
done

STAGE=$((RANK / 4))
TP_RANK=$((RANK % 4))
HOST="spark$(printf '%x' "$RANK")"
[ "$(hostname)" = "$HOST" ] ||
  fail "node order mismatch: rank $RANK expects $HOST but this job runs on \
$(hostname); --nodes order must stay identical to the mesh map ($MESH_RANKS)"

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOYED_PACK="/home/$HOST/sparkdata/k3.mxfp4.tp4pp4/packs/k3.stage${STAGE}.rank0${TP_RANK}.pack"
[ -f "$DEPLOYED_PACK" ] || fail "deployed rank pack missing: $DEPLOYED_PACK"

# --------------------------- PRIVATE RUNTIME PREP ----------------------------

mkdir -p "$ROOT/bin" "$ROOT/lib" "$ROOT/config" "$ROOT/packs" "$ROOT/kvcache"

# 1. Coherent artifacts: build the exact synced source (GPU job: the
#    serving adapter needs nvcc), or reuse a prior build on this node.
if [ -n "${K3_PREBUILT_DIR:-}" ]; then
  for artifact in sparkpipe_model_residentd libk3_serving_adapter.so \
      libhidden_transport_spark_host_rdma_verbs.so weightd_warm; do
    [ -f "$K3_PREBUILT_DIR/$artifact" ] ||
      fail "K3_PREBUILT_DIR missing artifact: $K3_PREBUILT_DIR/$artifact"
  done
  install -m 0755 "$K3_PREBUILT_DIR/sparkpipe_model_residentd" "$ROOT/bin/"
  install -m 0755 "$K3_PREBUILT_DIR/weightd_warm" "$ROOT/bin/"
  install -m 0644 "$K3_PREBUILT_DIR/libk3_serving_adapter.so" "$ROOT/lib/"
  install -m 0644 "$K3_PREBUILT_DIR/libhidden_transport_spark_host_rdma_verbs.so" \
    "$ROOT/lib/hidden_transport.so"
else
  PATH="/usr/local/cuda/bin:$PATH"
  export PATH
  make -C "$CHECKOUT" -j1 \
    build/sparkpipe_model_residentd \
    build/libk3_serving_adapter.so \
    build/libhidden_transport_spark_host_rdma_verbs.so \
    build/weightd_warm
  install -m 0755 "$CHECKOUT/build/sparkpipe_model_residentd" "$ROOT/bin/"
  install -m 0755 "$CHECKOUT/build/weightd_warm" "$ROOT/bin/"
  install -m 0644 "$CHECKOUT/build/libk3_serving_adapter.so" "$ROOT/lib/"
  install -m 0644 \
    "$CHECKOUT/build/libhidden_transport_spark_host_rdma_verbs.so" \
    "$ROOT/lib/hidden_transport.so"
fi

# 2. Private deployment + this rank's adapter configuration.
GENERATED="$ROOT/generate.$$"
mkdir -p "$GENERATED"
# shellcheck disable=SC2086  # empty when K3_KV_BACKING_BYTES is unset
python3 "$CHECKOUT/tools/k3_multidev_lane.py" \
  --runtime-root "$ROOT" \
  --weightd-socket "$SOCKET" \
  --output-dir "$GENERATED" \
  --rank "$RANK" \
  ${K3_KV_BACKING_BYTES:+--kv-backing-bytes "$K3_KV_BACKING_BYTES"}
install -m 0644 "$GENERATED/deployment.json" "$ROOT/deployment.json"
install -m 0644 "$GENERATED/adapter.json" "$ROOT/config/adapter.json"
rm -rf "$GENERATED"

# 3. Shared pack bytes through private symlinks, exactly one digest
#    sidecar (model_residentd refuses anything else), and the routed
#    expert manifest the lazy attach fails closed without.
PRIVATE_PACK="$ROOT/packs/$(basename "$DEPLOYED_PACK")"
ln -sfn "$DEPLOYED_PACK" "$PRIVATE_PACK"
if [ -f "$DEPLOYED_PACK.experts" ] && python3 - "$DEPLOYED_PACK.experts" <<'PYVER'
import struct, sys
with open(sys.argv[1], "rb") as handle:
    magic, version, count, _ = struct.unpack("<IIII", handle.read(16))
raise SystemExit(0 if (magic == 0x58504557 and version == 2 and count > 0) else 1)
PYVER
then
  # only a v2 routed-expert manifest with records is usable by the lazy
  # attach; stale v1 sidecars regenerate below like missing ones
  ln -sfn "$DEPLOYED_PACK.experts" "$PRIVATE_PACK.experts"
else
  bash "$CHECKOUT/tools/k3_multidev_experts_manifest.sh" "$PRIVATE_PACK"
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

# 3b. Optional cold-launch preload (M3): warm the smoke expert set
#     through the SAME shared socket the resident attaches to, before
#     exec. Opt-in via K3_PRELOAD_WSET (default: the committed
#     model-families/k3/smoke-k3-v1.wset when K3_PRELOAD=1). The set is
#     filtered to this rank's PP-stage layers - a rank pack manifest
#     holds only its stage's routed layers, so a full-model working set
#     cannot validate there (weightd_warm fails closed on foreign keys).
if [ -n "${K3_PRELOAD_WSET:-}" ] || [ "${K3_PRELOAD:-0}" = 1 ]; then
  WSET_SOURCE="${K3_PRELOAD_WSET:-$CHECKOUT/model-families/k3/smoke-k3-v1.wset}"
  [ -f "$WSET_SOURCE" ] || fail "preload working set missing: $WSET_SOURCE"
  python3 - "$CHECKOUT" "$WSET_SOURCE" "$STAGE" "$ROOT/preload.wset" <<'PYW'
import struct, sys
sys.path.insert(0, sys.argv[1] + "/tools")
import k3_smoke_experts as producer
data = open(sys.argv[2], "rb").read()
if len(data) == 0 or len(data) % 8 != 0:
    raise SystemExit("k3 preload: malformed working set")
pairs = [struct.unpack_from("<II", data, index)
         for index in range(0, len(data), 8)]
local = producer.stage_pairs(pairs, int(sys.argv[3]))
with open(sys.argv[4], "wb") as out:
    out.write(producer.wset_bytes(local))
print(f"k3 preload: {len(local)} of {len(pairs)} keys are stage "
      f"{sys.argv[3]}-local")
PYW
  SHA_HEX="$(cat "$ROOT/packs/pack.sha256")"
  SPARK_WEIGHTD_EXPERT_POOL_BYTES="$K3_EXPERT_POOL_BYTES" \
    "$ROOT/bin/weightd_warm" "$SOCKET" "$PRIVATE_PACK" "$SHA_HEX" x 16 \
    --family k3 --wset "$ROOT/preload.wset" 300 > "$ROOT/warm.log" 2>&1 ||
    { cat "$ROOT/warm.log" >&2; fail "working set warm failed"; }
  grep -q "WSET-WARM keys=" "$ROOT/warm.log" ||
    fail "working set warm produced no WSET-WARM receipt (see $ROOT/warm.log)"
fi

# ----------------------------- RESIDENT LAUNCH -------------------------------

export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_SOCKET="$SOCKET"
export SPARK_WEIGHTD_LANE="$LANE"
# SPARK_TP_MESH_RANKS is the TP GROUP's physical ranks (degree entries,
# parsed by SparkTpDeviceCollectiveMeshTopology): for TP4xPP4 that is
# this rank's stage slice, NOT the 16-node world identity (the parser
# rejects trailing entries with INVALID_ARGUMENT at mesh lane acquire —
# first-launch find #10). The world map stays in $MESH_RANKS for the
# node-order contract above.
export SPARK_TP_MESH_RANKS="$((STAGE * 4)),$((STAGE * 4 + 1)),$((STAGE * 4 + 2)),$((STAGE * 4 + 3))"
export SPARK_WEIGHTD_EXPERT_POOL_BYTES="$K3_EXPERT_POOL_BYTES"
export SPARK_WEIGHTD_SPINE_BUDGET_BYTES="$K3_SPINE_BUDGET_BYTES"
# Pinned CUDA environment for shared-lane smoke (template hard rule).
export CUDA_MODULE_LOADING=LAZY
export CUDA_MODULE_DATA_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=32
echo "k3-$FAMILY-lane$LANE: rank=$RANK host=$HOST stage=$STAGE tp=$TP_RANK \
pack=$PRIVATE_PACK pool=${K3_EXPERT_POOL_BYTES}B spine=${K3_SPINE_BUDGET_BYTES}B \
socket=$SOCKET"
exec "$ROOT/bin/sparkpipe_model_residentd" \
  --deployment "$ROOT/deployment.json" \
  --rank-index "$RANK"
