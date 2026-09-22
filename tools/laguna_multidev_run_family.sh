#!/usr/bin/env bash
# laguna_multidev_run_family.sh — the lane-8 laguna family wrapper for the
# shared developer lanes (TP8xPP2 over spark0..sparkf), following the GLM
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
#   SPARK_WEIGHTD_LANE=8          (one of the common mesh lanes)
#   SPARK_TP_MESH_RANKS=0,...,15  (identity: logical rank = --nodes index)
#   exec sparkpipe_model_residentd --deployment \
#        "$SPARK_QUEUE_RUNTIME_ROOT/deployment.json" \
#        --rank-index "$SPARK_QUEUE_RANK"
#
# The queue cmd that invokes this file stays DOLLAR-FREE (systemd expands
# $-syntax in cmd files - the lane-6 M1 finding): all resolution logic
# lives here, in the family script.
#
# Placed packs (fleet inventory, Sep 17 - prep2/ACC-2, verified on every
# node during the lane-8 M1 inventory): the filename rank is the GLOBAL
# rank, so this host's pack is stage$((RANK/8)).rank$RANK - NOT
# stageN.rank0-on-spark8. The packs carry .sha256 + .receipt.json
# sidecars but NO .experts sidecar: the wrapper generates the v2
# routed-expert manifest from the pack directory (loading fails closed
# without it) and derives the arena-side budgets from it - zero assumed
# byte counts.
#
# Required environment
#   SPARK_QUEUE_ATTEMPT        queue attempt id (32 hex chars)
#   SPARK_QUEUE_RUNTIME_ROOT   private per-attempt runtime root
#                              (/tmp/sparkqueue-$SPARK_QUEUE_ATTEMPT)
#   SPARK_QUEUE_PORTS          queue-reserved port ranges (FIRST:LAST,...)
#   SPARK_QUEUE_RANK           index in --nodes (0..15)
#   SPARK_QUEUE_SIZE           must be 16
#   LAGUNA_WEIGHTD_SOCKET      shared weightd socket path (supervised
#                              daemon; this wrapper never starts one).
#                              Defaults to the fleet shared unit:
#                              /run/sparkpipe-weightd-shared/weightd.sock
#   LAGUNA_EXPERT_POOL_BYTES   bounded routed-expert pool (arena-side
#                              sizing; default derived from this rank
#                              pack's .experts sidecar - the exact expert
#                              span sum)
#   LAGUNA_SPINE_BUDGET_BYTES  bounded full-resolution spine budget
#                              (arena-side; default the sidecar's spine
#                              complement inside the pack file)
# Optional environment
#   LAGUNA_PREBUILT_DIR        reuse built artifacts instead of compiling
#                              (from a prior queue build job on this node)
#   LAGUNA_KV_BACKING_BYTES    KV backing cap for the private root
#                              (default 8 GiB, must be finite)
#   LAGUNA_WORKING_SET         when non-empty: materialize the smoke-expert
#                              .wset and warm it through the shared socket
#                              before launch (milestone-3 cold-launch path)
set -euo pipefail

# ----------------------------- FAMILY PARAMETERS -----------------------------

FAMILY="laguna"
LANE=8
WORLD=16
# The TP8 collective's mesh map is GROUP-SCOPED (the quickstart topology
# table's TP-subset convention): SparkTpDeviceCollectiveMeshTopology
# parses exactly tp_degree entries, so a stage's ranks export their own
# eight physical ranks (attach-010: the 16-entry identity map failed the
# parser's terminator check after the 8th entry). Set after STAGE below."
EXPERT_CODEC="bf16"
# The placed packs' identity (every .receipt.json beside them pins this
# revision); the module build and the adapter serving pin must agree.
MODEL_REVISION="0f573140834b11cfac0c2af97a101a7a69a13e22"
CONTRACT="model_contracts/laguna_authoritative.json"
FIRMWARE="examples/model_descriptions/laguna_resident_decode_stage_firmware.json"

# Lane port math (tools/devcycle/lane_assignments.json): lane L owns
#   control 23000+16L..+15, collective 53000+16L..+15 (u16-valid, #1094),
#   transport 64000+16L..+15; the stage config's session matrix occupies
#   the extra session block 23168+64L..+63 (reserved with --ports; see
#   the port ledger in tools/laguna_multidev_lane.py).
CONTROL_BASE=$((23000 + 16 * LANE))
COLLECTIVE_BASE=$((53000 + 16 * LANE))
TRANSPORT_BASE=$((64000 + 16 * LANE))
SESSION_BASE=$((23168 + 64 * LANE))

fail() { echo "laguna-$FAMILY-lane$LANE: $*" >&2; exit 1; }

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
# The session matrix numbers (topology-only under the shared socket) live
# in the lane's extra session block: reserve and check both edges.
reserved_ok "$SESSION_BASE"
reserved_ok $((SESSION_BASE + 55))

for value in LAGUNA_EXPERT_POOL_BYTES LAGUNA_SPINE_BUDGET_BYTES LAGUNA_KV_BACKING_BYTES; do
  eval "text=\${$value:-}"
  # shellcheck disable=SC2154  # assigned by the eval above
  case "$text" in
    ''|*[!0-9]*) : ;;
    *) [ "$text" -gt 0 ] || fail "$value must be positive" ;;
  esac
done

# ------------------------------- WEIGHTD MODE --------------------------------

# shared-socket only: the lane-8 charter runs smoke and small B* under the
# shared lanes; a private daemon is never started by this wrapper.
SOCKET="${LAGUNA_WEIGHTD_SOCKET:-${SPARK_WEIGHTD_SOCKET:-/run/sparkpipe-weightd-shared/weightd.sock}}"
[ -S "$SOCKET" ] || fail "shared weightd socket $SOCKET is not a live socket; \
the operator must establish the shared daemon (never start one by hand)"

STAGE=$((RANK / 8))
TP_RANK=$((RANK % 8))
MESH_RANKS="$(seq -s, $((STAGE * 8)) $((STAGE * 8 + 7)))"
HOST="spark$(printf '%x' "$RANK")"
[ "$(hostname)" = "$HOST" ] ||
  fail "node order mismatch: rank $RANK expects $HOST but this job runs on \
$(hostname); --nodes order must stay identical to the mesh map ($MESH_RANKS)"

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOYED_PACK="/home/$HOST/sparkdata/laguna-s-2.1.bf16.tp8pp2/packs/laguna_stage.tp8.pp2.stage${STAGE}.rank${RANK}.lgsp"
[ -f "$DEPLOYED_PACK" ] || fail "deployed rank pack missing: $DEPLOYED_PACK \
(operator-placed set; grep the fleet pack inventory before any warm read)"

# --------------------------- PRIVATE RUNTIME PREP ----------------------------

mkdir -p "$ROOT/bin" "$ROOT/lib" "$ROOT/config" "$ROOT/packs" "$ROOT/kvcache"

# 1. Coherent artifacts: build the exact synced source (GPU job: the
#    module archive and driver need nvcc), or reuse a prior build on this
#    node. The chain is the family's proven publish flow: module archive
#    + adapter + module publish (GPU validator, synthetic fixture) so
#    sparkpipe_model_compile resolves the module from
#    build/module_library, then the firmware driver .so.
if [ -n "${LAGUNA_PREBUILT_DIR:-}" ]; then
  for artifact in sparkpipe_model_residentd weightd_warm \
      model_serving_adapter.so model_driver.so hidden_transport.so; do
    [ -f "$LAGUNA_PREBUILT_DIR/$artifact" ] ||
      fail "LAGUNA_PREBUILT_DIR missing artifact: $LAGUNA_PREBUILT_DIR/$artifact"
  done
  install -m 0755 "$LAGUNA_PREBUILT_DIR/sparkpipe_model_residentd" "$ROOT/bin/"
  install -m 0755 "$LAGUNA_PREBUILT_DIR/weightd_warm" "$ROOT/bin/"
  install -m 0644 "$LAGUNA_PREBUILT_DIR/model_serving_adapter.so" "$ROOT/lib/"
  install -m 0644 "$LAGUNA_PREBUILT_DIR/model_driver.so" "$ROOT/lib/"
  install -m 0644 "$LAGUNA_PREBUILT_DIR/hidden_transport.so" "$ROOT/lib/"
else
  PATH="/usr/local/cuda/bin:$PATH"
  export PATH
  CONTRACT_SHA256="$(sha256sum "$CHECKOUT/$CONTRACT" | awk '{print $1}')"
  make -C "$CHECKOUT" -j1 \
    build/sparkpipe_model_residentd \
    build/sparkpipe_model_compile \
    build/weightd_warm \
    hidden_transport_spark_host_rdma_verbs
  make -C "$CHECKOUT/modules/laguna_resident_decode_stage" -j1 \
    CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
    EXPERT_CODEC="$EXPERT_CODEC" \
    MODEL_REVISION="$MODEL_REVISION" \
    CONTRACT_SHA256="$CONTRACT_SHA256" \
    archive adapter publish
  ADAPTER="$CHECKOUT/build/modules/laguna_resident_decode_stage/$EXPERT_CODEC/liblaguna_serving_adapter_$EXPERT_CODEC.so"
  [ -f "$ADAPTER" ] || fail "adapter not built: $ADAPTER"
  rm -rf "$ROOT/driver"
  "$CHECKOUT/build/sparkpipe_model_compile" \
    --model "$CHECKOUT/$FIRMWARE" \
    --stage laguna_resident_decode_stage \
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
#    EXACT member set; see tools/laguna_multidev_lane.py). The wire
#    identifier follows the template derivation from the attempt id.
COLLECTIVE_ID="$(python3 -c "print(int('$ATTEMPT'[:15], 16) + 1 + $LANE)")"
GENERATED="$ROOT/generate.$$"
mkdir -p "$GENERATED"
# shellcheck disable=SC2086  # empty when LAGUNA_KV_BACKING_BYTES is unset
python3 "$CHECKOUT/tools/laguna_multidev_lane.py" \
  --runtime-root "$ROOT" \
  --weightd-socket "$SOCKET" \
  --output-dir "$GENERATED" \
  --rank "$RANK" \
  --collective-identifier "$COLLECTIVE_ID" \
  ${LAGUNA_KV_BACKING_BYTES:+--kv-backing-bytes "$LAGUNA_KV_BACKING_BYTES"}
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
    head = handle.read(16)
    record = handle.read(48)
magic, version, count, _ = struct.unpack("<IIII", head)
ok = magic == 0x58504557 and version == 2 and count > 0 and len(record) == 48
if ok:
    kind = struct.unpack_from("<4I2Q", record)[2]
    ok = kind in (28, 30)   # laguna range-kind convention (tensor_kind*2)
raise SystemExit(0 if ok else 1)
PYVER
then
  # only a v2 routed-expert manifest with records in the laguna kind
  # convention is usable by the lazy attach; stale sidecars (including
  # the k3-style 0/1 kinds of the first attempts) regenerate like
  # missing ones
  ln -sfn "$DEPLOYED_PACK.experts" "$PRIVATE_PACK.experts"
else
  bash "$CHECKOUT/tools/laguna_multidev_experts_manifest.sh" "$PRIVATE_PACK"
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

# 4. Arena-side budgets: default to the exact sidecar-derived numbers for
#    THIS rank pack (whole-pack 2 MiB chunk basis for the pool - the
#    daemon's acquire accounting - and the spine complement); the envs
#    only override (queue cmd stays bare). Exported HERE: the warm leg
#    below runs weightd_warm, which requires the pool env (the lane-8
#    attach-001 finding: the warm hook ran before the launch-block
#    exports and failed with the weightd_warm usage error).
BUDGETS="$(python3 "$CHECKOUT/tools/laguna_multidev_lane.py" --budgets "$PRIVATE_PACK")"
DEFAULT_POOL="${BUDGETS%% *}"
DEFAULT_SPINE="${BUDGETS##* }"
: "${LAGUNA_EXPERT_POOL_BYTES:=$DEFAULT_POOL}"
: "${LAGUNA_SPINE_BUDGET_BYTES:=$DEFAULT_SPINE}"
[ "$LAGUNA_EXPERT_POOL_BYTES" -gt 0 ] && [ "$LAGUNA_SPINE_BUDGET_BYTES" -gt 0 ] ||
  fail "expert-pool/spine budgets must be positive decimal byte counts"
export SPARK_WEIGHTD_EXPERT_POOL_BYTES="$LAGUNA_EXPERT_POOL_BYTES"
export SPARK_WEIGHTD_SPINE_BUDGET_BYTES="$LAGUNA_SPINE_BUDGET_BYTES"

# --------------------- COLD-LAUNCH PRELOAD (milestone 3) ---------------------

if [ -n "${LAGUNA_WORKING_SET:-}" ]; then
  # Per-rank filtered wset (weightd_warm rejects pairs outside this
  # pack's manifest), split into <=512-pair chunks
  # (SPARK_WEIGHTD_LEASE_GROUPS_MAX) warmed sequentially - the GLM
  # pin-experts chunked-lease precedent.
  WSET="$ROOT/smoke.wset"
  python3 "$CHECKOUT/tools/laguna_multidev_lane.py" \
    --emit-wset "$WSET" --rank "$RANK"
  rm -f "$WSET.chunk."*
  split -b 4096 -d "$WSET" "$WSET.chunk."
  : > "$ROOT/warm.log"
  for chunk in $(ls "$WSET.chunk."* | sort); do
    "$ROOT/bin/weightd_warm" "$SOCKET" "$PRIVATE_PACK" \
      "$(cat "$ROOT/packs/pack.sha256")" "$MODEL_REVISION" "$WORLD" \
      --wset "$chunk" 300 >> "$ROOT/warm.log" 2>&1 ||
      fail "working set warm failed (see $ROOT/warm.log, chunk $chunk)"
  done
  grep -q "WSET-WARM keys=" "$ROOT/warm.log" ||
    fail "working set warm failed (see $ROOT/warm.log)"
fi

# ----------------------------- RESIDENT LAUNCH -------------------------------

export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_SOCKET="$SOCKET"
# Deterministic mesh lane: the collective's sixteen ranks must sit in
# ONE lane's band pair and the fleet daemon is PER-NODE, so a
# daemon-assigned lane cannot be uniform across the job's nodes - the
# lane is pinned, never assigned. attach-009's assignment stopgap died
# with the 8-entry table (MESH-LANE-FAIL); the lane-9 constant bump
# (SPARK_WEIGHTD_MESH_MAX_LANES=16) + the manager's fleet restart wave
# make pin 8 acquirable - pinning before that wave fails MESH-LANE-FAIL
# by design, so this launch waits for the wave.
export SPARK_WEIGHTD_LANE="$LANE"
export SPARK_TP_MESH_RANKS="$MESH_RANKS"
# Budget envs were exported at derivation time (the warm leg needs them).
# Pinned CUDA environment for shared-lane smoke (template hard rule).
export CUDA_MODULE_LOADING=LAZY
export CUDA_MODULE_DATA_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=32
echo "laguna-$FAMILY-lane$LANE: rank=$RANK host=$HOST stage=$STAGE tp=$TP_RANK \
pack=$PRIVATE_PACK pool=${LAGUNA_EXPERT_POOL_BYTES}B spine=${LAGUNA_SPINE_BUDGET_BYTES}B \
socket=$SOCKET collective_id=$COLLECTIVE_ID"
if [ -n "${LAGUNA_GDB:-}" ]; then
  # Debug hook (crash triage): run the resident under batch gdb and
  # print the backtrace on fault before the queue reaps the job.
  exec gdb --batch -ex run -ex "bt 25" \
    --args "$ROOT/bin/sparkpipe_model_residentd" \
    --deployment "$ROOT/deployment.json" \
    --rank-index "$RANK"
fi
exec "$ROOT/bin/sparkpipe_model_residentd" \
  --deployment "$ROOT/deployment.json" \
  --rank-index "$RANK"
