#!/usr/bin/env bash
set -euo pipefail

FAMILY="laguna"
WORLD=16
EXPERT_CODEC="bf16"
MODEL_REVISION="0f573140834b11cfac0c2af97a101a7a69a13e22"
CONTRACT="model_contracts/laguna_authoritative.json"
FIRMWARE="examples/model_descriptions/laguna_resident_decode_stage_bf16_firmware.json"

fail() { echo "$FAMILY-lane${LANE:-?}: $*" >&2; exit 1; }

decimal() {
  case "$1" in
    ''|*[!0-9]*) return 1 ;;
  esac
  return 0
}

LANE="${LAGUNA_LANE:-}"
decimal "$LANE" || fail "LAGUNA_LANE is required: the weightd mesh lane and port lane, 0..15"
[ "$LANE" -lt 16 ] || fail "LAGUNA_LANE must be 0..15"
CONTROL_BASE=$((23000 + 16 * LANE))
COLLECTIVE_BASE=$((53000 + 16 * LANE))
TRANSPORT_BASE=$((64000 + 16 * LANE))
SESSION_BASE=$((23168 + 64 * LANE))

MODE="${LAGUNA_MODE:-}"
case "$MODE" in
  queue)
    ATTEMPT="${SPARK_QUEUE_ATTEMPT:?run through the authoritative spark queue}"
    case "$ATTEMPT" in
      *[!0-9a-f]*|"") fail "bad attempt id" ;;
    esac
    [ "${#ATTEMPT}" -eq 32 ] || fail "bad attempt id length"
    ROOT="${SPARK_QUEUE_RUNTIME_ROOT:?missing queue runtime root}"
    [ "$ROOT" = "/tmp/sparkqueue-$ATTEMPT" ] || fail "unexpected job namespace: $ROOT"
    [ "${SPARK_QUEUE_SIZE:-}" = "$WORLD" ] ||
      fail "SPARK_QUEUE_SIZE must be $WORLD (got '${SPARK_QUEUE_SIZE:-}')"
    RANK="${SPARK_QUEUE_RANK:?SPARK_QUEUE_RANK is required}"
    COLLECTIVE_ID="$(python3 -c "print(int('$ATTEMPT'[:15], 16) + 1 + $LANE)")"
    ;;
  direct)
    ROOT="${LAGUNA_RUNTIME_ROOT:-}"
    [ "$ROOT" = "/tmp/sp-laguna-lane$LANE" ] ||
      fail "LAGUNA_RUNTIME_ROOT must be /tmp/sp-laguna-lane$LANE in direct mode (got '$ROOT')"
    RANK="${LAGUNA_RANK:?LAGUNA_RANK is required in direct mode}"
    COLLECTIVE_ID="${LAGUNA_COLLECTIVE_ID:-}"
    decimal "$COLLECTIVE_ID" && [ "$COLLECTIVE_ID" -gt 0 ] ||
      fail "LAGUNA_COLLECTIVE_ID must be a positive decimal shared by all ranks"
    ;;
  *)
    fail "LAGUNA_MODE must be queue or direct (got '$MODE')"
    ;;
esac
decimal "$RANK" && [ "$RANK" -lt "$WORLD" ] || fail "rank must be 0..$((WORLD - 1))"

reserved_ok() {
  local port="$1" entry first last
  [ -n "${SPARK_QUEUE_PORTS:-}" ] || fail "queue reserved no ports"
  for entry in ${SPARK_QUEUE_PORTS//,/ }; do
    first="${entry%%:*}"; last="${entry##*:}"
    if [ "$port" -ge "$first" ] && [ "$port" -le "$last" ]; then return 0; fi
  done
  fail "listener port $port is not inside a queue-reserved range"
}

if [ "$MODE" = queue ]; then
  for r in $(seq 0 $((WORLD - 1))); do
    reserved_ok $((CONTROL_BASE + r))
    reserved_ok $((COLLECTIVE_BASE + r))
    reserved_ok $((TRANSPORT_BASE + r))
  done
  reserved_ok "$SESSION_BASE"
  reserved_ok $((SESSION_BASE + 55))
fi

KV_BACKING_BYTES="${LAGUNA_KV_BACKING_BYTES:-}"
decimal "$KV_BACKING_BYTES" && [ "$KV_BACKING_BYTES" -gt 0 ] ||
  fail "LAGUNA_KV_BACKING_BYTES must be a positive decimal byte cap"
KV_PAGE_CAPACITY="${LAGUNA_KV_PAGE_CAPACITY:-}"
decimal "$KV_PAGE_CAPACITY" && [ "$KV_PAGE_CAPACITY" -gt 0 ] ||
  fail "LAGUNA_KV_PAGE_CAPACITY must be a positive decimal page count"
for value in LAGUNA_EXPERT_POOL_BYTES LAGUNA_SPINE_BUDGET_BYTES; do
  text="${!value:-}"
  [ -z "$text" ] || { decimal "$text" && [ "$text" -gt 0 ]; } || fail "$value must be a positive decimal"
done

SOCKET="${LAGUNA_WEIGHTD_SOCKET:-}"
case "$SOCKET" in
  /*) ;;
  *) fail "LAGUNA_WEIGHTD_SOCKET must name the running weightd socket by absolute path (got '$SOCKET')" ;;
esac
[ -S "$SOCKET" ] || fail "weightd socket $SOCKET is not a live socket; this wrapper never starts a weightd"

STAGE=$((RANK / 8))
TP_RANK=$((RANK % 8))
MESH_RANKS="$(seq -s, $((STAGE * 8)) $((STAGE * 8 + 7)))"
HOST="spark$(printf '%x' "$RANK")"
[ "$(hostname)" = "$HOST" ] ||
  fail "node order mismatch: rank $RANK expects $HOST but this job runs on $(hostname)"

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOYED_PACK="/home/$HOST/sparkdata/laguna-s-2.1.bf16.tp8pp2/packs/laguna_stage.tp8.pp2.stage${STAGE}.rank${RANK}.lgsp"
[ -f "$DEPLOYED_PACK" ] || fail "deployed rank pack missing: $DEPLOYED_PACK"

mkdir -p "$ROOT/bin" "$ROOT/lib" "$ROOT/config" "$ROOT/packs" "$ROOT/kvcache"

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

GENERATED="$ROOT/generate.$$"
mkdir -p "$GENERATED"
python3 "$CHECKOUT/tools/laguna_multidev_lane.py" \
  --lane "$LANE" \
  --runtime-root "$ROOT" \
  --weightd-socket "$SOCKET" \
  --output-dir "$GENERATED" \
  --rank "$RANK" \
  --collective-identifier "$COLLECTIVE_ID" \
  --kv-backing-bytes "$KV_BACKING_BYTES" \
  --kv-page-capacity "$KV_PAGE_CAPACITY"
install -m 0644 "$GENERATED/deployment.json" "$ROOT/deployment.json"
install -m 0644 "$GENERATED/adapter.json" "$ROOT/config/adapter.json"
rm -rf "$GENERATED"

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
    ok = kind in (28, 30)
raise SystemExit(0 if ok else 1)
PYVER
then
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
  if [ -r "$DEPLOYED_PACK.receipt.json" ]; then
    RECEIPT_SHA="$(python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("output_sha256") or "")' \
      "$DEPLOYED_PACK.receipt.json")"
    [ "$RECEIPT_SHA" = "$(cat "$ROOT/packs/pack.sha256")" ] ||
      fail "pack digest disagrees with its emission receipt: $DEPLOYED_PACK"
  fi
fi
[ "$(find "$ROOT/packs" -maxdepth 1 -name '*.sha256' | wc -l)" = 1 ] ||
  fail "packs/ must contain exactly one .sha256 sidecar"

BUDGETS="$(python3 "$CHECKOUT/tools/laguna_multidev_lane.py" --budgets "$PRIVATE_PACK")"
DEFAULT_POOL="${BUDGETS%% *}"
DEFAULT_SPINE="${BUDGETS##* }"
: "${LAGUNA_EXPERT_POOL_BYTES:=$DEFAULT_POOL}"
: "${LAGUNA_SPINE_BUDGET_BYTES:=$DEFAULT_SPINE}"
export SPARK_WEIGHTD_EXPERT_POOL_BYTES="$LAGUNA_EXPERT_POOL_BYTES"
export SPARK_WEIGHTD_SPINE_BUDGET_BYTES="$LAGUNA_SPINE_BUDGET_BYTES"

if [ -n "${LAGUNA_WORKING_SET:-}" ]; then
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

export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_SOCKET="$SOCKET"
export SPARK_WEIGHTD_LANE="$LANE"
export SPARK_TP_MESH_RANKS="$MESH_RANKS"
export CUDA_MODULE_LOADING=LAZY
export CUDA_MODULE_DATA_LOADING=LAZY
export CUDA_DEVICE_MAX_CONNECTIONS=32
echo "$FAMILY-lane$LANE: mode=$MODE rank=$RANK host=$HOST stage=$STAGE tp=$TP_RANK \
pack=$PRIVATE_PACK pool=${LAGUNA_EXPERT_POOL_BYTES}B spine=${LAGUNA_SPINE_BUDGET_BYTES}B \
kv_pages=$KV_PAGE_CAPACITY socket=$SOCKET collective_id=$COLLECTIVE_ID"
if [ -n "${LAGUNA_GDB:-}" ]; then
  exec gdb --batch -ex run -ex "bt 25" \
    --args "$ROOT/bin/sparkpipe_model_residentd" \
    --deployment "$ROOT/deployment.json" \
    --rank-index "$RANK"
fi
exec "$ROOT/bin/sparkpipe_model_residentd" \
  --deployment "$ROOT/deployment.json" \
  --rank-index "$RANK"
