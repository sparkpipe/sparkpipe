#!/usr/bin/env bash
set -euo pipefail
fail() { printf 'gemma4_lane_resident: FAIL: %s\nusage: gemma4_lane_resident.sh --release DIR --deployment DIR --packs DIR --root DIR --rank N --lane N --mesh-ranks CSV --weightd-socket PATH --unit-prefix NAME --memory-max SIZE [--dry-run]\n' "$1" >&2; exit 1; }
DRY_RUN=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --release) RELEASE=$2; shift 2;;
        --deployment) DEPLOY_TREE=$2; shift 2;;
        --packs) PACK_DIR=$2; shift 2;;
        --root) ROOT=$2; shift 2;;
        --rank) RANK=$2; shift 2;;
        --lane) LANE=$2; shift 2;;
        --mesh-ranks) MESH_RANKS=$2; shift 2;;
        --weightd-socket) WEIGHTD_SOCKET=$2; shift 2;;
        --unit-prefix) UNIT_PREFIX=$2; shift 2;;
        --memory-max) MEMORY_MAX=$2; shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        *) fail "unknown argument $1";;
    esac
done
for name in RELEASE DEPLOY_TREE PACK_DIR ROOT RANK LANE MESH_RANKS WEIGHTD_SOCKET UNIT_PREFIX MEMORY_MAX; do
    [ -n "${!name:-}" ] || fail "missing --$(printf '%s' "$name" | tr 'A-Z_' 'a-z-')"
done
case "$RANK" in (*[!0-9]*) fail "--rank is not a number";; esac
case "$LANE" in (*[!0-9]*) fail "--lane is not a number";; esac
RELEASE=$(cd "$RELEASE" && pwd)
DEPLOY_TREE=$(cd "$DEPLOY_TREE" && pwd)
(cd "$RELEASE" && sha256sum --quiet --strict --check SHA256SUMS) || fail "release does not match its SHA256SUMS"
STAGE_CONFIG="$DEPLOY_TREE/config/stage_$(printf '%02d' "$RANK").json"
ENV_JSON="$DEPLOY_TREE/config/env_$(printf '%02d' "$RANK").json"
[ -f "$STAGE_CONFIG" ] || fail "stage config missing: $STAGE_CONFIG"
[ -f "$ENV_JSON" ] || fail "module env missing: $ENV_JSON"
PACK_PATH=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage_pack_path"])' "$STAGE_CONFIG")
PACK_NAME=$(basename "$PACK_PATH")
[ -f "$PACK_DIR/$PACK_NAME" ] || fail "rank pack missing: $PACK_DIR/$PACK_NAME"
[ -f "$PACK_DIR/$PACK_NAME.sha256" ] || fail "pack digest sidecar missing: $PACK_DIR/$PACK_NAME.sha256"
read -r SHA_HEX SHA_NAME < "$PACK_DIR/$PACK_NAME.sha256"
[ "${#SHA_HEX}" -eq 64 ] && [ "$SHA_NAME" = "$PACK_NAME" ] || fail "malformed sidecar $PACK_DIR/$PACK_NAME.sha256"
UNIT="sp-${UNIT_PREFIX}-resident-r${RANK}"
if systemctl --user is-active --quiet "$UNIT"; then
    fail "$UNIT is already active; stop it first"
fi
mkdir -p "$ROOT"/bin "$ROOT"/lib "$ROOT"/packs "$ROOT"/config "$ROOT"/kv
ln -sfn "$RELEASE/bin/sparkpipe_model_residentd" "$ROOT/bin/sparkpipe_model_residentd"
ln -sfn "$RELEASE/lib/model_serving_adapter.so" "$ROOT/lib/model_serving_adapter.so"
ln -sfn "$RELEASE/lib/hidden_transport.so" "$ROOT/lib/hidden_transport.so"
ln -sfn "$RELEASE/stages" "$ROOT/stages"
cp "$STAGE_CONFIG" "$ROOT/config/stage.json"
find "$ROOT/packs" -maxdepth 1 -name '*.sha256' ! -name "$PACK_NAME.sha256" -delete
ln -sfn "$PACK_DIR/$PACK_NAME" "$ROOT/packs/$PACK_NAME"
cp "$PACK_DIR/$PACK_NAME.sha256" "$ROOT/packs/$PACK_NAME.sha256"
cp "$DEPLOY_TREE/model_resident.json" "$ROOT/model_resident.json"
ENVIRONMENT=(
    --setenv=SPARK_WEIGHTD_ATTACH=1
    --setenv=SPARK_WEIGHTD_PACK_SHA256="$SHA_HEX"
    --setenv=SPARK_WEIGHTD_IDENTITY_MODEL=cuda.sm121.gemma4.31b.resident_decode_stage.bf16
    --setenv=SPARK_WEIGHTD_SOCKET="$WEIGHTD_SOCKET"
    --setenv=SPARK_WEIGHTD_LANE="$LANE"
    --setenv=SPARK_TP_MESH_RANKS="$MESH_RANKS"
    --setenv=CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0
    --setenv=LD_LIBRARY_PATH="$ROOT/lib"
)
while IFS='=' read -r key value; do
    [ -n "$key" ] && ENVIRONMENT+=(--setenv="$key=$value")
done < <(python3 -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.stdout.write("".join("%s=%s\n"%(k,v) for k,v in sorted(e.items())))' "$ENV_JSON")
COMMAND=(systemd-run --user --unit="$UNIT" --collect -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0
    --working-directory="$ROOT" "${ENVIRONMENT[@]}"
    -p StandardOutput=append:"$ROOT/residentd.log" -p StandardError=append:"$ROOT/residentd.log"
    "$ROOT/bin/sparkpipe_model_residentd" --deployment "$ROOT/model_resident.json" --rank-index "$RANK")
if [ "$DRY_RUN" -eq 1 ]; then
    printf 'gemma4_lane_resident: dry-run OK rank=%s pack=%s unit=%s\n' "$RANK" "$PACK_NAME" "$UNIT"
    printf '%q ' "${COMMAND[@]}"; printf '\n'
    exit 0
fi
[ -S "$WEIGHTD_SOCKET" ] || fail "weightd socket is not present: $WEIGHTD_SOCKET"
"${COMMAND[@]}"
