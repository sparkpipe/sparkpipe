#!/usr/bin/env bash
# gemma4-31b TP16 shared-socket wrapper (lane 6) - the run-family-job.sh
# command for tools/spark_queue.py per docs/MULTIDEV_QUICKSTART.md.
#
# Runs INSIDE one admitted gpu-shared queue job on each participating node
# (env provided by the queue: SPARK_QUEUE_RANK, SPARK_QUEUE_SIZE,
# SPARK_QUEUE_RUNTIME_ROOT). It prepares a private deployment under the
# runtime root, attaches to the SHARED weightd (never starts a daemon),
# and execs the resident launch so systemd keeps the process in the job
# cgroup:
#
#   export SPARK_WEIGHTD_SOCKET=$GEMMA4_SHARED_WEIGHTD_SOCKET
#   export SPARK_WEIGHTD_LANE=6
#   export SPARK_TP_MESH_RANKS=0,1,...,15
#   exec bin/sparkpipe_model_residentd --deployment "$ROOT/deployment.json" \
#       --rank-index "$SPARK_QUEUE_RANK"
#
# Prerequisites in the synced checkout (cwd):
#   - tools/gemma4_build_release.sh ran (build/gemma4_31b_tp16 with
#     SOURCE_COMMIT + SHA256SUMS)
#   - python3 tools/gemma4_tp16_gen_deployment.py --output deployment/gemma4_31b_tp16_lane6
#   - the landed rank pack on this node:
#     ~/sparkdata/gemma4_31b.bf16.tp16/packs/gemma4_31b_tp16_rank<hex>_stage0.gemma4sp
#     (+ its .sha256 sidecar; the nvfp4 arm has no sidecars yet and fails closed)
#
# Loading fails closed (docs/PARALLEL_DRIVER_DEBUG.md): missing socket, pack,
# sidecar, release artifacts or env are errors, never silent fallbacks.
#
# Usage:
#   tools/gemma4_tp16_shared_socket.sh [--dry-run]
#     GEMMA4_SHARED_WEIGHTD_SOCKET=/run/sparkpipe-weightd-shared/weightd.sock
#     GEMMA4_PACK_DIR=$HOME/sparkdata/gemma4_31b.bf16.tp16/packs
#     GEMMA4_RELEASE_DIR=build/gemma4_31b_tp16
#     GEMMA4_DEPLOYMENT_TREE=deployment/gemma4_31b_tp16_lane6
set -euo pipefail

LANE=6
RANKS=16
MESH_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
elif [ "$#" -ne 0 ]; then
    printf 'usage: %s [--dry-run]\n' "$0" >&2
    exit 2
fi

fail() { printf 'gemma4_tp16_shared_socket: FAIL: %s\n' "$1" >&2; exit 1; }

# Hard rule (fleet ruling, issue #1125 class): refuse to run without a finite
# queue-provided memory bound - an unbounded unit fences all co-admission on
# its node until exit. The queue exports SPARK_QUEUE_MEMORY_MIB on every
# properly submitted job; missing/zero means unbounded.
if [ -z "${SPARK_QUEUE_MEMORY_MIB:-}" ] || [ "${SPARK_QUEUE_MEMORY_MIB}" = "0" ]; then
    printf '%s
' "gemma4_tp16_shared_socket: FAIL: SPARK_QUEUE_MEMORY_MIB is unset/zero - submit through the queue with --memory-mib (finite MemoryMax is a hard requirement)" >&2
    exit 2
fi

[ -n "${SPARK_QUEUE_RANK:-}" ] || fail "SPARK_QUEUE_RANK is not set (run inside a spark_queue job)"
RANK="$SPARK_QUEUE_RANK"
case "$RANK" in (*[!0-9]*|'') fail "SPARK_QUEUE_RANK is not a rank: $RANK";; esac
[ "$RANK" -lt "$RANKS" ] || fail "SPARK_QUEUE_RANK $RANK outside 0..$((RANKS-1))"
[ -n "${SPARK_QUEUE_RUNTIME_ROOT:-}" ] || fail "SPARK_QUEUE_RUNTIME_ROOT is not set"
ROOT="$SPARK_QUEUE_RUNTIME_ROOT"

CHECKOUT="$(pwd)"
RELEASE="${GEMMA4_RELEASE_DIR:-build/gemma4_31b_tp16}"
DEPLOY_TREE="${GEMMA4_DEPLOYMENT_TREE:-deployment/gemma4_31b_tp16_lane6}"
PACK_DIR="${GEMMA4_PACK_DIR:-$HOME/sparkdata/gemma4_31b.bf16.tp16/packs}"
WEIGHTD_SOCKET="${GEMMA4_SHARED_WEIGHTD_SOCKET:-/run/sparkpipe-weightd-shared/weightd.sock}"
RANK_HEX="$(printf '%x' "$RANK")"
PACK_NAME="gemma4_31b_tp16_rank${RANK_HEX}_stage0.gemma4sp"

# --- fail-closed prerequisites --------------------------------------------
[ -d "$RELEASE" ] || fail "release missing: $CHECKOUT/$RELEASE (run tools/gemma4_build_release.sh in a queue job first)"
[ -f "$RELEASE/SOURCE_COMMIT" ] || fail "release provenance missing: $RELEASE/SOURCE_COMMIT"
[ -f "$RELEASE/SHA256SUMS" ] || fail "release checksums missing: $RELEASE/SHA256SUMS"
for artifact in bin/sparkpipe_model_residentd lib/model_serving_adapter.so \
    lib/hidden_transport.so stages/stage_000/model_driver.so; do
    [ -e "$RELEASE/$artifact" ] || fail "release artifact missing: $RELEASE/$artifact"
done
[ -f "$DEPLOY_TREE/model_resident.json" ] || fail "deployment tree missing: $DEPLOY_TREE/model_resident.json (run tools/gemma4_tp16_gen_deployment.py)"
STAGE_CONFIG="$DEPLOY_TREE/config/stage_$(printf '%02d' "$RANK").json"
ENV_JSON="$DEPLOY_TREE/config/env_$(printf '%02d' "$RANK").json"
[ -f "$STAGE_CONFIG" ] || fail "stage config missing: $STAGE_CONFIG"
[ -f "$ENV_JSON" ] || fail "module env missing: $ENV_JSON"
[ -f "$PACK_DIR/$PACK_NAME" ] || fail "rank pack missing: $PACK_DIR/$PACK_NAME"
[ -f "$PACK_DIR/$PACK_NAME.sha256" ] || fail "pack digest sidecar missing: $PACK_DIR/$PACK_NAME.sha256 (unverified pack; placement receipts required)"

# The sidecar is the placement-time acceptance: a well-formed
# "<hex64>  <pack basename>" line naming exactly this pack. The shared
# weightd attach path performs its own full-digest verification; this
# check fails closed on absent/malformed identity without re-hashing
# 3.9 GB on every launch (the < 5 s cold-launch budget).
SHA_LINE="$(tr -d '\n' < "$PACK_DIR/$PACK_NAME.sha256")"
SHA_HEX="${SHA_LINE%% *}"
SHA_NAME="${SHA_LINE##* }"
case "$SHA_HEX" in
    ''|*[!0-9a-f]*) fail "malformed pack digest sidecar: $PACK_DIR/$PACK_NAME.sha256";;
esac
[ "${#SHA_HEX}" -eq 64 ] || fail "malformed pack digest sidecar: $PACK_DIR/$PACK_NAME.sha256"
[ "$SHA_NAME" = "$PACK_NAME" ] || fail "sidecar names '$SHA_NAME', expected '$PACK_NAME'"

# The pack-identity attach path requires the digest env (no fallback:
# runtime/spark_weightd_attach.c fails "no_identity" without it) plus the
# explicit opt-in; the model identity is pinned for deterministic receipts.
export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_PACK_SHA256="$SHA_HEX"
export SPARK_WEIGHTD_IDENTITY_MODEL=spark.gemma4.31b.resident_decode_stage.bf16.linear_bf16.kv_bf16.h5376.l60.v1

if [ "$DRY_RUN" -eq 0 ]; then
    [ -S "$WEIGHTD_SOCKET" ] || fail "shared weightd socket is not present: $WEIGHTD_SOCKET (weightd is operator-managed; never start it by hand)"
fi

# --- private deployment under the runtime root ----------------------------
mkdir -p "$ROOT"/bin "$ROOT"/lib "$ROOT"/packs "$ROOT"/config "$ROOT"/logs
# residentd requires the kv backing directory to be a real directory
# (node/model_residentd.c ValidateDirectories fails io_error otherwise).
mkdir -p "$ROOT/kv"
ln -sfn "$CHECKOUT/$RELEASE/bin/sparkpipe_model_residentd" "$ROOT/bin/sparkpipe_model_residentd"
ln -sfn "$CHECKOUT/$RELEASE/lib/model_serving_adapter.so" "$ROOT/lib/model_serving_adapter.so"
ln -sfn "$CHECKOUT/$RELEASE/lib/hidden_transport.so" "$ROOT/lib/hidden_transport.so"
ln -sfn "$CHECKOUT/$RELEASE/stages" "$ROOT/stages"
cp "$STAGE_CONFIG" "$ROOT/config/stage.json"
cp "$ENV_JSON" "$ROOT/config/env.json"
# Exactly one valid *.sha256 digest for shared weightd attachment; the pack
# bytes themselves are shared read-only through the symlink.
ln -sfn "$PACK_DIR/$PACK_NAME" "$ROOT/packs/$PACK_NAME"
cp "$PACK_DIR/$PACK_NAME.sha256" "$ROOT/packs/$PACK_NAME.sha256"
pack_sidecars="$(find "$ROOT/packs" -maxdepth 1 -name '*.sha256' | wc -l | tr -d ' ')"
[ "$pack_sidecars" -eq 1 ] || fail "runtime packs/ must hold exactly one *.sha256 digest, found $pack_sidecars"
sed -e "s|\${SPARK_QUEUE_RUNTIME_ROOT}|$ROOT|g" "$DEPLOY_TREE/model_resident.json" > "$ROOT/deployment.json"

# --- environment: shared socket + lane mesh + per-rank module env ---------
while IFS='=' read -r key value; do
    [ -n "$key" ] || continue
    export "$key=$value"
done < <(python3 -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.stdout.write("".join("%s=%s\n"%(k,v) for k,v in e.items()))' "$ENV_JSON")
export SPARK_WEIGHTD_SOCKET="$WEIGHTD_SOCKET"
export SPARK_WEIGHTD_LANE="$LANE"
export SPARK_TP_MESH_RANKS="$MESH_RANKS"

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'gemma4_tp16_shared_socket: dry-run OK rank=%d (%s) root=%s\n' "$RANK" "$PACK_NAME" "$ROOT"
    printf '  socket=%s lane=%s mesh=%s\n' "$WEIGHTD_SOCKET" "$LANE" "$MESH_RANKS"
    printf '  would exec: %s --deployment %s/deployment.json --rank-index %s\n' \
        "$ROOT/bin/sparkpipe_model_residentd" "$ROOT" "$RANK"
    exit 0
fi

exec "$ROOT/bin/sparkpipe_model_residentd" \
    --deployment "$ROOT/deployment.json" \
    --rank-index "$RANK"
