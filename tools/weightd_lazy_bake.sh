#!/usr/bin/env bash
# weightd_lazy_bake.sh — per-node PURE-LAZY stagepack bake.
#
# For every completed arm hosted on this node (canonical roster; legacy
# trees mapped in tools/stagepack_naming.json): generate the segment
# manifest once (8 MiB streaming, bounded RSS), then run the lazy smoke
# - ATTACH_LAZY with a pool <= LAZY_POOL_MIB, ENSURE only a spread of
# segments (exactly what a smoke test touches; NOTHING else loads),
# verify per-segment ck128 (the fail-closed digest law), detach, reclaim.
# No whole-pack loads, no pinned arenas: committed memory stays inside
# the per-attach pool.
#
# usage: weightd_lazy_bake.sh <repo-checkout>
set -uo pipefail

RR=${1:?usage: weightd_lazy_bake.sh <repo-checkout>}
CUDA=${CUDA_HOME:-/usr/local/cuda}
OUT=${BAKE_OUT:-/tmp/weightd_lazy_bake.log}
SOCK=/tmp/spark_weightd_bake.sock
UNIT=sparkpipe-weightd-bake
LAZY_POOL_MIB=${LAZY_POOL_MIB:-1024}
TOUCHES=${TOUCHES:-8}
touch "$OUT"

ARMS=(
    hy4.fp8.safetensors
    glm53flash.bf16.tp16
    glm53flash.fp8.tp8
    glm53flash.fp8.tp4pp4
    qwen3flash.bf16.tp4
    qwen3flash.bf16.tp8
    qwen3flash.bf16.tp4pp4
    qwen3flash.fp8.tp8
    qwen3flash.fp8.tp4pp4
    qwen3flash.nvfp4.tp8
    dsv4flash.fp8.tp16
    dsv4flash.fp8.tp4pp4
    dsv4pro.fp8.tp16
    dsv4pro.fp8.tp4pp4
    qwen27b.bf16.tp4pp4
    qwen27b.nvfp4a16.tp4pp4
    qwen27b.nvfp4a16.tp4
    glm53full.bf16.tp4pp4
    glm53full.fp8.tp4pp4
    glm53full.nvfp4.tp4pp4
    k3.mxfp4.tp4pp4
)

log() { printf '%s\n' "$*" | tee -a "$OUT"; }

build_binaries() {
    make -j20 build/libsparkpipe_runtime.a build/libsparkpipe_core.a >>"$OUT" 2>&1 || return 1
    cc -O2 -std=c11 -D_GNU_SOURCE -I. -Iinclude -Isrc -I"$CUDA/include" \
        -c runtime/spark_weightd.c -o /tmp/wb_weightd.o >>"$OUT" 2>&1 || return 1
    cc -O2 -std=c11 -D_GNU_SOURCE -I. -Iinclude -Isrc -I"$CUDA/include" \
        -c node/weightd.c -o /tmp/wb_main.o >>"$OUT" 2>&1 || return 1
    cc /tmp/wb_main.o /tmp/wb_weightd.o build/libsparkpipe_runtime.a \
        build/libsparkpipe_core.a -L"$CUDA/lib64" -lcudart -lcuda -lpthread -lm \
        -o /tmp/wb_weightd >>"$OUT" 2>&1 || return 1
    cc -O2 -std=c11 -D_GNU_SOURCE -I. -Iinclude -Isrc \
        tools/weightd_smoke.c /tmp/wb_weightd.o build/libsparkpipe_runtime.a \
        build/libsparkpipe_core.a -L"$CUDA/lib64" -lcudart -lcuda -lpthread -lm \
        -o /tmp/wb_smoke >>"$OUT" 2>&1 || return 1
    cc -O2 -std=c11 -D_GNU_SOURCE -I. -Iinclude -Isrc \
        tools/weightd_expert_segments.c build/libsparkpipe_core.a \
        -lpthread -lm -o /tmp/wb_segments >>"$OUT" 2>&1 || return 1
}

start_daemon() {
    if [ -S "$SOCK" ]; then
        return 0
    fi
    systemctl --user reset-failed "$UNIT" >/dev/null 2>&1
    systemctl --user stop "$UNIT" >/dev/null 2>&1
    unlink "$SOCK" 2>/dev/null || true
    systemd-run --user --unit="$UNIT" --collect \
        /tmp/wb_weightd --socket "$SOCK" --device-bytes-max 4294967296 \
        >>"$OUT" 2>&1 || return 1
    local i
    for i in $(seq 1 40); do
        [ -S "$SOCK" ] && return 0
        sleep 0.25
    done
    return 1
}

discover() {
    local arm p d
    for arm in "${ARMS[@]}"; do
        if [ "$arm" = "hy4.fp8.safetensors" ]; then
            for p in "$HOME"/hy4-fp8-packs/model-fp8-tp16-rank-*.safetensors; do
                [ -f "$p" ] && printf '%s\n' "$p"
            done
        else
            for d in "$HOME"/sparkdata/"$arm"/packs \
                      "$HOME"/sparkdata/"$arm"/packs_v4; do
                for p in "$d"/*; do
                    case "$p" in
                        *.json|*.sha256|*.ck128|*.receipt*|*.experts) continue ;;
                    esac
                    [ -f "$p" ] && printf '%s\n' "$p"
                done
            done
        fi
    done | awk '!seen[$0]++'
}

cd "$RR" || exit 2
export SPARK_WEIGHTD_SOCKET="$SOCK"
export SPARK_WEIGHTD_ATTACH=1
log "LAZY-BAKE-PASS-BEGIN $(date -u +%FT%TZ) pool_mib=$LAZY_POOL_MIB touches=$TOUCHES"
build_binaries || { log "LAZY-BAKE-BUILD-FAIL"; exit 2; }
start_daemon || { log "LAZY-BAKE-DAEMON-FAIL"; exit 2; }

ok=0
bad=0
while IFS= read -r pack; do
    case "$pack" in
        */hy4-fp8-packs/*) model=hy4.fp8.tp16 ;;
        *) model=$(basename "$(dirname "$(dirname "$pack")")") ;;
    esac
    if [ ! -f "$pack.experts" ]; then
        /tmp/wb_segments "$pack" >>"$OUT" 2>&1 || {
            bad=$((bad+1))
            log "LAZY host=$(hostname) model=$model pack=$(basename "$pack") state=SEGMENTS-FAIL"
            continue
        }
    fi
    verdict=$(/tmp/wb_smoke "$pack" "$model" bake "$LAZY_POOL_MIB" "$TOUCHES" 2>&1 | tail -1)
    case "$verdict" in
        SMOKE*) ok=$((ok+1)); state=OK ;;
        *) bad=$((bad+1)); state=BAD ;;
    esac
    log "LAZY host=$(hostname) model=$model pack=$(basename "$pack") state=$state verdict=$verdict"
done < <(discover)

log "LAZY-BAKE-END host=$(hostname) ok=$ok bad=$bad"
[ "$bad" -eq 0 ] || exit 1
exit 0