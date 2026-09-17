#!/usr/bin/env bash
# weightd_stagepack_bake.sh — per-node stagepack residency bake.
#
# For every COMPLETED stagepack arm (operator roster 09-04, all 16/16)
# hosted on this node plus the hy4 FP8 safetensors FIRST: cold-load the
# pack through weightd (full-file digest verification + device arena +
# consumer import/map via weightdctl — the same path a driver takes),
# then release it. MTP variants and drafter arms are excluded
# (speculation moved to the rtx5090 node, operator 09-04).
#
# usage: weightd_stagepack_bake.sh <repo-checkout>
set -uo pipefail

RR=${1:?usage: weightd_stagepack_bake.sh <repo-checkout>}
CUDA=${CUDA_HOME:-/usr/local/cuda}
OUT=${BAKE_OUT:-/tmp/weightd_stagepack_bake.log}
SOCK=/tmp/spark_weightd_bake.sock
UNIT=sparkpipe-weightd-bake
touch "$OUT"
log "BAKE-PASS-BEGIN $(date -u +%FT%TZ)"

ARMS=(
    hy4.fp8.safetensors
    qwen4_flash.tp4
    qwenflash.tp8
    qwenflash.tp8.fp8
    qwenflash.tp8.nvfp4
    qwenflash.tp4pp4
    qwenflash.tp4pp4.fp8
    dsv4flash.tp16
    dsv4flash.tp4pp4
    dsv4_pro.tp16
    dsv4_pro.tp4pp4
    glm5_next.tp16
    glm5_next.tp4pp4
    glm5_next.tp8.fp8
    glm5_next.bf16.tp16
    glm53full.bf16.tp4pp4
    glm53full.fp8.tp4pp4
    glm53full.nvfp4.tp4pp4
    k3.mxfp4.tp4pp4
    qwen27b.tp4pp4
    qwen38_27b.tp4pp4
    qwen38-27b.nvfp4a16.tp4
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
    cc -O2 -std=c11 -D_GNU_SOURCE -I. -Iinclude -Isrc -I"$CUDA/include" \
        tools/weightdctl.c /tmp/wb_weightd.o build/libsparkpipe_runtime.a \
        build/libsparkpipe_core.a -L"$CUDA/lib64" -lcudart -lcuda -lpthread -lm \
        -o /tmp/wb_weightdctl >>"$OUT" 2>&1 || return 1
}

start_daemon() {
    if [ -S "$SOCK" ]; then
        return 0
    fi
    systemctl --user reset-failed "$UNIT" >/dev/null 2>&1
    systemctl --user stop "$UNIT" >/dev/null 2>&1
    unlink "$SOCK" 2>/dev/null || true
    systemd-run --user --unit="$UNIT" --collect \
        /tmp/wb_weightd --socket "$SOCK" --device-bytes-max 96636764160 \
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
                        *.json|*.sha256|*.ck128|*.receipt*) continue ;;
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
log "BAKE-BEGIN host=$(hostname) checkout=$RR"
build_binaries || { log "BAKE-BUILD-FAIL"; exit 2; }
log "BAKE-BUILD-OK"
start_daemon || { log "BAKE-DAEMON-FAIL"; exit 2; }
log "BAKE-DAEMON-UP socket=$SOCK"

ok=0
bad=0
total_bytes=0
total_secs=0
while IFS= read -r pack; do
    case "$pack" in
        */hy4-fp8-packs/*) model=hy4.fp8.tp16 ;;
        *) model=$(basename "$(dirname "$(dirname "$pack")")") ;;
    esac
    bytes=$(stat -c%s "$pack" 2>/dev/null || echo 0)
    t0=$(date +%s.%N)
    verdict=$(/tmp/wb_weightdctl load "$pack" "$model" bake 2>&1 | tail -1)
    t1=$(date +%s.%N)
    /tmp/wb_weightdctl unload "$pack" "$model" bake >/dev/null 2>&1
    secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
    case "$verdict" in
        ATTACHED*) ok=$((ok+1)); state=OK ;;
        *) bad=$((bad+1)); state=BAD ;;
    esac
    total_bytes=$((total_bytes+bytes))
    total_secs=$(awk -v a="$total_secs" -v b="$secs" 'BEGIN{printf "%.2f", a+b}')
    log "BAKE host=$(hostname) model=$model pack=$(basename "$pack") bytes=$bytes secs=$secs state=$state verdict=$verdict"
done < <(discover)

systemctl --user stop "$UNIT" >/dev/null 2>&1
gbps=$(awk -v b="$total_bytes" -v s="$total_secs" 'BEGIN{if (s>0) printf "%.2f", b/s/1073741824; else printf "0"}')
log "BAKE-END host=$(hostname) ok=$ok bad=$bad packs_bytes=$total_bytes secs=$total_secs gbps=$gbps"
[ "$bad" -eq 0 ] || exit 1
exit 0