#!/usr/bin/env bash
# qwen38max smoke validation on the CURRENT main lineage (sota branch):
# kernel-tier gate (with the validator fixes) + module tier with the
# PIPELINE_SLOTS env. Smoke scale (4-layer synth pack), coexists with
# other lanes' resident sets. Detached result: ~/q38max_sota_smoke.status
set -uo pipefail
REPO="$HOME/q38max-sota"
VDIR="$REPO/modules/qwen38_max_resident_decode_stage/validation"
cd "$REPO" || exit 13

make -C modules/qwen38_max_resident_decode_stage REPOSITORY_ROOT="$REPO" archive || exit 13
ARCHIVE="$REPO/build/modules/qwen38_resident_decode_stage/libqwen38_resident_decode_stage.a"

SYNTH="$REPO/build/q38max_synth"
EXPECTED_PACK_BYTES=111078450176
PACK="$HOME/q38max_sota_smoke.qwen38sp"
if [ ! -s "$SYNTH" ]; then
    cc -std=c11 -O2 -I"$REPO" -I"$REPO/include" -I"$REPO/src" \
       -I"$REPO/model-families/common/include" \
       -I"$REPO/model-families/qwen38_max/include" \
       -I"$REPO/modules/qwen38_max_resident_decode_stage/include" \
       -I"$REPO/modules/qwen38_max_resident_decode_stage/source" \
       modules/qwen38_max_resident_decode_stage/tools/qwen38_max_pack_synthesize.c \
       "$REPO/runtime/stagepack_format.c" -o "$SYNTH" || exit 13
fi
if [ ! -s "$PACK" ] || [ "$(stat -c %s "$PACK" 2>/dev/null || echo 0)" != "$EXPECTED_PACK_BYTES" ]; then
    "$SYNTH" --output "$PACK" --first-layer 0 --layer-count 4 || exit 13
fi

STATUS="$HOME/q38max_sota_smoke.status"
rm -f "$STATUS"
setsid nohup bash -c '
    export NVCC=/usr/local/cuda/bin/nvcc
    export SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION=1
    export SPARK_QWEN38_MAX_STAGE_PACK_PATH="$1"
    export SPARK_QWEN38_MAX_STAGE_COUNT=2
    export SPARK_QWEN38_MAX_STAGE_INDEX=0
    export SPARK_QWEN38_MAX_STAGE_FIRST_LAYER=0
    export SPARK_QWEN38_MAX_STAGE_LAYER_COUNT=4
    export SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES=8
    export SPARK_QWEN38_MAX_STAGE_KV_BLOCKS=64
    export SPARK_QWEN38_MAX_STAGE_MTP=0
    export SPARK_QWEN38_MAX_TP_DEGREE=1
    export SPARK_QWEN38_MAX_STAGE_PIPELINE_SLOTS=2
    CFG=$(sha256sum "'$VDIR'/spark_qwen38_max_resident_decode_stage_cuda_validation.cu" | awk "{print \$1}")
    export SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256="$CFG"
    bash "'$VDIR'/validate_qwen38_max_resident_decode_stage_cuda.sh" "$CFG" "$2"
    echo "q38max_sota_smoke exit=$?" > "$3"
' _ "$PACK" "$ARCHIVE" "$STATUS" \
    > "$HOME/q38max_sota_smoke_detached.log" 2>&1 < /dev/null &
echo "detached: validation running, status at $STATUS"
exit 0
