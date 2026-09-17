#!/usr/bin/env bash
# qwen38max TP16-enablement no-regression validation (mid-pipeline stage 0,
# 4 layers, synth v2 MXFP4 pack). Exercises the patched kernels' TP1 module
# tier + TP4 kernel-tier rank-local checks; the macros must reduce to the
# pre-TP16 numerics. Run on spark7 via the spark queue.
set -uo pipefail
REPO="$HOME/sparkpipe-lane"
PACK="$HOME/q38max_tp16regress.qwen38sp"
LOG="$HOME/q38max_tp16regress.log"
cd "$REPO" || exit 13

if [ ! -s "$PACK" ]; then
    cc -std=c11 -O2 -I"$REPO" -I"$REPO/modules/qwen38_max_resident_decode_stage/source" \
       -I"$REPO/model-families/qwen38_max/include" -I"$REPO/modules/qwen38_max_resident_decode_stage/include" \
       modules/qwen38_max_resident_decode_stage/tools/qwen38_max_pack_synthesize.c -o /tmp/q38max_synth || exit 13
    /tmp/q38max_synth --output "$PACK" --first-layer 0 --layer-count 4 || exit 13
fi

make -C modules/qwen38_max_resident_decode_stage REPOSITORY_ROOT="$REPO" archive || exit 13

export NVCC=/usr/local/cuda/bin/nvcc
export SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION=1
export SPARK_QWEN38_MAX_STAGE_PACK_PATH="$PACK"
export SPARK_QWEN38_MAX_STAGE_COUNT=2
export SPARK_QWEN38_MAX_STAGE_INDEX=0
export SPARK_QWEN38_MAX_STAGE_FIRST_LAYER=0
export SPARK_QWEN38_MAX_STAGE_LAYER_COUNT=4
export SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES=8
export SPARK_QWEN38_MAX_STAGE_KV_BLOCKS=64
export SPARK_QWEN38_MAX_STAGE_MTP=0
export SPARK_QWEN38_MAX_TP_DEGREE=1

VDIR=modules/qwen38_max_resident_decode_stage/validation
CFG_SHA=$(sha256sum "$VDIR/spark_qwen38_max_resident_decode_stage_cuda_validation.cu" | awk '{print $1}')
export SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256="$CFG_SHA"

bash "$VDIR/validate_qwen38_max_resident_decode_stage_cuda.sh" "$CFG_SHA" \
     "$REPO/build/modules/qwen38_resident_decode_stage/libqwen38_resident_decode_stage.a" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "q38max_tp16_regression exit=$rc (log $LOG)"
