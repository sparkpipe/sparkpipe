#!/usr/bin/env bash
# qwen38max head-split module WIP no-regression (TP1 module tier).
# Same cell as q38max-tp16k-regress4 but against the headsplit-design tree
# (kv-replication kernels + attn-projection slice loader + local geometry).
# At tp_degree=1 the slice branch is inert; the PASS must match pre-change.
set -uo pipefail
REPO="$HOME/q38max-headsplit"
VDIR="$REPO/modules/qwen38_max_resident_decode_stage/validation"
cd "$REPO" || exit 13

make -C modules/qwen38_max_resident_decode_stage REPOSITORY_ROOT="$REPO" archive || exit 13
ARCHIVE="$REPO/build/modules/qwen38_resident_decode_stage/libqwen38_resident_decode_stage.a"

SYNTH="$REPO/build/q38max_synth"
cc -std=c11 -O2 -I"$REPO" -I"$REPO/model-families/qwen38_max/include" \
   -I"$REPO/modules/qwen38_max_resident_decode_stage/include" \
   -I"$REPO/modules/qwen38_max_resident_decode_stage/source" \
   modules/qwen38_max_resident_decode_stage/tools/qwen38_max_pack_synthesize.c -o "$SYNTH" || exit 13
PACK="$HOME/q38max_hs_regress.qwen38sp"
"$SYNTH" --output "$PACK" --first-layer 0 --layer-count 4 || exit 13

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

CFG_SHA=$(sha256sum "$VDIR/spark_qwen38_max_resident_decode_stage_cuda_validation.cu" | awk '{print $1}')
export SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256="$CFG_SHA"

bash "$VDIR/validate_qwen38_max_resident_decode_stage_cuda.sh" "$CFG_SHA" "$ARCHIVE"
rc=$?
echo "q38max_headsplit_regress exit=$rc"
exit $rc
