#!/usr/bin/env bash
# validate_ga_pro.sh SLICE_COUNT [FIRST_LAYER] [STAGE_INDEX] - build a GA
# val slice pack on spark3, ship it to sparkb, and run the single-spark GPU
# validation against the GA (ga0813) b1 module. Defaults: the 0+4 val4 slice.
set -euo pipefail

COUNT="${1:-4}"
FIRST="${2:-0}"
STAGE_INDEX="${3:-0}"
STAGE_COUNT=16
SOURCE_DIR="/home/spark3/extnvme/models/hf/deepseek-ai/DeepSeek-V4-Pro-0813"
PACK="/home/spark3/pro-repo/dsv4_pro_ga.val${FIRST}p${COUNT}.spstage"

echo "== pack (first=${FIRST} count=${COUNT}) =="
ssh -o BatchMode=yes spark3 "cd /home/spark3/pro-repo && python3 tools/dsv4_pro_stagepack.py --model-dir ${SOURCE_DIR} --first-layer ${FIRST} --layer-count ${COUNT} --output $(basename ${PACK})" 2>&1 | tail -4

echo "== ship to sparkb =="
ssh -o BatchMode=yes sparkb "rsync -a spark3:${PACK} /home/sparkb/sparkdata/" 2>&1 | tail -1

echo "== build validator (GA b1 archive) =="
ssh -o BatchMode=yes sparkb "cd /tmp/sparkpipe-pro-dev && /usr/local/cuda-13.0/bin/nvcc -std=c++17 -O3 --expt-relaxed-constexpr -gencode arch=compute_121a,code=sm_121a -DSPARK_DSV4_PRO_BUILD=1 -DSPARK_BATCH_BUCKET=1 -I/tmp/sparkpipe-pro-dev/include -I/tmp/sparkpipe-pro-dev/model-families/dsv4/include -I/tmp/sparkpipe-pro-dev/modules/dsv4_resident_decode_stage/include -I/tmp/sparkpipe-pro-dev/modules/dsv4_resident_decode_stage/source modules/dsv4_resident_decode_stage/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu build/modules/dsv4_pro_resident_decode_stage/libdsv4_resident_decode_stage_b1.a build/libsparkpipe_runtime.a build/libsparkpipe_core.a -L/usr/local/cuda-13.0/lib64 -lcuda -lcudart -ldl -lm -Xcompiler -pthread -o /tmp/dsv4pro-validator-ga" 2>&1 | tail -2

echo "== run validation (stage=${STAGE_INDEX}/${STAGE_COUNT} slice=${FIRST}+${COUNT}) =="
CONFIG_SHA="$(printf '%s' "dsv4_pro ga validation stage=${STAGE_INDEX}/${STAGE_COUNT} slice=${FIRST}+${COUNT} rows=1 max_seq=4096 logical_pages=1024 physical_pages=1024 mtp=0 graphs=0" | shasum -a 256 | cut -d' ' -f1)"
ssh -o BatchMode=yes sparkb "cd /tmp/sparkpipe-pro-dev && SPARK_DSV4_STAGE_COUNT=${STAGE_COUNT} SPARK_DSV4_STAGE_INDEX=${STAGE_INDEX} SPARK_DSV4_STAGE_FIRST_LAYER=${FIRST} SPARK_DSV4_STAGE_LAYER_COUNT=${COUNT} SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES=1 SPARK_DSV4_STAGE_PIPELINE_SLOTS=1 SPARK_DSV4_STAGE_MAX_SEQ=4096 SPARK_DSV4_STAGE_LOGICAL_PAGES=1024 SPARK_DSV4_STAGE_PHYSICAL_PAGES=1024 SPARK_DSV4_STAGE_MTP=0 SPARK_DSV4_STAGE_GRAPHS=0 SPARK_DSV4_STAGE_PACK_PATH=/home/sparkb/sparkdata/$(basename ${PACK}) /tmp/dsv4pro-validator-ga ${CONFIG_SHA} > /tmp/dsv4pro-ga-validate.log 2>&1; echo exit=$?; tail -6 /tmp/dsv4pro-ga-validate.log"
