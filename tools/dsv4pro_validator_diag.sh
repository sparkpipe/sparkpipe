#!/bin/bash
# DSV4 Pro validator DIAG: run the module publish's validator step by
# hand under bash -x so the real gate_route failure is visible (which
# binary runs, which archive links, the actual cuda error string).
# After the run: strings-probe the validator binary + link units for
# the diag marker and list what was linked. CPU/GPU on THIS node only.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p /mnt/model-warm/packbuild/dsv4pro
LOG=/mnt/model-warm/packbuild/dsv4pro/validator_diag_$(date +%s).log
exec > "$LOG" 2>&1
export PATH=/usr/local/cuda/bin:$PATH

PACK="$HOME/sparkdata/dsv4_pro.tp4pp4/packs/dsv4_pro.tp4_pp4.rank00.spstage"
[[ -s "$PACK" ]] || { echo "MISSING-RANK0-PACK $PACK"; exit 1; }

echo "== build module objects + archive (incremental)"
make -C modules/dsv4_resident_decode_stage -f Makefile.pro archive \
  PRO_EXPERT_CODEC=mxfp4 PRO_KV_CODEC=bf16 \
  STAGE_PACK_PATH="$PACK" \
  STAGE_COUNT=4 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=16 \
  MAX_ACTIVE_SEQUENCES=1024 MAX_SEQUENCE_POSITIONS=33024 \
  PIPELINE_SLOT_COUNT=13 PHYSICAL_PAGE_CAPACITY=1024 \
  LOGICAL_PAGE_CAPACITY=16384 MTP_LAYER_COUNT=3 CUDA_GRAPH_COUNT=0
BUILD_RC=$?
echo "archive-rc=$BUILD_RC"
[[ $BUILD_RC -ne 0 ]] && exit 1

ARCHIVE=build/modules/dsv4_pro_resident_decode_stage/libdsv4_resident_decode_stage.a
echo "== archive sha"
sha256sum "$ARCHIVE"

echo "== validator under bash -x"
CFG=$(printf '%s\n' \
  "SPARK_DSV4_STAGE_PACK_PATH=$PACK" \
  "SPARK_DSV4_STAGE_COUNT=4" \
  "SPARK_DSV4_STAGE_INDEX=0" \
  "SPARK_DSV4_STAGE_FIRST_LAYER=0" \
  "SPARK_DSV4_STAGE_LAYER_COUNT=16" \
  "SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES=1024" \
  "SPARK_DSV4_STAGE_MAX_SEQ=33024" \
  "SPARK_DSV4_STAGE_PIPELINE_SLOTS=13" \
  "SPARK_DSV4_STAGE_PHYSICAL_PAGES=1024" \
  "SPARK_DSV4_STAGE_LOGICAL_PAGES=16384" \
  "SPARK_DSV4_STAGE_MTP=3" \
  "SPARK_DSV4_STAGE_GRAPHS=0" \
  | sha256sum | awk '{print $1}')
echo "config-hash=$CFG"
export SPARK_MODULE_BATCH_BUCKET=1024
export SPARK_DSV4_STAGE_PACK_PATH="$PACK"
export SPARK_DSV4_STAGE_COUNT=4 SPARK_DSV4_STAGE_INDEX=0
export SPARK_DSV4_STAGE_FIRST_LAYER=0 SPARK_DSV4_STAGE_LAYER_COUNT=16
export SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES=1024
export SPARK_DSV4_STAGE_MAX_SEQ=33024 SPARK_DSV4_STAGE_PIPELINE_SLOTS=13
export SPARK_DSV4_STAGE_PHYSICAL_PAGES=1024 SPARK_DSV4_STAGE_LOGICAL_PAGES=16384
export SPARK_DSV4_STAGE_MTP=3 SPARK_DSV4_STAGE_GRAPHS=0
export SPARK_DSV4_VALIDATION_DEFINES="-DSPARK_DSV4_PRO_BUILD=1"
export SPARK_DSV4_CUDA_VALIDATOR_SHA256=$(sha256sum modules/dsv4_resident_decode_stage/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu | awk '{print $1}')
export SPARK_DSV4_REFERENCE_VERIFIER_SHA256=$(sha256sum tools/verify_dsv4_ga_reference_fixture.py | awk '{print $1}')
export SPARK_DSV4_REFERENCE_MANIFEST_SHA256=9ef837975bc4ddbd3cf0de0ea19c59c2c4c8a3750a8b8f302a19df0e09f39fa3
bash -x modules/dsv4_resident_decode_stage/validation/validate_dsv4_resident_decode_stage_cuda.sh \
  "$CFG" "$ARCHIVE"
echo "validator-rc=$?"

echo "== post-failure probes"
VDIR=$(find build -name dsv4_resident_decode_stage_validator -type f 2>/dev/null | head -1)
echo "validator-binary=$VDIR"
if [ -n "$VDIR" ]; then
  strings "$VDIR" | grep -c "gateroute diag" || true
  ls -la "$VDIR"
fi
ls -la build/module_library/link_units/ 2>/dev/null | tail -6
echo DIAG-DONE
