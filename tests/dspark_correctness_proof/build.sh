#!/bin/sh
# DSpark speculative-path correctness proof builder.
# Extracts the VERBATIM kernel bodies from the patched dspark headers and
# compiles them into a host-side CUDA-subset emulator (proof.cpp), then links
# the VERBATIM spark_dsv4_dspark_pro_chain.cuh against emulated launchers and
# an independent double-precision reference model. Runs under ASan+UBSan.
# Usage: tests/dspark_correctness_proof/build.sh [--trials N]
set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC=$REPO/modules/dsv4_resident_decode_stage/source
OUT=$REPO/tmp/dspark_spec_fixes/proof_build
INC=$OUT/extracted_kernels.inc
TRIALS=48
while [ $# -gt 0 ]; do
  case "$1" in
    --trials) TRIALS="$2"; shift 2 ;;
    *) echo "usage: build.sh [--trials N]" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"

extract() {
  awk -v sig="$2" '
    !started && index($0, sig) == 1 { started = 1 }
    started { print; if ($0 == "}") exit }
  ' "$1"
}

: > "$INC"
extract "$SRC/spark_dsv4_dspark_kernels.cuh" \
  "static __global__ void SparkDsv4DsparkAttentionKernel(" >> "$INC"
echo >> "$INC"
extract "$SRC/spark_dsv4_dspark_kernels.cuh" \
  "static __global__ void SparkDsv4DsparkMarkovBiasAccumKernel(" >> "$INC"
echo >> "$INC"
extract "$SRC/spark_dsv4_dspark_kernels.cuh" \
  "static __global__ void SparkDsv4DsparkArgmaxKernel(" >> "$INC"
echo >> "$INC"
extract "$SRC/spark_dsv4_dspark_kernels.cuh" \
  "static __global__ void SparkDsv4DsparkExpandStreamsKernel(" >> "$INC"
echo >> "$INC"
extract "$SRC/spark_dsv4_dspark_pro_kernels.cuh" \
  "static __global__ void SparkDsv4DSparkMeanReductionKernel(" >> "$INC"
echo >> "$INC"
extract "$SRC/spark_dsv4_dspark_pro_kernels.cuh" \
  "static __global__ void SparkDsv4DSparkMainKvWriteKernel(" >> "$INC"
echo >> "$INC"
extract "$SRC/spark_dsv4_dspark_pro_kernels.cuh" \
  "static __global__ void SparkDsv4DSparkConfidenceKernel(" >> "$INC"

# attention's dynamic shared-memory declaration becomes an emulator call;
# everything else stays byte-identical to the patched source.
sed 's|extern __shared__ unsigned char grouped_attention_shared\[\];|unsigned char *grouped_attention_shared = emu::emu_dynamic_smem();|' \
  "$INC" > "$INC.tmp" && mv "$INC.tmp" "$INC"

echo "== extracted $(grep -c "static __global__" "$INC") kernel bodies =="

CXX=${CXX:-c++}
if command -v clang++ >/dev/null 2>&1; then CXX=clang++; fi

MODEL_HDR=$REPO/model-families/dsv4/include/sparkpipe/spark_dsv4_model.h
CHAIN_HDR=$SRC/spark_dsv4_dspark_pro_chain.cuh
printf "#include \"%s\"\n" "$CHAIN_HDR" > "$OUT/probe.cpp"
PROBE_FLAGS="-std=c++17 -E -x c++ -I$REPO/include -I$REPO/model-families/dsv4/include -I$REPO/model-families/common/include -include $MODEL_HDR $OUT/probe.cpp"
OFF=$($CXX $PROBE_FLAGS 2>/dev/null | grep -c SparkDsv4DsparkProChainDraftHead || true)
ON=$($CXX -DSPARK_DSV4_PRO_BUILD=1 $PROBE_FLAGS 2>/dev/null | grep -c SparkDsv4DsparkProChainDraftHead || true)
echo "GUARD_OFF_COUNT=$OFF"
echo "GUARD_ON_COUNT=$ON"

$CXX -std=c++17 -O1 -g -Wall -Wextra -Wno-unused-parameter \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DSPARK_DSV4_PRO_BUILD=1 \
  -I"$REPO/include" \
  -I"$REPO/model-families/dsv4/include" \
  -I"$REPO/model-families/common/include" \
  -include "$MODEL_HDR" \
  -DNDEBUG \
  -I"$OUT" \
  "$REPO/tests/dspark_correctness_proof/proof.cpp" \
  -o "$OUT/dspark_proof"

ASAN_OPTIONS=detect_leaks=0 "$OUT/dspark_proof" --trials "$TRIALS"
