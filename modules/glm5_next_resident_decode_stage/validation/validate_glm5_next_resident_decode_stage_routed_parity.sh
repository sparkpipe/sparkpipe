#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 <bf16|fp8|nvfp4|int8|int6|int7|mxfp4> <pack.sp> <out-dir> [ceiling-dump-dir]" >&2
    echo "env: ROUTED_PARITY_LAYERS (default 3,23,44), ROUTED_PARITY_ROWS (default 1,9,16,64,256), ROUTED_PARITY_SEED (default 20260929)" >&2
    exit 2
fi
codec=$1
pack=$2
out=$3
ceiling=${4:-}
declare -A codec_ids=([bf16]=1 [int6]=2 [int7]=3 [int8]=4 [fp8]=5 [nvfp4]=6 [mxfp4]=7)
codec_id=${codec_ids[$codec]:-}
[ -n "$codec_id" ] || { echo "unknown codec $codec" >&2; exit 2; }
[ -f "$pack" ] || { echo "pack not found: $pack" >&2; exit 2; }
layers=${ROUTED_PARITY_LAYERS:-3,23,44}
rows=${ROUTED_PARITY_ROWS:-1,9,16,64,256}
seed=${ROUTED_PARITY_SEED:-20260929}
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
module="$repository_root/modules/glm5_next_resident_decode_stage"
cuda_arch=${CUDA_ARCH:-sm_121a}
nvcc=${NVCC:-nvcc}
mkdir -p "$out"
binary="$out/routed_parity_$codec"
"$nvcc" -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo -gencode "arch=${cuda_arch/sm_/compute_},code=$cuda_arch" \
    -I"$repository_root" -I"$repository_root/include" -I"$repository_root/model-families/common/include" \
    -I"$repository_root/model-families/glm5_next/include" -I"$module/include" -I"$module/source" \
    -include "$repository_root/model-families/glm5_next/include/sparkpipe/spark_glm5_next_model.h" \
    -DSPARK_BATCH_BUCKET=1024u -DGLM5_NEXT_EXPERT_WEIGHT_CODEC="$codec_id" -DGLM5_NEXT_EXPERT_CODEC_NAME="\"$codec\"" \
    -DGLM5_NEXT_MODEL_REVISION="\"routed-parity\"" -DGLM5_NEXT_CONTRACT_SHA256="\"routed-parity\"" \
    -Xcompiler -Wall,-Wextra \
    "$module/validation/spark_glm5_next_resident_decode_stage_routed_parity.cu" -o "$binary" -lcuda
"$binary" "$pack" "$layers" "$rows" "$seed" "$out/dump"
oracle_args=("$out/dump" --receipt "$out/receipt.json")
if [ -n "$ceiling" ]; then
    oracle_args+=(--ceiling "$ceiling")
fi
python3 "$repository_root/tools/glm5_next_routed_oracle.py" "${oracle_args[@]}"
