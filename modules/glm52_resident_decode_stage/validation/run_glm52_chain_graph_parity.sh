#!/usr/bin/env bash
set -euo pipefail
codec="${1:-fp8}"
bucket="${2:-16}"
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
module="modules/glm52_resident_decode_stage"
declare -A codec_ids=([bf16]=1 [int6]=2 [int7]=3 [int8]=4 [fp8]=5 [nvfp4]=6 [mxfp4]=7)
codec_id="${codec_ids[$codec]:?unknown codec $codec}"
read -r revision contract < <(python3 "$repository_root/tools/glm52_model_contract.py" --print-build-identity "$codec")
archive="$repository_root/build/modules/glm52_resident_decode_stage/$codec/libglm52_resident_decode_stage_${codec}_b${bucket}.a"
cd "$repository_root"
make -C "$module" -j8 CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" CUDA_ARCH="${CUDA_ARCH:-sm_121a}" EXPERT_CODEC="$codec" MODEL_REVISION="$revision" CONTRACT_SHA256="$contract" MODULE_BATCH_VARIANT_BUCKETS="$bucket" EXECUTION_ROW_CAPACITY="$bucket" variants
make -j8 build/libsparkpipe_core.a build/libsparkpipe_runtime.a
"${NVCC:-nvcc}" -std=c++17 -O3 --expt-relaxed-constexpr -gencode "arch=compute_121a,code=sm_121a" \
    -Iinclude -Imodel-families/glm52/include -I. -I"$module/include" -I"$module/source" \
    -DSPARK_BATCH_BUCKET="$bucket" -DGLM_EXPERT_WEIGHT_CODEC="$codec_id" -DGLM_EXPERT_CODEC_NAME="\"$codec\"" \
    -DGLM_MODEL_REVISION="\"$revision\"" -DGLM_CONTRACT_SHA256="\"$contract\"" \
    "$module/validation/glm52_chain_graph_parity.cu" "$archive" build/libsparkpipe_runtime.a build/libsparkpipe_core.a \
    -L"${CUDA_HOME:-/usr/local/cuda}/lib64" -lcuda -lcudart -ldl -lm -Xcompiler -pthread -o build/glm52_chain_graph_parity
./build/glm52_chain_graph_parity
