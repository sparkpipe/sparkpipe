#!/usr/bin/env bash
set -euo pipefail

# glm5_next (GLM 5.3 Flash) hardware validation driver (sm_121a).
#
# Compiles the validator translation unit against the MODULE ARCHIVE and
# runs it on the pack named by the pinned configuration. The binary runs
# the host oracle selftest first (bounded decay, expert-major codec
# addressing, e4m3, mHC sinkhorn, kpool expansion, and end-to-end
# KDA/MLA/router oracle executions at real geometry), then the GPU tier
# drivers: tier 1 the KDA layer with its dense MLP through both mHC sites,
# tier 2a the DSA (rope-0 MLA) layer with routed experts, each against the
# same fp32 oracle with a bit-exact determinism re-walk. The binary FAILS
# (nonzero) until every wired tier passes; a not-yet-wired tier is a hard
# failure, never a silent pass.

if [[ $# -ne 2 ]]; then
    echo "usage: $0 VALIDATION_CONFIGURATION_SHA256 MODULE_ARCHIVE" >&2
    exit 2
fi
configuration_hash="$1"
module_archive="$2"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_directory="$(cd "${script_directory}/.." && pwd)"
repository_root="$(cd "${module_directory}/../.." && pwd)"
validation_directory="$(mktemp -d)"
cuda_validator="${script_directory}/spark_glm5_next_resident_decode_stage_cuda_validation.cu"
trap 'rm -rf "${validation_directory}"' EXIT

if [[ ! -s "${module_archive}" ]]; then
    echo "module archive is missing or empty: ${module_archive}" >&2
    exit 2
fi
if [[ -z "${SPARK_GLM5_NEXT_STAGE_PACK_PATH:-}" || ! -s "${SPARK_GLM5_NEXT_STAGE_PACK_PATH}" ]]; then
    echo "SPARK_GLM5_NEXT_STAGE_PACK_PATH must name a readable non-empty stage pack" >&2
    exit 2
fi

glm5_next_codecs=(int6 int7 int8 fp8 nvfp4 mxfp4)
glm5_next_codec_ids=(2 3 4 5 6 7)
codec_index=-1
for index in "${!glm5_next_codecs[@]}"; do
    if [[ "${SPARK_GLM5_NEXT_EXPERT_CODEC:-}" == "${glm5_next_codecs[${index}]}" ]]; then
        codec_index="${index}"
    fi
done
if (( codec_index < 0 )); then
    echo "glm5_next hardware validation requires SPARK_GLM5_NEXT_EXPERT_CODEC to name one of: ${glm5_next_codecs[*]}" >&2
    exit 2
fi
if [[ -z "${SPARK_GLM5_NEXT_STAGE_MAX_ACTIVE_SEQUENCES:-}" ]] || (( SPARK_GLM5_NEXT_STAGE_MAX_ACTIVE_SEQUENCES < 1 )); then
    echo "glm5_next hardware validation requires SPARK_GLM5_NEXT_STAGE_MAX_ACTIVE_SEQUENCES >= 1" >&2
    exit 2
fi

nvcc_path="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"
if [[ "${cuda_architecture}" != "sm_121a" ]]; then
    echo "glm5_next hardware validation admits only CUDA_ARCH=sm_121a" >&2
    exit 2
fi
if ! command -v "${nvcc_path}" >/dev/null 2>&1; then
    echo "nvcc unavailable for glm5_next hardware validation" >&2
    exit 2
fi

model_revision="${SPARK_GLM5_NEXT_MODEL_REVISION:-synthesized}"
contract_sha256="${SPARK_GLM5_NEXT_CONTRACT_SHA256:-0000000000000000000000000000000000000000000000000000000000000000}"

make -C "${repository_root}" \
    build/libsparkpipe_core.a \
    build/libsparkpipe_runtime.a

"${nvcc_path}" \
    -std=c++17 \
    -O3 \
    --expt-relaxed-constexpr \
    -gencode arch=compute_121a,code=sm_121a \
    -I"${repository_root}/include" \
    -I"${repository_root}/model-families/glm5_next/include" \
    -I"${module_directory}/include" \
    -I"${module_directory}/source" \
    -DGLM5_NEXT_EXPERT_WEIGHT_CODEC="${glm5_next_codec_ids[${codec_index}]}" \
    -DGLM5_NEXT_EXPERT_CODEC_NAME="\"${SPARK_GLM5_NEXT_EXPERT_CODEC}\"" \
    -DGLM5_NEXT_MODEL_REVISION="\"${model_revision}\"" \
    -DGLM5_NEXT_CONTRACT_SHA256="\"${contract_sha256}\"" \
    "${cuda_validator}" \
    "${module_archive}" \
    "${repository_root}/build/libsparkpipe_runtime.a" \
    "${repository_root}/build/libsparkpipe_core.a" \
    -L"${CUDA_HOME:-/usr/local/cuda}/lib64" \
    -lcuda \
    -lcudart \
    -ldl \
    -lm \
    -Xcompiler -pthread \
    -o "${validation_directory}/glm5_next_resident_decode_stage_validator"

"${validation_directory}/glm5_next_resident_decode_stage_validator" "${configuration_hash}"
