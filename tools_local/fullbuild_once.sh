#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/cuda/bin:$PATH"
cd ~/sparkpipe-build
make -j8 build/sparkpipe_model_compile build/sparkpipe_model_residentd \
    build/sparkpipe_model_api build/sparkpipe_weightd
SHA=$(shasum -a 256 model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1)
make -C modules/glm5_next_resident_decode_stage adapter \
    EXPERT_CODEC=fp8 MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 \
    CONTRACT_SHA256="$SHA" NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a
make -C modules/glm5_next_resident_decode_stage publish \
    EXPERT_CODEC=fp8 MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 \
    CONTRACT_SHA256="$SHA" NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a
build/sparkpipe_model_compile \
    --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json \
    --library build/module_library --output /home/sparkf/sparkdata/out \
    --cc /usr/bin/cc --include include \
    --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
    --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm \
    --cc-arg -ldl --cc-arg -pthread
bash tools/publish_local.sh glm5_next_resident_decode_stage fp8 glm53flash.fp8.tp16
cp build/sparkpipe_weightd /tmp/weightd.new
echo FULLBUILD-DONE
