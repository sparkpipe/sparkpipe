#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-lane/glm53-tree-2}"
ROOT="${2:-origin}"
TREE="$HOME/sparkpipe-build"
HUB_REF="${G5_HUB_REF:-rtx5090:release}"
NAME="glm53flash.fp8.tp16"

cd "$TREE"
git fetch -q "$ROOT" "$BRANCH"
git reset -q --hard FETCH_HEAD
git clean -q -fd build modules/glm5_next_resident_decode_stage 2>/dev/null || true
REV=$(git rev-parse --short HEAD)
echo "== $REV"

export PATH="/usr/local/cuda/bin:$PATH"
SHA=$(shasum -a 256 model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1)

echo "== host build"
make -q build/sparkpipe_model_compile || make -j8 build/sparkpipe_model_compile build/sparkpipe_model_residentd build/sparkpipe_model_api

echo "== park local agent + daemon (validator needs the GPU; UPDATE restores the fleet)"
systemctl --user stop fleet-agent 2>/dev/null || true
for l in $(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model_residentd | sed 's|.*/proc/\([0-9]*\)/exe.*|\1|'); do
    kill -TERM "$l" 2>/dev/null || true
done
for t in $(seq 1 15); do
    ls -l /proc/[0-9]*/exe 2>/dev/null | grep -q sparkpipe_model_residentd || break
    sleep 1
done

echo "== module publish (GPU receipts)"
make -C modules/glm5_next_resident_decode_stage publish \
    EXPERT_CODEC=fp8 \
    MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 \
    CONTRACT_SHA256="$SHA" \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a 2>&1 | tail -2

echo "== driver compile"
mkdir -p "$HOME/sparkdata/out"
build/sparkpipe_model_compile \
    --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json \
    --library build/module_library --output "$HOME/sparkdata/out" \
    --cc /usr/bin/cc --include include \
    --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
    --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm \
    --cc-arg -ldl --cc-arg -pthread 2>&1 | tail -1

echo "== install into hub reference"
ssh -o BatchMode=yes "${HUB_REF%%:*}" "mkdir -p '${HUB_REF#*:}/$NAME/stages/stage_000'"
rsync -c "$HOME/sparkdata/out/stages/stage_000/model_driver.so" \
    "${HUB_REF}/$NAME/stages/stage_000/model_driver.so"
ssh -o BatchMode=yes "${HUB_REF%%:*}" "touch '${HUB_REF#*:}/$NAME/UPDATE'"
systemctl --user start fleet-agent 2>/dev/null || true
echo "released $REV"
