#!/usr/bin/env bash
set -euo pipefail

FAMILY="${1:?module family dir under modules/}"
CODEC="${2:?expert codec (fp8, bf16, nvfp4, ...)}"
ROOT_NAME="${3:?runtime root name under ~/sparkdata and the hub release dir}"
REVISION="${4:?model source revision}"
CONTRACT="${5:?contract json path}"
BRANCH="${6:-${G5_BRANCH:-origin/lane/glm53-p0}}"
TREE="$HOME/sparkpipe-build"
FIRMWARE="examples/model_descriptions/${FAMILY}_${CODEC}_firmware.json"

cd "$TREE"
git fetch -q origin main "${G5_SOURCE_BRANCH:-lane/glm53-p0}" || git fetch -q origin main
if ! git cat-file -e "$BRANCH^{commit}" 2>/dev/null; then
    BRANCH="${G5_SOURCE_BRANCH:-origin/lane/glm53-p0}"
fi
git reset -q --hard "$BRANCH"
git clean -q -fd build "modules/$FAMILY" 2>/dev/null || true
REV=$(git rev-parse --short HEAD)
echo "== $REV"

export PATH="/usr/local/cuda/bin:$PATH"
SHA=$(shasum -a 256 "$CONTRACT" | cut -d' ' -f1)

echo "== host build"
rm -f build/sparkpipe_model_compile build/libhidden_transport_spark_host_rdma_verbs.so build/sparkpipe_weightd
make -j8 build/sparkpipe_model_compile build/sparkpipe_model_residentd build/sparkpipe_model_api build/sparkpipe_weightd build/libhidden_transport_spark_host_rdma_verbs.so
make -C "modules/$FAMILY" adapter EXPERT_CODEC="$CODEC" MODEL_REVISION="$REVISION" CONTRACT_SHA256="$SHA" NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a > /dev/null
ADAPTER_SO="build/modules/$FAMILY/$CODEC/libglm5_next_serving_adapter_$CODEC.so"
[ -f "$ADAPTER_SO" ] || { echo "adapter not built"; exit 1; }
strings build/libhidden_transport_spark_host_rdma_verbs.so | grep QP-WIRE > /dev/null || { echo "DSO stale"; exit 1; }

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
make -C "modules/$FAMILY" publish \
    EXPERT_CODEC="$CODEC" \
    MODEL_REVISION="$REVISION" \
    CONTRACT_SHA256="$SHA" \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a 2>&1 | tail -2

echo "== driver compile"
mkdir -p "$HOME/sparkdata/out"
build/sparkpipe_model_compile \
    --model "$FIRMWARE" \
    --library build/module_library --output "$HOME/sparkdata/out" \
    --cc /usr/bin/cc --include include \
    --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
    --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm \
    --cc-arg -ldl --cc-arg -pthread 2>&1 | tail -1

"$(dirname "$0")/publish_local.sh" "$FAMILY" "$CODEC" "$ROOT_NAME"
systemctl --user start fleet-agent 2>/dev/null || true
exec "$(dirname "$0")/publish_core.sh"
