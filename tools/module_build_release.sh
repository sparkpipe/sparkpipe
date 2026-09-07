#!/usr/bin/env bash
set -euo pipefail

FAMILY="${1:?module family dir under modules/}"
CODEC="${2:?expert codec (fp8, bf16, nvfp4, ...)}"
ROOT_NAME="${3:?runtime root name under ~/sparkdata and the hub release dir}"
REVISION="${4:?model source revision}"
CONTRACT="${5:?contract json path}"
BRANCH="${6:-${G5_BRANCH:-lane/glm53-tree-2}}"
REMOTE="${7:-origin}"
TREE="$HOME/sparkpipe-build"
HUB_REF="${HUB_REF:-rtx5090:release}"
FIRMWARE="examples/model_descriptions/${FAMILY}_${CODEC}_firmware.json"

cd "$TREE"
git fetch -q "$REMOTE" "${G5_SOURCE_BRANCH:-lane/glm53-tree-2}"
if ! git cat-file -e "$BRANCH^{commit}" 2>/dev/null; then
    BRANCH="${G5_SOURCE_BRANCH:-lane/glm53-tree-2}"
fi
git reset -q --hard "$BRANCH"
git clean -q -fd build "modules/$FAMILY" 2>/dev/null || true
REV=$(git rev-parse --short HEAD)
echo "== $REV"

export PATH="/usr/local/cuda/bin:$PATH"
SHA=$(shasum -a 256 "$CONTRACT" | cut -d' ' -f1)

echo "== host build"
rm -f build/sparkpipe_model_compile build/libhidden_transport_spark_host_rdma_verbs.so
make -j8 build/sparkpipe_model_compile build/sparkpipe_model_residentd build/sparkpipe_model_api build/libhidden_transport_spark_host_rdma_verbs.so
strings build/libhidden_transport_spark_host_rdma_verbs.so | grep -q QP-WIRE || { echo "DSO stale"; exit 1; }

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

echo "== install into hub reference"
ssh -o BatchMode=yes "${HUB_REF%%:*}" "mkdir -p '${HUB_REF#*:}/$ROOT_NAME/stages/stage_000' '${HUB_REF#*:}/$ROOT_NAME/lib'"
rsync -c "$HOME/sparkdata/out/stages/stage_000/model_driver.so" \
    "${HUB_REF}/$ROOT_NAME/stages/stage_000/model_driver.so"
rsync -c build/libhidden_transport_spark_host_rdma_verbs.so \
    "${HUB_REF}/$ROOT_NAME/lib/hidden_transport.so"
ssh -o BatchMode=yes "${HUB_REF%%:*}" "rm -f /srv/qpn/*.rec; touch '${HUB_REF#*:}/$ROOT_NAME/UPDATE'"
systemctl --user start fleet-agent 2>/dev/null || true
echo "released $REV"
