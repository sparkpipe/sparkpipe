#!/usr/bin/env bash
# dsv4_flash build + release from the repo. Runs ON a spark (aarch64 + GB10).
# Any node can recreate the tree: git pull, publish (GPU receipts), compile
# the driver, install into the hub reference, drop UPDATE.
#
# PAIRING LAW (cell1 saga root cause #4): the tp16 serving adapter embeds the
# UNBUCKETED contract sha (SPARK_BATCH_BUCKET defaults 1024 -> a9bb380e...),
# so the driver MUST be compiled from the DEFAULT firmware description
# (dsv4_resident_decode_stage_firmware.json) against the DEFAULT archive
# publish. A firmware_b1-compiled driver fails target_mismatch on every rank.
set -euo pipefail

BRANCH="${1:-lane/dsv4flash-dev}"
ROOT="${2:-origin}"
TREE="$HOME/sparkpipe-build"
HUB_REF="${DSV4_HUB_REF:-rtx5090:release}"
NAME="dsv4flash.fp8.tp16"
VALIDATION_PACK="${DSV4_VALIDATION_PACK:-$HOME/sparkdata/dsv4flash.tp16/packs/dsv4flash.tp16.rank0.spstage}"

cd "$TREE"
git fetch -q "$ROOT" "$BRANCH"
git reset -q --hard FETCH_HEAD
git clean -q -fd build modules/dsv4_resident_decode_stage 2>/dev/null || true
REV=$(git rev-parse --short HEAD)
echo "== $REV"

export PATH="/usr/local/cuda/bin:$PATH"

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

echo "== module publish (GPU receipts; default archive = the tp16 pairing)"
make -C modules/dsv4_resident_decode_stage publish \
    STAGE_PACK_PATH="$VALIDATION_PACK" \
    STAGE_COUNT=16 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=43 \
    MAX_ACTIVE_SEQUENCES=8 PIPELINE_SLOT_COUNT=1 \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a 2>&1 | tail -2

echo "== driver compile (default firmware description)"
mkdir -p "$HOME/sparkdata/out"
build/sparkpipe_model_compile \
    --model examples/model_descriptions/dsv4_resident_decode_stage_firmware.json \
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
