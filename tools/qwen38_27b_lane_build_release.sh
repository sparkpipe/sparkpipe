#!/usr/bin/env bash
# Qwen 3.8 27B lane-1 firmware build: coherent verified bundle for the
# shared-socket lane wrapper (tools/qwen38_27b_lane_launch.sh).
#
# This family builds its serving adapter at the repository top level
# (build/libqwen38_27b_serving_adapter.so with the TP4 serving constants)
# and its module through the shared resident_decode_stage rules, so the
# generic module_build_release.sh adapter step does not apply; this script
# is the family-specific equivalent of tools/glm5_next_build_release.sh,
# with the same rigor: run inside a GPU-owned queue job, refuse a dirty
# tracked tree or preexisting outputs, nonblocking build lock, receipts,
# SOURCE_COMMIT + relative SHA256SUMS + tar.gz.
#
# Usage (inside the queue-synced checkout, on a lane node with the packs):
#   QWEN38_27B_LANE_PACK=/home/<host>/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rank00.q38sp \
#   bash tools/qwen38_27b_lane_build_release.sh
set -euo pipefail

: "${SPARK_QUEUE_ID:?run inside a GPU-owned spark_queue job (module publish executes device validation)}"
PACK="${QWEN38_27B_LANE_PACK:-/home/$(hostname | sed 's/-.*//')/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rank00.q38sp}"
OUTPUT_NAME="${QWEN38_27B_LANE_BUILD_NAME:-qwen38_27b_lane1}"
cd "$(dirname -- "$0")/.."

git diff --quiet HEAD || { printf '%s\n' 'tracked build inputs differ from HEAD' >&2; exit 2; }
source_commit=$(git rev-parse HEAD)
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
[ -r "$PACK" ] || { printf '%s\n' "stage pack missing: $PACK" >&2; exit 2; }
[ -f "$PACK.sha256" ] || { printf '%s\n' "pack digest sidecar missing (generate sidecars first): $PACK.sha256" >&2; exit 2; }
[ -f "$PACK.experts" ] || { printf '%s\n' "pack experts manifest missing: $PACK.experts" >&2; exit 2; }
if [ -d build/obj ] || [ -d build/modules ] || [ -e "build/$OUTPUT_NAME" ]; then
    printf '%s\n' 'use a fresh queue-synced checkout; existing objects or output can carry a different contract' >&2
    exit 2
fi
mkdir -p build
exec 9>build/.firmware-build.lock
flock -n 9 || { printf '%s\n' 'another build owns this checkout' >&2; exit 2; }
output="build/$OUTPUT_NAME"
receipts="$output/qualification/serving-receipts"
mkdir -p "$receipts"
exec 3>&1
exec > "$receipts/build.log" 2>&1
trap 'code=$?; if [ "$code" -ne 0 ]; then tail -40 "$receipts/build.log" >&3; fi' EXIT
printf '%s\n' "$source_commit" > "$receipts/source-commit.txt"
printf '%s\n' "$PACK" > "$receipts/stage-pack.txt"
( cd "$(dirname "$PACK")" && sha256sum "$(basename "$PACK")" ) > "$receipts/stage-pack.sha256"
export CUDA_MODULE_LOADING=LAZY CUDA_MODULE_DATA_LOADING=LAZY CUDA_DEVICE_MAX_CONNECTIONS=32
export PATH="/usr/local/cuda/bin:$PATH"

# Host services, transport, adapter (top-level target carries the TP4
# serving constants and the firmware-json contract sha).
make -j4 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
    build/sparkpipe_weightd build/weightd_warm build/sparkpipe_model_residentd \
    build/sparkpipe_model_api build/sparkpipe_model_batch \
    build/sparkpipe_model_compile build/sparkpipe_module_publish \
    build/sparkpipe_driver_inspect build/libqwen38_27b_serving_adapter.so \
    hidden_transport_spark_host_rdma_verbs
nvcc --version > "$receipts/toolchain.txt"
cc --version >> "$receipts/toolchain.txt"

# Module archive + device validation + library publication against the real
# rank pack (whole-stack 64-layer TP slice, TP4 lane shape).
# Whole-stack TP tier per the validator contract: rank 0 in STANDALONE
# collective mode (consistency + determinism gate here; cross-rank
# numerics gate at the lane E2E run), unqualified execution admitted by
# the retained-receipt validator itself. The stage module refuses direct
# pack loads outright (shared-node law), so the module tier attaches to a
# PRIVATE weightd inside this job cgroup - the qualified single-node
# validation shape; it never touches the node's shared daemon.
pack_sha=$(cut -d' ' -f1 "$PACK.sha256")
# The validation rank must match the pack's own TP shard (header u32 at
# offset 100): rank0N packs validate as rank N.
validate_tp_rank=$(python3 -c 'import struct,sys; print(struct.unpack_from("<I", open(sys.argv[1],"rb").read(120), 100)[0])' "$PACK")
# sockaddr_un caps the path at 108 bytes and the queue checkout path is
# long: keep the private validation socket under /tmp instead.
weightd_socket="/tmp/qwen38-27b-lane-build-weightd-${SPARK_QUEUE_ATTEMPT:-$$}.sock"
rm -f "$weightd_socket"
# Private in-job daemon: unique latch port (the 61900 default can collide
# with another lane's build weightd on the node); build-validation only.
SPARK_WEIGHTD_LATCH_PORT=31901 \
build/sparkpipe_weightd --socket "$weightd_socket" \
    --device-bytes-max $((12800 * 1024 * 1024)) \
    >"$receipts/build-weightd.log" 2>&1 &
weightd_pid=$!
for _ in $(seq 1 300); do
    grep -q "spark_weightd ready" "$receipts/build-weightd.log" 2>/dev/null && break
    kill -0 "$weightd_pid" 2>/dev/null || { cat "$receipts/build-weightd.log" >&2; exit 2; }
    sleep 0.2
done
grep -q "spark_weightd ready" "$receipts/build-weightd.log" || { echo "private build weightd not ready" >&2; exit 2; }
cleanup_weightd() { kill -TERM "$weightd_pid" 2>/dev/null || true; wait "$weightd_pid" 2>/dev/null || true; }
trap 'cleanup_weightd; code=$?; if [ "$code" -ne 0 ]; then tail -40 "$receipts/build.log" >&3; fi' EXIT
SPARK_WEIGHTD_ATTACH=1 \
SPARK_WEIGHTD_SOCKET="$weightd_socket" \
SPARK_WEIGHTD_PACK_SHA256="$pack_sha" \
make -j4 -C modules/qwen38_27b_resident_decode_stage \
    CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
    STAGE_PACK_PATH="$(cd "$(dirname "$PACK")" && pwd)/$(basename "$PACK")" \
    ALLOW_UNQUALIFIED_EXECUTION=1 TP_DEGREE=4 TP_RANK=$validate_tp_rank TP_STANDALONE=1 \
    STAGE_COUNT=1 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=64 \
    MAX_ACTIVE_SEQUENCES=16 KV_BLOCK_COUNT=256 \
    publish > "$receipts/module-publish.log" 2>&1
publish_status=$?
cleanup_weightd
trap 'code=$?; if [ "$code" -ne 0 ]; then tail -40 "$receipts/build.log" >&3; fi' EXIT
[ "$publish_status" -eq 0 ] || exit "$publish_status"

# Driver link against the published module library.
build/sparkpipe_model_compile \
    --model examples/model_descriptions/qwen38_27b_resident_decode_stage_firmware.json \
    --library build/module_library --output "$output/compiled" \
    --cc /usr/bin/cc --include include \
    --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
    --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ \
    --cc-arg -lm --cc-arg -ldl --cc-arg -pthread \
    > "$receipts/driver-link.log" 2>&1
build/sparkpipe_driver_inspect "$output/compiled/stages/stage_000/model_driver.so" \
    cuda.sm121.qwen38_27b.resident_decode_stage.bf16 \
    > "$receipts/driver-inspect.log" 2>&1
ldd -r "$output/compiled/stages/stage_000/model_driver.so" > "$receipts/driver-dependencies.log" 2>&1
if grep -Eq 'not found|undefined symbol' "$receipts/driver-dependencies.log"; then
    cat "$receipts/driver-dependencies.log"; exit 1
fi

mkdir -p "$output/bin" "$output/lib"
cp build/weightd_warm build/sparkpipe_model_residentd build/sparkpipe_model_api \
    build/sparkpipe_model_batch "$output/bin/"
cp build/libhidden_transport_spark_host_rdma_verbs.so "$output/lib/hidden_transport.so"
cp build/libqwen38_27b_serving_adapter.so "$output/lib/model_serving_adapter.so"
mv "$output/compiled/stages" "$output/compiled/model_package.json" "$output/"
rmdir "$output/compiled"
cp -a build/module_library/active "$output/qualification/module-records"
cp examples/model_descriptions/qwen38_27b_resident_decode_stage_firmware.json \
   model_contracts/qwen38_27b_authoritative.json "$output/qualification/"
git diff --quiet HEAD || { printf '%s\n' 'tracked build inputs changed during build' >&2; exit 1; }
[ "$(git rev-parse HEAD)" = "$source_commit" ] || exit 1
printf '%s\n' "$source_commit" > "$output/SOURCE_COMMIT"
exec 1>&3 2>&1
(cd "$output" && find bin lib stages qualification model_package.json SOURCE_COMMIT -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -czf "$output.tar.gz" -C "$output" .
sha256sum "$output.tar.gz" > "$output.tar.gz.sha256"
printf 'BUILD-PASS %s (pack %s)\n' "$output.tar.gz" "$(basename "$PACK")" >&3
