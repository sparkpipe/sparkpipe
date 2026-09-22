#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 5 ]; then
    printf '%s\n' 'usage: module_build_release.sh FAMILY CODEC OUTPUT_NAME MODEL_REVISION CONTRACT; build from the queue-synced checkout, no branch argument' >&2
    exit 2
fi
: "${SPARK_QUEUE_ID:?run inside a GPU-owned spark_queue job}"
family=$1
codec=$2
name=$3
revision=$4
contract=$5
for value in "$family" "$codec" "$name"; do
    [[ "$value" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || exit 2
done
cd "$(dirname "$0")/.."
git diff --quiet HEAD || { printf '%s\n' 'tracked build inputs differ from HEAD' >&2; exit 2; }
source_commit=$(git rev-parse HEAD)
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
if [ -d build/obj ] || [ -d build/modules ]; then
    printf '%s\n' 'use a fresh queue-synced checkout; existing objects can carry a different contract' >&2
    exit 2
fi
mkdir -p build
exec 9>build/.firmware-build.lock
flock -n 9 || { printf '%s\n' 'another build owns this checkout' >&2; exit 2; }
output="build/$name"
[ ! -e "$output" ] || { printf '%s\n' "output already exists: $output" >&2; exit 2; }
mkdir -p "$output/qualification/serving-receipts"
receipts="$output/qualification/serving-receipts"
exec 3>&1
exec > "$receipts/build.log" 2>&1
trap 'code=$?; if [ "$code" -ne 0 ]; then tail -30 "$receipts/build.log" >&3; fi' EXIT
printf '%s\n' "$source_commit" > "$receipts/source-commit.txt"
sha256sum "$contract" > "$receipts/contract.sha256"
contract_sha=$(cut -d' ' -f1 "$receipts/contract.sha256")
export CUDA_MODULE_LOADING=LAZY CUDA_MODULE_DATA_LOADING=LAZY CUDA_DEVICE_MAX_CONNECTIONS=32
export PATH="/usr/local/cuda/bin:$PATH"
make -j4 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a build/weightd_warm build/sparkpipe_model_residentd build/sparkpipe_weightd build/sparkpipe_model_api build/sparkpipe_model_batch build/sparkpipe_model_compile build/sparkpipe_module_publish build/sparkpipe_driver_inspect hidden_transport_spark_host_rdma_verbs
nvcc --version > "$receipts/toolchain.txt"
cc --version >> "$receipts/toolchain.txt"
make -j4 -C "modules/$family" CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a EXPERT_CODEC="$codec" MODEL_REVISION="$revision" CONTRACT_SHA256="$contract_sha" archive adapter
prefix=${family%_resident_decode_stage}
adapter="build/modules/$family/$codec/lib${prefix}_serving_adapter_${codec}.so"
[ -f "$adapter" ] || { printf '%s\n' "adapter not built: $adapter" >&2; exit 1; }
env_prefix="SPARK_$(printf '%s' "$prefix" | tr '[:lower:]' '[:upper:]')"
export "${env_prefix}_MODEL_REVISION=$revision" "${env_prefix}_CONTRACT_SHA256=$contract_sha"
make -j4 -C "modules/$family" CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a EXPERT_CODEC="$codec" MODEL_REVISION="$revision" CONTRACT_SHA256="$contract_sha" publish > "$receipts/publish.log" 2>&1
firmware="${FIRMWARE_JSON:-examples/model_descriptions/${family}_${codec}_firmware.json}"
# Static archives come before the -l libs: module archives reference
# runtime (stagepack format) and core (sha256/ck128, pulled into every
# module archive by the spine DAEMON_SHA change) symbols that only
# libsparkpipe_runtime.a / libsparkpipe_core.a define - without them the
# driver link fails undefined on every family's release build (evidence:
# gemma4-fleet-build-3, 16/16 nodes; same cluster as #1127).
build/sparkpipe_model_compile --model "$firmware" --library build/module_library --output "$output/compiled" --cc /usr/bin/cc --include include --cc-arg build/libsparkpipe_runtime.a --cc-arg build/libsparkpipe_core.a --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl --cc-arg -pthread > "$receipts/driver-link.log" 2>&1
target=$(python3 -c 'import json,sys; v=json.load(open(sys.argv[1])); assert len(v["stages"])==1; print(v["stages"][0]["target"])' "$firmware")
build/sparkpipe_driver_inspect "$output/compiled/stages/stage_000/model_driver.so" "$target" > "$receipts/driver-inspect.log" 2>&1
ldd -r "$output/compiled/stages/stage_000/model_driver.so" > "$receipts/driver-dependencies.log" 2>&1
if grep -Eq 'not found|undefined symbol' "$receipts/driver-dependencies.log"; then cat "$receipts/driver-dependencies.log"; exit 1; fi
mkdir -p "$output/bin" "$output/lib"
cp build/weightd_warm build/sparkpipe_model_residentd build/sparkpipe_weightd build/sparkpipe_model_api build/sparkpipe_model_batch build/sparkpipe_model_compile "$output/bin/"
cp build/libhidden_transport_spark_host_rdma_verbs.so "$output/lib/hidden_transport.so"
cp "$adapter" "$output/lib/model_serving_adapter.so"
mv "$output/compiled/stages" "$output/compiled/model_package.json" "$output/"
rmdir "$output/compiled"
cp -a build/module_library/active "$output/qualification/module-records"
cp "$firmware" "$contract" "$output/qualification/"
git diff --quiet HEAD || { printf '%s\n' 'tracked build inputs changed during build' >&2; exit 1; }
[ "$(git rev-parse HEAD)" = "$source_commit" ] || exit 1
printf '%s\n' "$source_commit" > "$output/SOURCE_COMMIT"
exec 1>&3 2>&1
(cd "$output" && find bin lib stages qualification model_package.json SOURCE_COMMIT -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar -czf "$output.tar.gz" -C "$output" .
sha256sum "$output.tar.gz" > "$output.tar.gz.sha256"
printf 'BUILD-PASS %s\n' "$output.tar.gz" >&3
