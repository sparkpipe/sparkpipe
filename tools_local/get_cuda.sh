#!/bin/sh
set -eu

prefix=${1:-/opt/cuda}
release=${CUDA_REDISTRIBUTABLE_RELEASE:-13.3.1}
base_url=https://developer.download.nvidia.com/compute/cuda/redist
work_directory=$(mktemp -d)
trap 'rm -rf "$work_directory"' EXIT HUP INT TERM
manifest_path="$work_directory/redistrib.json"

curl --fail --location --retry 5 --retry-all-errors \
    "$base_url/redistrib_${release}.json" \
    --output "$manifest_path"

for component_name in \
    cuda_nvcc \
    cuda_cudart \
    cuda_cccl \
    cuda_crt \
    cuda_cuobjdump \
    cuda_nvdisasm \
    libnvvm \
    libnvptxcompiler \
    libcublas
do
    component_record=$(python3 - "$manifest_path" "$component_name" <<'PY'
import json
import sys

manifest_path, component_name = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
component = manifest.get(component_name, {}).get("linux-x86_64")
if component is None:
    raise SystemExit(f"missing linux-x86_64 CUDA component: {component_name}")
print(component["relative_path"])
print(component["sha256"])
PY
)
    relative_path=$(printf '%s\n' "$component_record" | sed -n '1p')
    expected_sha256=$(printf '%s\n' "$component_record" | sed -n '2p')
    archive_path="$work_directory/$(basename "$relative_path")"
    curl --fail --location --retry 5 --retry-all-errors \
        "$base_url/$relative_path" \
        --output "$archive_path"
    actual_sha256=$(sha256sum "$archive_path" | awk '{print $1}')
    if [ "$actual_sha256" != "$expected_sha256" ]
    then
        echo "CUDA component checksum mismatch: $component_name" >&2
        exit 1
    fi
    tar -xf "$archive_path" -C "$work_directory"
done

install_directory="$work_directory/install"
mkdir -p "$install_directory"
for archive_directory in "$work_directory"/*-archive
do
    if [ -d "$archive_directory" ]
    then
        cp -a "$archive_directory"/. "$install_directory"/
    fi
done

mkdir -p "$prefix"
cp -a "$install_directory"/. "$prefix"/

"$prefix/bin/nvcc" --version
"$prefix/bin/ptxas" --version

probe_source="$work_directory/sm121a_probe.cu"
probe_object="$work_directory/sm121a_probe.o"
printf '%s\n' \
    '__global__ void SparkSm121aProbe(float *output) { output[threadIdx.x] = 1.0f; }' \
    > "$probe_source"
"$prefix/bin/nvcc" \
    -std=c++17 \
    -gencode arch=compute_121a,code=sm_121a \
    -c "$probe_source" \
    -o "$probe_object"
