#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" != "0" ]]; then
    echo "run as root" >&2
    exit 2
fi
if [[ ! -r /etc/os-release ]]; then
    echo "cannot identify the operating system" >&2
    exit 2
fi
. /etc/os-release
if [[ "${ID:-}" != "debian" || "${VERSION_ID:-}" != "13" ]]; then
    echo "this installer is restricted to Debian 13" >&2
    exit 2
fi

repository_base="https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64"
work_directory="$(mktemp -d)"
trap 'rm -rf "${work_directory}"' EXIT

curl --fail --location --retry 5 \
    "${repository_base}/cuda-keyring_1.1-1_all.deb" \
    --output "${work_directory}/cuda-keyring.deb"
dpkg -i "${work_directory}/cuda-keyring.deb"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    cuda-compiler-13-3 \
    cuda-cudart-dev-13-3 \
    cuda-nvdisasm-13-3 \
    cuda-cuobjdump-13-3 \
    libcublas-dev-13-3 \
    libibverbs-dev

ln -sfn /usr/local/cuda-13.3 /usr/local/cuda
/usr/local/cuda/bin/nvcc --version
