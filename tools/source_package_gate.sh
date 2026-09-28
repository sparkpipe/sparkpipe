#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
revision=${1:-HEAD}
package_directory=$(mktemp -d)
trap 'rm -rf "${package_directory}"' EXIT INT TERM

# CI checks the repository out as a different user inside the CUDA container;
# archive is read-only, so trust this checkout for this one command.
git -c safe.directory="${repository_root}" -C "${repository_root}" archive --format=tar "${revision}" | tar -xf - -C "${package_directory}"
python3 "${package_directory}/tools/generate_package_manifest.py" >/dev/null
python3 "${package_directory}/tools/generate_sha256sums.py" >/dev/null
python3 "${package_directory}/tools/verify_package_manifest.py"
