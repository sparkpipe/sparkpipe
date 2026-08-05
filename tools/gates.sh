#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "${repository_root}"

make -j8 test
tools/cuda13_sm121a_compile_gate.sh
python3 tools/verify_package_manifest.py
