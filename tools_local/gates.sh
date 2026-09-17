#!/bin/sh
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "${repository_root}"

# THE COMPLEXITY CEILING — the merge gate that makes the plan bind:
# the max CCN of the production scope must not exceed the committed
# ceiling in tests/test_complexity_ceiling.py; any increase needs an
# in-commit justification there (the size ratchet's discipline).
python3 tests/test_complexity_ceiling.py

make -j8 test
tools/cuda13_sm121a_compile_gate.sh
python3 tools/verify_package_manifest.py
