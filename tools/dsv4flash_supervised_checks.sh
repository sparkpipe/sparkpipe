#!/usr/bin/env bash
# dsv4flash: supervised-startup host checks (main's own tests) — the
# deployment contract the serving smoke will run under. CPU job.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== test_weightd_supervised"
python3 tests/test_weightd_supervised.py
echo "== test_model_resident_deadline (build + run)"
make -j4 build/test_model_resident_deadline
./build/test_model_resident_deadline
echo "SUPERVISED-CHECKS-PASS"
