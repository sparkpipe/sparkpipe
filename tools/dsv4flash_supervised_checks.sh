#!/usr/bin/env bash
# dsv4flash: supervised-startup host checks (main's own tests) — the
# deployment contract the serving smoke will run under. CPU job.
# The checkout path is lane-pinned (queue --cwd matches it).
set -euo pipefail
cd /home/spark2/lane-dsv4flash-build
echo "== test_weightd_supervised"
python3 tests/test_weightd_supervised.py
echo "== test_model_resident_deadline (build + run)"
make -j4 build/test_model_resident_deadline
./build/test_model_resident_deadline
echo "SUPERVISED-CHECKS-PASS"
