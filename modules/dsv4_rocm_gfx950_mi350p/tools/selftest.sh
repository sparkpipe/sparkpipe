#!/bin/bash
# Selftest for the rocm.gfx950.mi350p archive's host-runnable logic.
# Builds and runs tests/test_spark_hw_rocm_pure.c (status mapping +
# SparkHwStatusToString) against the vendored HIP headers - no GPU, no ROCm
# install, no HIP library at link time because nothing here calls into it.
#
# Run:  tools/selftest.sh          # vendored headers (default)
#       HIP_HEADERS=dir tools/selftest.sh   # real ROCm headers instead

set -u

MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_INCLUDE="$MODULE_DIR/../../include"
HIP_ROOT="${HIP_HEADERS:-$MODULE_DIR/vendor/hip-headers}"
OUT="${OUT:-$MODULE_DIR/.selftest}"

CC="${CC:-clang}"
FLAGS="-std=c11 -Wall -Wextra -Werror -D__HIP_PLATFORM_AMD__ -I$HIP_ROOT -I$REPO_INCLUDE -I$MODULE_DIR/source"

echo "compiler: $CC"
echo "HIP headers: $HIP_ROOT"

mkdir -p "$OUT"

if ! "$CC" $FLAGS "$MODULE_DIR/tests/test_spark_hw_rocm_pure.c" \
        "$MODULE_DIR/source/spark_hw_rocm_status_string.c" -o "$OUT/test_spark_hw_rocm_pure"; then
  echo "SELFTEST_BUILD_FAILED"
  exit 1
fi

if "$OUT/test_spark_hw_rocm_pure"; then
  echo "SELFTEST_OK"
  exit 0
fi
echo "SELFTEST_FAILED"
exit 1

