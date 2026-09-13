#!/bin/bash
# Compile-check the rocm.gfx950.mi350p HIP runtime backend against ROCm HIP
# headers. This proves the archive builds against the real API surface without
# needing an AMD GPU or a full ROCm install on the build machine.
#
#   ./compile_check.sh          # vendored ROCm 7.14 host-API headers (default)
#   ./compile_check.sh system   # real ROCm install (/opt/rocm/include, or set
#                               # HIP_HEADERS=/path/to/hip/include)
#   OUT=dir ./compile_check.sh  # object output directory override
#
# On machines WITH ROCm, prefer 'system' mode: it compiles against the exact
# headers that ship with the driver. hipcc builds get __HIP_PLATFORM_AMD__
# automatically; plain cc/clang host compiles must define it explicitly.

set -u

MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$MODULE_DIR/source"
REPO_INCLUDE="$MODULE_DIR/../../include"
OUT="${OUT:-$MODULE_DIR/.compile-check}"

MODE="${1:-vendor}"
case "$MODE" in
  vendor)
    HIP_ROOT="$MODULE_DIR/vendor/hip-headers"
    ;;
  system)
    HIP_ROOT="${HIP_HEADERS:-/opt/rocm/include}"
    ;;
  *)
    echo "usage: $0 [vendor|system]" >&2
    exit 2
    ;;
esac

CC="${CC:-clang}"
FLAGS="-std=c11 -Wall -Wextra -Werror -D__HIP_PLATFORM_AMD__ -I$HIP_ROOT -I$REPO_INCLUDE"

echo "compiler: $CC"
echo "HIP headers: $HIP_ROOT ($MODE mode)"

mkdir -p "$OUT"
rc=0
for src in "$SRC"/spark_hw_rocm_*.c; do
  obj="$OUT/$(basename "${src%.c}").o"
  if "$CC" $FLAGS -c "$src" -o "$obj"; then
    echo "  OK   $(basename "$src")"
  else
    echo "  FAIL $(basename "$src")"
    rc=1
  fi
done

if [ "$rc" -eq 0 ]; then
  echo "COMPILE_CHECK_OK: all sources compiled against $MODE HIP headers."
else
  echo "COMPILE_CHECK_FAILED."
fi
exit "$rc"
