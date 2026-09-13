#!/bin/bash
# Compile-check the S7 step-1 island entries (E0 prologue.embed,
# L5 layer.moe_shared) of the rocm.gfx950.mi350p archive.
#
# Box limitation (docs/coord/plan_amd_gfx950_mi350p.md section 2): the
# authoring workstation has no ROCm toolchain, so this script runs two
# HOST-side proofs and honestly labels them as such:
#
#   1. ISLAND_TU_HOST_PROOF - the full .hip translation unit parsed as
#      ordinary C++ against the HIP headers, with the device annotations
#      rendered inert exactly like a real host pass. Exercises every kernel
#      body, helper, entry signature, launch call, and status mapping.
#   2. ENTRY_HEADER_C_PROOF - the core-facing entry header consumed as C11
#      twice (guard idempotence), the way dsv4_core will include it.
#
# Neither proof executes device code and neither replaces
#
#     hipcc --offload-arch=gfx950 -c spark_dsv4_rocm_islands.hip
#
# on MI350P hardware or a ROCm install - run ISLAND_TU_HIPCC when available:
#
#   ./island_compile_check.sh hipcc    # real toolchain (hipcc in PATH)
#   ./island_compile_check.sh vendor   # default host proofs, vendored headers
#   ./island_compile_check.sh system   # host proofs vs /opt/rocm/include
#   HIP_HEADERS=/path ./island_compile_check.sh system
#   OUT=dir ./island_compile_check.sh

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
    echo "usage: $0 [vendor|system|hipcc]" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"
rc=0

if [ "$MODE" = "hipcc" ]
then
  command -v hipcc >/dev/null 2>&1 || { echo "hipcc not in PATH" >&2; exit 2; }
  echo "ISLAND_TU_HIPCC: hipcc --offload-arch=gfx950 (real device compile)"
  if hipcc --offload-arch=gfx950 -std=c++14 -Wall -Wextra -Werror \
       -I"$REPO_INCLUDE" -I"$SRC" \
       -c "$SRC/spark_dsv4_rocm_islands.hip" \
       -o "$OUT/spark_dsv4_rocm_islands_gfx950.o"
  then
    echo "  OK   spark_dsv4_rocm_islands.hip -> gfx950 code object"
  else
    echo "  FAIL spark_dsv4_rocm_islands.hip"
    rc=1
  fi
  if [ "$rc" -eq 0 ]
  then
    echo "COMPILE_CHECK_OK: island TU compiled for gfx950 with hipcc."
  else
    echo "COMPILE_CHECK_FAILED."
  fi
  exit "$rc"
fi

CC="${CC:-clang}"
CXX="${CXX:-clang++}"
WARN="-Wall -Wextra -Werror"
COMMON="-I$HIP_ROOT -I$REPO_INCLUDE -I$SRC"

# A plain host compiler would let spark_hardware_topology.h fall back to the
# cuda.sm121.gb10 profile (wavefront 32); the gfx950 archive is compiled for
# its own profile, so the proof pins the frozen target id explicitly.
TARGET_PIN="-DSPARK_HW_TARGET_ID=SPARK_HW_TARGET_ROCM_GFX950_MI350P"

echo "compiler: $CXX / $CC"
echo "HIP headers: $HIP_ROOT ($MODE mode)"
echo
# -x c++ matters twice over: clang auto-selects HIP language mode on the
# .hip suffix and then demands ROCm device libraries, and the vendored AMD
# vector-type header uses message-less static_assert (C++17).
echo "ISLAND_TU_HOST_PROOF: full .hip TU as plain C++ (device annotations inert)"
if "$CXX" -x c++ -std=c++17 $WARN -D__HIP_PLATFORM_AMD__ $TARGET_PIN $COMMON \
     -c "$SRC/spark_dsv4_rocm_islands.hip" \
     -o "$OUT/spark_dsv4_rocm_islands_hostproof.o"
then
  echo "  OK   spark_dsv4_rocm_islands.hip parses end to end as host C++"
else
  echo "  FAIL spark_dsv4_rocm_islands.hip"
  rc=1
fi

echo "ENTRY_HEADER_C_PROOF: entry surface consumed as C11, included twice"
cat > "$OUT/island_header_c_driver.c" <<'DRIVER'
#include "spark_dsv4_rocm_islands.h"
#include "spark_dsv4_rocm_islands.h"
int spark_dsv4_rocm_islands_header_c_driver(void)
{
    return SPARK_DSV4_ROCM_WEIGHT_FORMAT_BF16 == 0u &&
           SPARK_DSV4_ROCM_WEIGHT_FORMAT_FP8_E4M3 == 4u ? 0 : 1;
}
DRIVER
if "$CC" -std=c11 $WARN $TARGET_PIN $COMMON \
     -c "$OUT/island_header_c_driver.c" \
     -o "$OUT/island_header_c_driver.o"
then
  echo "  OK   spark_dsv4_rocm_islands.hip header is valid C11 and self-contained"
else
  echo "  FAIL island header C consumption"
  rc=1
fi

# EXECUTION-GRADE PROBE: run the C2 integer island kernels on the host
# under the executor coordinate shims (blockDim 1 walks every strided loop;
# cross-lane-reduction kernels are excluded by design and covered by the
# formula probes plus, later, hardware). This is real executed evidence,
# not another parse.
PROBE_SRC="$MODULE_DIR/tools/l3_c2_host_probe.cpp"
if "$CXX" -x c++ -std=c++17 $WARN -D__HIP_PLATFORM_AMD__ $TARGET_PIN \
     -DSPARK_DSV4_ROCM_HOST_EXECUTOR $COMMON \
     "$PROBE_SRC" -o "$OUT/l3_c2_host_probe" && \
     "$OUT/l3_c2_host_probe"
then
  echo "  OK   l3_c2_host_probe (executed C2 surfaces)"
else
  echo "  FAIL l3_c2_host_probe"
  rc=1
fi

# L1 hcEnter probe: executes the pre-reduce kernel under the same shims
# and replays the FIXED split-K/Sinkhorn trees deterministically against
# naive fp64 ground truth plus an independent finalize transcription.
HCENTER_SRC="$MODULE_DIR/tools/l1_hcenter_host_probe.cpp"
if "$CXX" -x c++ -std=c++17 $WARN -D__HIP_PLATFORM_AMD__ $TARGET_PIN \
     -DSPARK_DSV4_ROCM_HOST_EXECUTOR $COMMON \
     "$HCENTER_SRC" -o "$OUT/l1_hcenter_host_probe" && \
     "$OUT/l1_hcenter_host_probe"
then
  echo "  OK   l1_hcenter_host_probe (L1 hcEnter executed + replayed trees)"
else
  echo "  FAIL l1_hcenter_host_probe"
  rc=1
fi

# The primitives keep proving themselves too; one command, whole module.
for src in "$SRC"/spark_hw_rocm_*.c
do
  obj="$OUT/$(basename "${src%.c}").o"
  if "$CC" -std=c11 $WARN -D__HIP_PLATFORM_AMD__ $TARGET_PIN $COMMON \
       -c "$src" -o "$obj"
  then
    echo "  OK   $(basename "$src") (primitive family regression)"
  else
    echo "  FAIL $(basename "$src")"
    rc=1
  fi
done

echo
if [ "$rc" -eq 0 ]
then
  echo "COMPILE_CHECK_OK: host-surface proofs green against $MODE HIP headers."
  echo "Device-code verification still owed on gfx950 hardware (ran-where note)."
else
  echo "COMPILE_CHECK_FAILED."
fi
exit "$rc"
