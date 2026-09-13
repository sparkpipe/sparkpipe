#!/bin/bash
# run_l4_route_validation.sh - orchestrate the L4 synthetic sealed-route
# driver (EVIDENCE_S7_STEP3_L4_GROUPED_MOE.md owed item).
#
#   ./run_l4_route_validation.sh gen [outdir]   HOST: build + selfcheck +
#                                  generate fixtures (no GPU needed)
#   ./run_l4_route_validation.sh check <fixture-dir>   HARDWARE: hipcc-build
#                                  the islands TU + primitives + device
#                                  checker, verify fixture golden hashes,
#                                  run every case on gfx950
#
# gen runs anywhere with a C11 compiler; check requires hipcc and an AMD
# GPU (MI350P or any ROCm target - the kernels are arch-neutral C++ HIP).

set -u

MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$MODULE_DIR/../.." && pwd)"
VALIDATION_DIR="$MODULE_DIR/validation"
OUT="${OUT:-$MODULE_DIR/.compile-check}"
MODE="${1:-gen}"

build_gen() {
  mkdir -p "$OUT"
  echo "building host driver (C11)..."
  ${CC:-cc} -std=c11 -Wall -Wextra -Werror \
    -I"$REPO_ROOT/include" \
    "$VALIDATION_DIR/l4_route_driver.c" "$REPO_ROOT/src/spark_sha256.c" \
    -o "$OUT/l4_route_driver" || return 1
}

run_gen() {
  local fixdir="${2:-$VALIDATION_DIR/fixtures}"
  build_gen || return 1
  echo "selfcheck..."
  "$OUT/l4_route_driver" selfcheck || return 1
  mkdir -p "$fixdir"
  echo "generating fixtures into $fixdir..."
  "$OUT/l4_route_driver" gen "$fixdir"
}

run_check() {
  command -v hipcc >/dev/null 2>&1 || {
    echo "hipcc not in PATH - device check runs on the MI350P node only" >&2
    exit 2
  }
  local fixdir="${2:-$VALIDATION_DIR/fixtures}"
  local manifest="$fixdir/manifest.txt"
  [ -f "$manifest" ] || { echo "missing $manifest (run gen first)" >&2; exit 2; }

  mkdir -p "$OUT"
  echo "hipcc-building islands TU + primitives + device checker..."
  hipcc --offload-arch=gfx950 -std=c++14 -Wall -Wextra \
    -I"$REPO_ROOT/include" -I"$VALIDATION_DIR" \
    "$VALIDATION_DIR/l4_route_device_check.hip" \
    "$MODULE_DIR/source/spark_dsv4_rocm_islands.hip" \
    "$MODULE_DIR/source/spark_hw_rocm_memory.c" \
    "$MODULE_DIR/source/spark_hw_rocm_queue.c" \
    "$MODULE_DIR/source/spark_hw_rocm_event.c" \
    "$REPO_ROOT/src/spark_sha256.c" \
    -o "$OUT/l4_route_device_check" || exit 1

  local rc=0 line file expected actual
  while read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    file=$(echo "$line" | awk '{print $1}')
    expected=$(echo "$line" | awk '{print $2}')
    actual=$(command sha256sum "$fixdir/$file" 2>/dev/null | awk '{print $1}' \
             || command shasum -a 256 "$fixdir/$file" | awk '{print $1}')
    if [ "$actual" != "$expected" ]; then
      echo "GOLDEN HASH MISMATCH: $file" >&2
      rc=1
      continue
    fi
    echo "hash ok: $file"
    "$OUT/l4_route_device_check" "$fixdir/$file" || rc=1
  done < "$manifest"

  if [ "$rc" -eq 0 ]; then
    echo "L4_ROUTE_VALIDATION_OK: all cases passed on hardware."
  else
    echo "L4_ROUTE_VALIDATION_FAILED."
  fi
  exit "$rc"
}

case "$MODE" in
  gen) run_gen "${2:-}" ;;
  check) run_check "${2:-}" ;;
  *) echo "usage: $0 gen [outdir] | check [fixture-dir]" >&2; exit 2 ;;
esac
