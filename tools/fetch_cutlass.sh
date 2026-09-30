#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="${CUTLASS_URL:-https://github.com/NVIDIA/cutlass.git}"
COMMIT=0b55a2f691d69981583568fd9eb69687b1f0de8a
DEST="$ROOT/build/third_party/cutlass"
if [ -d "$DEST/.git" ] && [ "$(git -C "$DEST" rev-parse HEAD)" = "$COMMIT" ]; then
  echo "cutlass $COMMIT present at $DEST"
  exit 0
fi
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
git clone --quiet --filter=blob:none --no-checkout "$URL" "$DEST"
git -C "$DEST" checkout --quiet "$COMMIT"
got="$(git -C "$DEST" rev-parse HEAD)"
[ "$got" = "$COMMIT" ] || { echo "cutlass fetch: got $got, want $COMMIT" >&2; exit 1; }
test -f "$DEST/include/cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp" || { echo "cutlass fetch: $COMMIT lacks the sm120 blockwise grouped collective" >&2; exit 1; }
echo "cutlass $COMMIT fetched to $DEST"
