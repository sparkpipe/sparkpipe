#!/usr/bin/env bash
# qwen38max_multidev_experts_manifest.sh — compile and run the family's
# expert-manifest tool (tools/qwen38max_experts_manifest.c, the lane-2
# charter precedent) against a rank pack, publishing the .experts sidecar
# the shared weightd's lazy attach fails closed without. Adapts the
# family's proven single-spark flow (tools/qwen38max_rebuild_rank.sh) to
# the shared-socket lane workflow: any host, any pack, private cell
# directory. Existing .experts files are left untouched.
set -euo pipefail
PACK="${1:?usage: qwen38max_multidev_experts_manifest.sh PACK_PATH}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CELL="${QMAX_MANIFEST_CELL:-${TMPDIR:-/tmp}/qmax-manifest-cell.$$}"
[ -f "$PACK" ] || { echo "pack not found: $PACK" >&2; exit 1; }
if [ -f "$PACK.experts" ]; then
  echo "exists: $PACK.experts"
  exit 0
fi
mkdir -p "$CELL"
trap 'rm -rf "$CELL"' EXIT
cc -std=c11 -O2 \
  -I"$ROOT" -I"$ROOT/include" \
  "$ROOT/tools/qwen38max_experts_manifest.c" \
  "$ROOT/src/spark_ck128.c" \
  "$ROOT/src/spark_status.c" \
  "$ROOT/runtime/spark_weightd_manifest.c" \
  -o "$CELL/qwen38max_experts_manifest"
"$CELL/qwen38max_experts_manifest" "$PACK"
