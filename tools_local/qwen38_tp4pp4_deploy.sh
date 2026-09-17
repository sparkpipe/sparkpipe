#!/bin/bash
# Deploy the qwen38max TP4xPP4 rank packs to ONE spark node.
#
# The manifest (tools/qwen38_tp4pp4_packs.py output) maps each of the 16
# world ranks to its stage pack and host. This script deploys only the packs
# the given host needs (deduplicated: a TP group shares one pack file).
#
# Usage:
#   qwen38_tp4pp4_deploy.sh --manifest PATH --host sparkN [--target-dir DIR] [--dry-run]
#
# No node is hardcoded; the host comes from --host and the manifest.
set -euo pipefail

MANIFEST=""
HOST=""
TARGET_DIR=""
DRY_RUN=0
DEFAULT_TARGET_ROOT="sparkdata/qwen38max.tp4_pp4/packs"

while [ $# -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --target-dir) TARGET_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

[ -n "$MANIFEST" ] || { echo "--manifest required" >&2; exit 2; }
[ -n "$HOST" ] || { echo "--host required" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }

DEPLOY_LIST=$(python3 - "$MANIFEST" "$HOST" <<'PY'
import json, sys
manifest, host = sys.argv[1], sys.argv[2]
ranks = [r for r in json.load(open(manifest))["ranks"] if r["host"] == host]
if not ranks:
    sys.exit(f"manifest has no rank assigned to host {host}")
print("\n".join(sorted({r["pack"] for r in ranks})))
PY
) || exit 1

TARGET_DIR="${TARGET_DIR:-/home/${HOST}/${DEFAULT_TARGET_ROOT}}"
RSYNC=(rsync -a --partial --inplace)
[ "$DRY_RUN" = "1" ] && RSYNC+=(--dry-run)

echo "deploying to ${HOST}:${TARGET_DIR}"
ssh "$HOST" "mkdir -p '${TARGET_DIR}'"
for pack in $DEPLOY_LIST; do
  echo "  $(basename "$pack") ($(stat -f %z "$pack" 2>/dev/null || stat -c %s "$pack") bytes)"
  "${RSYNC[@]}" "$pack" "${HOST}:${TARGET_DIR}/"
  "${RSYNC[@]}" "${pack}.receipt.json" "${HOST}:${TARGET_DIR}/"
done
echo "deployed $(echo "$DEPLOY_LIST" | wc -l | tr -d ' ') pack(s) to ${HOST}:${TARGET_DIR}"
