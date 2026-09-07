#!/bin/bash
# DSV4 Pro TP16 verify-only pass for ranks whose packs already exist and
# passed --verify-output (skips sharding; runs the contract-derived
# verifier + sha, then writes the receipt the resume check expects).
# Usage: dsv4pro_tp16_verify_only.sh <rank> [<rank>...]
set -euo pipefail
SRC=/mnt/model-warm/deepseek-v4-pro-0813-ga
OUT=/mnt/model-warm/packbuild/dsv4pro-tp16
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -ge 1 ]] || { echo "usage: $0 <rank>..."; exit 2; }
for rank in "$@"; do
  pack="$OUT/rank${rank}.spstage"
  [[ -s "$pack" ]] || { echo "RANK${rank}-NO-PACK"; exit 1; }
  python3 "$TOOLS/dsv4_pro_rank_pack_verify.py" --pack "$pack" \
    --rank "$rank" --tp-degree 16 --pp-stages 1 --model-dir "$SRC" \
    --json > "$OUT/rank${rank}.receipt.json" 2> "$OUT/logs/rank${rank}.verify.err"
  sha256sum "$pack" | awk '{print $1}' > "$OUT/rank${rank}.sha"
  echo "RANK${rank}-VERIFIED sha=$(cat "$OUT/rank${rank}.sha")"
done
echo "TP16-VERIFY-ONLY-DONE $*"
