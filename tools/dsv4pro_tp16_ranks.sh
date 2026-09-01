#!/bin/bash
# DSV4 Pro TP16 rank sharding + two-layer verification (CPU-only).
# Usage: dsv4pro_tp16_ranks.sh <rank> [<rank>...]
# Per rank: shard, --verify-output vs full pack, contract-derived
# verifier (dir + sampled payload/scale bytes vs GA checkpoint), sha.
set -euo pipefail
SRC=/mnt/model-warm/deepseek-v4-pro-0813-ga
OUT=/mnt/model-warm/packbuild/dsv4pro-tp16
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -ge 1 ]] || { echo "usage: $0 <rank>..."; exit 2; }
[[ -s "$OUT/dsv4_pro_full.spstage" ]] || { echo "MISSING-FULLPACK"; exit 1; }
mkdir -p "$OUT/logs"
for rank in "$@"; do
  pack="$OUT/rank${rank}.spstage"
  if [[ -s "$pack" && -s "$OUT/rank${rank}.receipt.json" && -s "$OUT/rank${rank}.sha" ]]; then
    echo "RANK${rank}-ALREADY-DONE"; continue
  fi
  python3 "$TOOLS/dsv4_tp16_stagepack.py" \
    --input-pack "$OUT/dsv4_pro_full.spstage" --output "$pack" \
    --rank "$rank" --tp-degree 16 --pp-stages 1 --pp-stage 0 \
    --model pro > "$OUT/logs/rank${rank}.shard.log" 2>&1
  python3 "$TOOLS/dsv4_tp16_stagepack.py" \
    --input-pack "$OUT/dsv4_pro_full.spstage" --output "$pack" \
    --rank "$rank" --tp-degree 16 --pp-stages 1 --pp-stage 0 \
    --model pro --verify-output >> "$OUT/logs/rank${rank}.shard.log" 2>&1
  python3 "$TOOLS/dsv4_pro_rank_pack_verify.py" --pack "$pack" \
    --rank "$rank" --tp-degree 16 --pp-stages 1 --model-dir "$SRC" \
    --json > "$OUT/rank${rank}.receipt.json" 2> "$OUT/logs/rank${rank}.verify.err"
  sha256sum "$pack" | awk '{print $1}' > "$OUT/rank${rank}.sha"
  echo "RANK${rank}-OK $(stat -c%s "$pack") bytes sha=$(cat "$OUT/rank${rank}.sha")"
done
echo "TP16-RANKS-DONE $*"
