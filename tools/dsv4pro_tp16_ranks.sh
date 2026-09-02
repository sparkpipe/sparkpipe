#!/bin/bash
# DSV4 Pro TP16 rank sharding + two-layer verification (CPU-only).
# Usage: [env] dsv4pro_tp16_ranks.sh <rank> [<rank>...]
# Per rank: shard, --verify-output vs full pack, contract-derived
# verifier (dir + sampled payload/scale bytes vs GA checkpoint), sha.
# Env overrides:
#   DSSV4PRO_TP16_OUT  output dir for the pack (default: warm central;
#                      set to a node's LOCAL packs dir to skip the
#                      warm-write bottleneck - the deployed shard is the
#                      artifact of record)
#   DSSV4PRO_TP16_DEST host:/dir to ship the verified pack to (tar over
#                      ssh, sha-checked); receipt+sha always land on warm
set -euo pipefail
SRC="${DSSV4PRO_TP16_SRC:-/mnt/model-warm/deepseek-v4-pro-0813-ga}"
CENTRAL=/mnt/model-warm/packbuild/dsv4pro-tp16
FULL="${DSSV4PRO_TP16_FULL:-$CENTRAL/dsv4_pro_full.spstage}"
OUT="${DSSV4PRO_TP16_OUT:-$CENTRAL}"
DEST="${DSSV4PRO_TP16_DEST:-}"
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -ge 1 ]] || { echo "usage: $0 <rank>..."; exit 2; }
[[ -s "$FULL" ]] || { echo "MISSING-FULLPACK $FULL"; exit 1; }
mkdir -p "$OUT" "$CENTRAL/logs"
for rank in "$@"; do
  pack="$OUT/rank${rank}.spstage"
  if [[ -s "$pack" && -s "$CENTRAL/rank${rank}.receipt.json" && -s "$CENTRAL/rank${rank}.sha" ]]; then
    echo "RANK${rank}-ALREADY-DONE"; continue
  fi
  rm -f "$OUT"/.rank${rank}.spstage.*.tmp 2>/dev/null || true
  python3 "$TOOLS/dsv4_tp16_stagepack.py" \
    --input-pack "$FULL" --output "$pack" \
    --rank "$rank" --tp-degree 16 --pp-stages 1 --pp-stage 0 \
    --model pro > "$CENTRAL/logs/rank${rank}.shard.log" 2>&1
  python3 "$TOOLS/dsv4_tp16_stagepack.py" \
    --input-pack "$FULL" --output "$pack" \
    --rank "$rank" --tp-degree 16 --pp-stages 1 --pp-stage 0 \
    --model pro --verify-output >> "$CENTRAL/logs/rank${rank}.shard.log" 2>&1
  python3 "$TOOLS/dsv4_pro_rank_pack_verify.py" --pack "$pack" \
    --rank "$rank" --tp-degree 16 --pp-stages 1 --model-dir "$SRC" \
    --json > "$OUT/rank${rank}.receipt.json" 2> "$CENTRAL/logs/rank${rank}.verify.err"
  sha256sum "$pack" | awk '{print $1}' > "$OUT/rank${rank}.sha"
  if [[ -n "$DEST" ]]; then
    host="${DEST%%:*}"; dir="${DEST#*:}"
    ssh -o BatchMode=yes "$host" "mkdir -p '$dir'"
    tar -C "$OUT" -cf - "rank${rank}.spstage" | ssh -o BatchMode=yes "$host" "tar -xf - -C '$dir'"
    rsha=$(ssh -o BatchMode=yes "$host" "sha256sum '$dir/rank${rank}.spstage'" | awk '{print $1}')
    [[ "$rsha" == "$(cat "$OUT/rank${rank}.sha")" ]] || { echo "RANK${rank}-SHIP-SHA-MISMATCH"; exit 1; }
  fi
  scp -q "$OUT/rank${rank}.receipt.json" "$OUT/rank${rank}.sha" "$CENTRAL/" 2>/dev/null \
    || cp "$OUT/rank${rank}.receipt.json" "$OUT/rank${rank}.sha" "$CENTRAL/"
  echo "RANK${rank}-OK sha=$(cat "$OUT/rank${rank}.sha") out=$OUT${DEST:+ shipped=$DEST}"
done
echo "TP16-RANKS-DONE $*"
