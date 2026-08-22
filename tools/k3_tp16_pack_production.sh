#!/usr/bin/env bash
# k3_tp16_pack_production.sh CHECKPOINT_DIR OUT_DIR — TP16 pack production,
# the run that closes PR #667's "TP16 pack production" item.
#
#   1. pack the FULL model (all 93 layers) with expert_tile_k 32 — the only
#      tile size whose grid divides both the w1 diagonal (224 k-tiles x 384
#      cells at TP16) and the w2 latent split (192); see docs/K3_TP16_REPACK.md.
#   2. shard the full pack 16 ways with tools/k3_shard.py — one PP stage of
#      sixteen ranks, every rank carries every layer.
#   3. verify each of the 16 rank packs exists and is non-empty, price them
#      against the full pack, and write OUT_DIR/SHA256SUMS.tp16 for the
#      rsync --append-verify deploy (tools/k3_deploy_tp16.sh).
#
# Run on ONE node that holds the whole checkpoint (~1.6 TB, shards 1-96).
# Disk budget: full pack ~400 GB + 16 rank packs ~400 GB total; the script
# refuses to start under K3_PACK_MIN_FREE_GB (default 850) GB free on OUT_DIR's
# filesystem - override it only for synthetic-mini dry runs.
#
# POSIX-portable on purpose (df/wc/shasum fallbacks): the dry-run host is a
# mac; the production node is Linux. Both paths run the identical script.
set -euo pipefail
CKPT="${1:?usage: k3_tp16_pack_production.sh CHECKPOINT_DIR OUT_DIR}"
OUT="${2:?usage: k3_tp16_pack_production.sh CHECKPOINT_DIR OUT_DIR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILE_K=32
DEGREE=16
# keep in step with k3_shard.py's manifest_reserve (262128): the fixed
# per-rank bookkeeping the size guard subtracts before its 1/16 window
K3_RANK_MANIFEST_RESERVE="${K3_RANK_MANIFEST_RESERVE:-262128}"

if command -v sha256sum >/dev/null 2>&1; then
    sha256() { sha256sum "$@"; }
else
    sha256() { shasum -a 256 "$@"; }
fi
filesize() { wc -c < "$1" | tr -d "[:space:]"; }

mkdir -p "$OUT"
MIN_FREE_GB="${K3_PACK_MIN_FREE_GB:-850}"
free_kb=$(df -k "$OUT" | awk 'NR==2 {print $4}')
if [ "${free_kb:?}" -lt $((MIN_FREE_GB * 1024 * 1024)) ]; then
    echo "PACK PRODUCTION FAILURE: $OUT has ${free_kb} KB free; need >= $MIN_FREE_GB GB" >&2
    exit 1
fi

FULL="$OUT/k3.full.tilek${TILE_K}.pack"
if [ ! -s "$FULL" ]; then
    # full model = layer slice 0..num_hidden_layers at expert_tile_k TILE_K;
    # k3_pack.py's CLI takes [first_layer layer_count [expert_tile_k]] and its
    # config may nest under text_config (the released Kimi-K3 checkpoint does)
    LAYERS=$(CKPT="$CKPT" python3 -c 'import json, os; c = json.load(open(os.path.join(os.environ["CKPT"], "config.json"))); c = c.get("text_config", c); print(c["num_hidden_layers"])')
    echo "[1/3] packing all $LAYERS layers at expert_tile_k $TILE_K -> $FULL"
    PYTHONDONTWRITEBYTECODE=1 python3 "$SCRIPT_DIR/k3_pack.py" \
        "$CKPT" "$FULL" 0 "$LAYERS" "$TILE_K"
else
    echo "[1/3] full pack already present, keeping: $FULL"
fi
test -s "$FULL"

echo "[2/3] sharding $DEGREE ways -> $OUT/k3.tp16.rankNN.pack"
PYTHONDONTWRITEBYTECODE=1 python3 "$SCRIPT_DIR/k3_shard.py" \
    "$FULL" "$OUT/k3.tp16" "$DEGREE"

echo "[3/3] verifying rank packs"
full_bytes=$(filesize "$FULL")
fail=0
: > "$OUT/SHA256SUMS.tp16"
for rank in $(seq 0 $((DEGREE - 1))); do
    out="$OUT/k3.tp16.rank$(printf '%02d' "$rank").pack"
    if [ ! -s "$out" ]; then
        echo "VERIFY FAILURE: missing or empty $out" >&2
        fail=1
        continue
    fi
    bytes=$(filesize "$out")
    # each rank owns 1/16 of the model plus its slice bookkeeping; a rank
    # pack far from full/DEGREE means the sharder dropped or duplicated a
    # tensor class (the bd34381 defect class). The bookkeeping has a known
    # FIXED additive term - the rank-manifest reserve tools/k3_shard.py
    # writes in front of every rank pack - subtracted before the window
    # check so the guard holds at every scale (at mini dry-run sizes the
    # raw reserve alone exceeds 10% of full/16 and would cry wolf on a
    # perfect shard)
    payload=$((bytes - K3_RANK_MANIFEST_RESERVE))
    lo=$((full_bytes / DEGREE * 9 / 10))
    hi=$((full_bytes / DEGREE * 11 / 10))
    if [ "$payload" -lt "$lo" ] || [ "$payload" -gt "$hi" ]; then
        echo "VERIFY WARNING: $out is $bytes bytes ($payload payload), outside 1/16 +/- 10% of $full_bytes" >&2
    fi
    du -sh "$out"
    (cd "$OUT" && sha256 "k3.tp16.rank$(printf '%02d' "$rank").pack") \
        >> "$OUT/SHA256SUMS.tp16"
done
[ "$fail" = 0 ] || exit 1

echo "k3_tp16_pack_production done: full tile_k $TILE_K pack ($full_bytes bytes)," \
     "$DEGREE verified rank packs, checksums in $OUT/SHA256SUMS.tp16"
