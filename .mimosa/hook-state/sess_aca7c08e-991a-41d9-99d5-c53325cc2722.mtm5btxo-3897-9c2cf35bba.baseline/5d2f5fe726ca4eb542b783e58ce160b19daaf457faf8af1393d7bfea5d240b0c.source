#!/usr/bin/env bash
# k3_tp16_deploy.sh — slice -> cross-verify -> ship -> delete, ONE rank at
# a time. Runs on the build host that holds the 93-layer TP16 stage pack
# (expert_tile_k=32). A full 16-rank shard run would write ~1.6 TB of rank
# packs BESIDE the 1.56 TB stage pack - over any node's disk - so each rank
# is sliced alone (byte-identical to the full run; tested on the probe),
# cross-verified against the stage pack, sha256-verified after scp into
# the target's sparkdata/k3.mxfp4.tp16/packs/, and the LOCAL copy deleted.
# Receipts (<work>/rankNN.receipt, digest + host) make the loop resumable.
#
# usage: k3_tp16_deploy.sh STAGE_PACK WORK_DIR [FIRST_RANK] [LAST_RANK]
set -euo pipefail
PACK="${1:?usage: k3_tp16_deploy.sh STAGE_PACK WORK_DIR [FIRST_RANK] [LAST_RANK]}"
WORK="${2:?usage: k3_tp16_deploy.sh STAGE_PACK WORK_DIR [FIRST_RANK] [LAST_RANK]}"
FIRST="${3:-0}"
LAST="${4:-15}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$WORK"
for r in $(seq "$FIRST" "$LAST"); do
  hex=$(printf '%x' "$r")
  host="spark$hex"
  two=$(printf '%02d' "$r")
  dst="/home/$host/sparkdata/k3.mxfp4.tp16/packs"
  local_pack="$WORK/k3.stage0.rank${two}.pack"
  if [ -s "$WORK/rank${two}.receipt" ]; then
    echo "skip rank $r (receipt present)"
    continue
  fi
  avail=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
  if [ "$avail" -lt 300 ]; then
    echo "FATAL: ${avail}G free in $WORK, need 300G for one rank pack"
    exit 1
  fi
  echo "[rank $r] slicing"
  PYTHONDONTWRITEBYTECODE=1 python3 "$ROOT/tools/k3_shard.py" "$PACK" \
    "$WORK/tp16" 16 "$r"
  mv "$WORK/tp16.rank${two}.pack" "$local_pack"
  echo "[rank $r] cross-verify vs stage pack"
  PYTHONDONTWRITEBYTECODE=1 python3 "$ROOT/tools/k3_verify_pack.py" \
    "$local_pack" --stage-pack "$PACK" --expect-tp-degree 16 \
    --expect-rank "$r" --expect-first 0 --expect-layers 93
  sum=$(sha256sum "$local_pack" | awk '{print $1}')
  echo "[rank $r] deploying to $host"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "mkdir -p $dst"
  scp -q "$local_pack" "$host:$dst/k3.stage0.rank${two}.pack"
  rsum=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
    "sha256sum $dst/k3.stage0.rank${two}.pack" | awk '{print $1}')
  if [ "$sum" != "$rsum" ]; then
    echo "FAIL rank $r: remote digest $rsum != local $sum"
    exit 1
  fi
  rm -f "$local_pack"
  echo "rank $r $host $sum $(date -u +%FT%TZ)" > "$WORK/rank${two}.receipt"
  echo "DEPLOYED rank $r on $host $sum"
done
echo "TP16 deploy complete ranks $FIRST..$LAST"
