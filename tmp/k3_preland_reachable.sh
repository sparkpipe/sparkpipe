#!/bin/bash
# Pre-land TP16 rank packs on REACHABLE hosts only, using k3_deploy_tp16.sh's
# exact per-rank mechanics (rsync --partial --append-verify + sha256 vs
# SHA256SUMS.tp16). Zero ring time; the OFFICIAL deployer still runs at
# 16/16 as the receipt-producing pass (it skips already-landed bytes).
set -uo pipefail
SRC="${1:?usage: k3_preland_reachable.sh RANK_PACK_DIR}"
SUMS="$SRC/SHA256SUMS.tp16"
LOG="$SRC/preland.log"
test -s "$SUMS" || { echo "missing $SUMS" >&2; exit 1; }
echo "preland start $(date -u +%FT%TZ)" >> "$LOG"
for idx in $(seq 0 15); do
    hex=$(printf '%x' "$idx"); dst="spark$hex"
    name="k3.tp16.rank$(printf '%02d' "$idx").pack"
    remote_dir="/home/$dst/sparkdata/k3.mxfp4.tp16/packs"
    if ! timeout 8 ssh -o ConnectTimeout=5 -o BatchMode=yes "$dst" true 2>/dev/null; then
        echo "skip $dst unreachable $(date -u +%FT%TZ)" >> "$LOG"
        continue
    fi
    ssh "$dst" "mkdir -p $remote_dir"
    rsync --partial --append-verify -e ssh "$SRC/$name" "$dst:$remote_dir/$name"
    want=$(grep "$name" "$SUMS" | awk '{print $1}')
    got=$(ssh "$dst" sha256sum "$remote_dir/$name" | awk '{print $1}')
    if [ "$want" = "$got" ]; then
        echo "deployed $dst:$remote_dir/$name ${got:0:12}... $(date -u +%FT%TZ)" >> "$LOG"
    else
        echo "SHA MISMATCH $dst:$remote_dir/$name (want ${want:0:12} got ${got:0:12}) $(date -u +%FT%TZ)" >> "$LOG"
    fi
done
echo "preland done $(date -u +%FT%TZ)" >> "$LOG"
