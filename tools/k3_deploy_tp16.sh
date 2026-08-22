#!/usr/bin/env bash
# k3_deploy_tp16.sh RANK_PACK_DIR — push the sixteen TP16 rank packs to their
# hosts from the production node (no workstation in the data path). World
# rank i lands on spark$i (hex digits 0-9,a-f) under
# /home/<host>/sparkdata/k3.mxfp4.tp16/packs/. Resumable: rsync
# --append-verify skips the bytes already landed; every pack is sha256-checked
# against RANK_PACK_DIR/SHA256SUMS.tp16 (written by
# tools/k3_tp16_pack_production.sh) after the copy.
set -euo pipefail
SRC="${1:?usage: k3_deploy_tp16.sh RANK_PACK_DIR}"
SUMS="$SRC/SHA256SUMS.tp16"
test -s "$SUMS" || { echo "missing $SUMS - run k3_tp16_pack_production.sh first" >&2; exit 1; }

for idx in $(seq 0 15); do
    hex=$(printf '%x' "$idx"); dst="spark$hex"
    name="k3.tp16.rank$(printf '%02d' "$idx").pack"
    remote_dir="/home/$dst/sparkdata/k3.mxfp4.tp16/packs"
    ssh "$dst" "mkdir -p $remote_dir"
    rsync --partial --append-verify -e ssh "$SRC/$name" "$dst:$remote_dir/$name"
    want=$(grep "$name" "$SUMS" | awk '{print $1}')
    got=$(ssh "$dst" sha256sum "$remote_dir/$name" | awk '{print $1}')
    if [ "$want" = "$got" ]; then
        echo "deployed $dst:$remote_dir/$name ${got:0:12}..."
    else
        echo "SHA MISMATCH $dst:$remote_dir/$name (want ${want:0:12} got ${got:0:12})" >&2
        exit 1
    fi
done
echo "k3_tp16 deploy done: 16/16 rank packs verified"
