#!/usr/bin/env bash
# k3_rededeploy_stage.sh STAGE_IDX PREFIX — replace this stage's four rank
# packs on their rank hosts with the RE-SLICED ones (the fixed sharder's
# diagonal w1). The new rank packs are SMALLER than the old, so the deploy
# REPLACES the remote file (no --append-verify: appending to a larger stale
# file would corrupt it). Run ON the stage node that holds the re-sliced
# packs.
set -u
S=$1
PREFIX=$2
for t in 0 1 2 3; do
  idx=$((S*4+t)); hex=$(printf '%x' "$idx"); dst="spark$hex"
  remote_dir="/home/$dst/sparkdata/k3.mxfp4.tp4pp4/packs"
  remote_pack="$remote_dir/k3.stage${S}.rank0${t}.pack"
  ssh "$dst" "rm -f $remote_pack"
  rsync -e ssh "${PREFIX}.rank0${t}.pack" "$dst:$remote_pack"
  src_sum=$(sha256sum "${PREFIX}.rank0${t}.pack" | awk '{print $1}')
  dst_sum=$(ssh "$dst" sha256sum "$remote_pack" | awk '{print $1}')
  if [ "$src_sum" = "$dst_sum" ]; then
    echo "redeployed $remote_pack ${dst_sum:0:12}..."
  else
    echo "SHA MISMATCH $remote_pack"
  fi
done
echo "stage $S redeploy done"
