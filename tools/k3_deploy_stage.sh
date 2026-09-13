#!/usr/bin/env bash
# k3_deploy_stage.sh STAGE_IDX PREFIX — push this stage's four rank packs to
# their four rank hosts from the stage node itself (no workstation in the
# data path). Resumable: rsync --append-verify skips the bytes already
# landed. Usage on the stage node: nohup bash /tmp/k3_deploy_stage.sh 0 /tmp/k3_stage_0_24 ...
set -u
S=$1
PREFIX=$2
for t in 0 1 2 3; do
  idx=$((S*4+t)); hex=$(printf '%x' "$idx"); dst="spark$hex"
  remote_dir="/home/$dst/sparkdata/k3.mxfp4.tp4pp4/packs"
  remote_pack="$remote_dir/k3.stage${S}.rank0${t}.pack"
  ssh "$dst" "mkdir -p $remote_dir"
  rsync --partial --append-verify -e ssh "${PREFIX}.rank0${t}.pack" "$dst:$remote_pack"
  src_sum=$(sha256sum "${PREFIX}.rank0${t}.pack" | awk '{print $1}')
  dst_sum=$(ssh "$dst" sha256sum "$remote_pack" | awk '{print $1}')
  if [ "$src_sum" = "$dst_sum" ]; then
    echo "deployed $remote_pack ${dst_sum:0:12}..."
  else
    echo "SHA MISMATCH $remote_pack"
  fi
done
echo "stage $S deploy done"
