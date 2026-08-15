#!/usr/bin/env bash
# k3_deploy_ranks.sh — copy completed TP4 rank packs from the stage nodes to
# their rank hosts and sha256-verify each copy. Stage s ranks t live on
# spark[s*4+t]: stage0->spark0-3, stage1->spark4-7, stage2->spark8-b,
# stage3->sparkc-f. Run from a workstation with ssh access to the ring.
set -euo pipefail
STAGE_HOSTS="spark1 spark4 spark8 sparkc"
STAGE_PREFIXES="/tmp/k3_stage_0_24 /tmp/k3_stage_24_23 /tmp/k3_stage_47_23 /tmp/k3_stage_70_23"
s=0
for entry in $STAGE_PREFIXES; do
  src=$(echo "$STAGE_HOSTS" | cut -d' ' -f$((s+1)))
  for t in 0 1 2 3; do
    idx=$((s*4+t)); hex=$(printf '%x' "$idx"); dst="spark$hex"
    local_pack="${entry}.rank0${t}.pack"
    remote_dir="/home/$dst/sparkdata/k3.mxfp4.tp4pp4/packs"
    remote_pack="$remote_dir/k3.stage${s}.rank0${t}.pack"
    src_sum=$(ssh "$src" sha256sum "$local_pack" | awk '{print $1}')
    ssh "$dst" "mkdir -p $remote_dir"
    scp -q "$src:$local_pack" "$dst:$remote_pack"
    dst_sum=$(ssh "$dst" sha256sum "$remote_pack" | awk '{print $1}')
    if [ "$src_sum" != "$dst_sum" ]; then echo "SHA MISMATCH $remote_pack" >&2; exit 1; fi
    echo "deployed $remote_pack ${dst_sum:0:12}..."
  done
  s=$((s+1))
done
echo "k3 rank deployment verified: 16 packs"
