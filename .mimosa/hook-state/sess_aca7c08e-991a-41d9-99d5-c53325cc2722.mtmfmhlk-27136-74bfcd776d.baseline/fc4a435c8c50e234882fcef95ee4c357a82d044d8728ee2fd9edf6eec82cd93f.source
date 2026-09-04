#!/usr/bin/env bash
# k3_deploy_stage_par.sh STAGE_IDX PREFIX — push this stage's four rank packs
# to their four rank hosts CONCURRENTLY (k3_deploy_stage.sh is serial and
# costs ~1 h/stage; four parallel rsyncs from the stage node cost ~20 min).
# Same receipts as the serial tool: per-rank rsync --append-verify (resumable)
# plus a both-sides sha256 comparison. Ranks land at
# spark{hex(4*STAGE+t)}:/home/<host>/sparkdata/k3.mxfp4.tp4pp4/packs/.
set -u
S="${1:?usage: k3_deploy_stage_par.sh STAGE_IDX PREFIX}"
PREFIX="${2:?usage: k3_deploy_stage_par.sh STAGE_IDX PREFIX}"
LOGDIR="${PREFIX}.deploys"
mkdir -p "$LOGDIR"

pids=""
for t in 0 1 2 3; do
  idx=$((S * 4 + t)); hex=$(printf '%x' "$idx"); dst="spark$hex"
  (
    remote_dir="/home/$dst/sparkdata/k3.mxfp4.tp4pp4/packs"
    remote_pack="$remote_dir/k3.stage${S}.rank0${t}.pack"
    ssh "$dst" "mkdir -p $remote_dir"
    rsync --partial --append-verify -e ssh \
      "${PREFIX}.rank0${t}.pack" "$dst:$remote_pack"
    src_sum=$(sha256sum "${PREFIX}.rank0${t}.pack" | awk '{print $1}')
    dst_sum=$(ssh "$dst" sha256sum "$remote_pack" | awk '{print $1}')
    if [ "$src_sum" = "$dst_sum" ]; then
      echo "deployed $remote_pack $dst_sum" > "$LOGDIR/rank0${t}.receipt"
    else
      echo "SHA MISMATCH $remote_pack src=$src_sum dst=$dst_sum" > "$LOGDIR/rank0${t}.receipt"
    fi
  ) >> "$LOGDIR/rank0${t}.log" 2>&1 &
  pids="$pids $!"
done
rc=0
for p in $pids; do wait "$p" || rc=1; done
cat "$LOGDIR"/rank0*.receipt
[ "$rc" = 0 ] && echo "stage $S parallel deploy done" || echo "stage $S deploy had failures"
exit "$rc"
