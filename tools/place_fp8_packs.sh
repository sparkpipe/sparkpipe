#!/bin/sh
cd ~/hy4-fp8-packs || exit 1
place() {
  r=$1
  node=$2
  p="model-fp8-tp16-rank-$r.safetensors"
  if [ -f "pl_${r}.done" ]; then
    echo "$r already placed"
    return 0
  fi
  if [ ! -f "rb_${r}.done" ] && [ "$node" != "sparkc" ]; then
    echo "$r rebuild not done yet, skipping"
    return 2
  fi
  if [ "$node" = "sparkc" ]; then
    mkdir -p ~/sparkdata/hy4.fp8.tp16/packs/rank-$r || return 1
    cp -n "$p" ~/sparkdata/hy4.fp8.tp16/packs/rank-$r/ || return 1
  else
    ssh "$node" "mkdir -p ~/sparkdata/hy4.fp8.tp16/packs/rank-$r" || return 1
    scp -q "$p" "$node":sparkdata/hy4.fp8.tp16/packs/rank-$r/ || return 1
  fi
  scp -q "manifest-rank-$r.json" "$p.sha256" \
    "$node":sparkdata/hy4.fp8.tp16/packs/rank-$r/ || return 1
  ssh "$node" "cd ~/sparkdata/hy4.fp8.tp16/packs/rank-$r && \
    sha256sum -c $p.sha256" || return 1
  touch "pl_${r}.done"
  echo "$r -> $node placed $(date +%H:%M)" >> placement.log
  return 0
}
place 00 spark0 || exit 1
place 01 spark1 || exit 1
place 02 spark2 || exit 1
place 03 spark3 || exit 1
place 04 spark4 || exit 1
place 05 spark5 || exit 1
place 06 spark6 || exit 1
place 07 spark7 || exit 1
place 08 spark8 || exit 1
place 09 spark9 || exit 1
place 10 sparka || exit 1
place 11 sparkb || exit 1
place 12 sparkc || exit 1
place 13 sparkd || exit 1
place 14 sparke || exit 1
place 15 sparkf || exit 1
echo PLACEMENT_DONE >> placement.log
