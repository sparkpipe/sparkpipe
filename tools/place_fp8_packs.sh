#!/bin/sh
cd ~/hy4-fp8-packs || exit 1
place_once() {
  r=$1
  node=$2
  p="model-fp8-tp16-rank-$r.safetensors"
  if [ "$node" = "sparkc" ]; then
    mkdir -p "$HOME/sparkdata/hy4.fp8.tp16/packs/rank-$r" || return 1
    cp -f "$p" "$HOME/sparkdata/hy4.fp8.tp16/packs/rank-$r/" || return 1
    cp -f "manifest-rank-$r.json" "$p.sha256" \
      "$HOME/sparkdata/hy4.fp8.tp16/packs/rank-$r/" || return 1
    cd "$HOME/sparkdata/hy4.fp8.tp16/packs/rank-$r" || return 1
    sha256sum -c "$p.sha256" || return 1
    cd ~/hy4-fp8-packs || exit 1
    return 0
  fi
  ssh "$node" "mkdir -p ~/sparkdata/hy4.fp8.tp16/packs/rank-$r" || return 1
  scp -q "$p" "$node":sparkdata/hy4.fp8.tp16/packs/rank-$r/ || return 1
  scp -q "manifest-rank-$r.json" "$p.sha256" \
    "$node":sparkdata/hy4.fp8.tp16/packs/rank-$r/ || return 1
  ssh "$node" "cd ~/sparkdata/hy4.fp8.tp16/packs/rank-$r && \
    sha256sum -c $p.sha256" || return 1
  return 0
}
place() {
  r=$1
  node=$2
  if [ -f "pl_${r}.done" ]; then
    echo "$r already placed"
    return 0
  fi
  if [ ! -f "rb_${r}.done" ] && [ "$node" != "sparkc" ]; then
    echo "$r rebuild not done yet, skipping"
    return 2
  fi
  attempt=0
  while [ $attempt -lt 8 ]; do
    if place_once "$r" "$node"; then
      touch "pl_${r}.done"
      echo "$r -> $node placed $(date +%H:%M) attempt $attempt" >> placement.log
      return 0
    fi
    attempt=$((attempt + 1))
    echo "$r -> $node attempt $attempt failed, retrying" >> placement.log
    sleep 45
  done
  echo "$r -> $node FAILED after 8 attempts" >> placement.log
  return 1
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
