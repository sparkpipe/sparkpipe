#!/bin/sh
cd ~/hy4-fp8-packs || exit 1
dist() {
  r=$1
  node=$2
  e="model-fp8-tp16-rank-$r.safetensors.experts"
  attempt=0
  while [ $attempt -lt 4 ]; do
    if scp -q "$e" "$node":sparkdata/hy4.fp8.tp16/packs/rank-$r/; then
      echo "$r -> $node $(date +%H:%M)" >> experts_dist.log
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 20
  done
  echo "$r -> $node FAILED" >> experts_dist.log
  return 1
}
dist 00 spark0 || exit 1
dist 01 spark1 || exit 1
dist 02 spark2 || exit 1
dist 03 spark3 || exit 1
dist 04 spark4 || exit 1
dist 05 spark5 || exit 1
dist 06 spark6 || exit 1
dist 07 spark7 || exit 1
dist 08 spark8 || exit 1
dist 09 spark9 || exit 1
dist 10 sparka || exit 1
dist 11 sparkb || exit 1
cp -f model-fp8-tp16-rank-12.safetensors.experts \
  "$HOME/sparkdata/hy4.fp8.tp16/packs/rank-12/" || exit 1
echo "12 -> sparkc local" >> experts_dist.log
dist 13 sparkd || exit 1
dist 14 sparke || exit 1
dist 15 sparkf || exit 1
echo EXPERTS_DIST_DONE >> experts_dist.log
