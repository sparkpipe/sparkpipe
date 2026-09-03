#!/bin/bash
# hy4 FP8 TP16 pack chain: build one rank on spark2 (warm source), ship to
# the rank's home node, verify remote sha, delete local. Rank r -> spark{hex r}.
set -x
cd ~/hy4-fp8-packs || exit 1
RANKS="00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15"
NODES="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
i=0
for rr in $RANKS; do
  node=$(echo $NODES | cut -d" " -f$((i + 1)))
  i=$((i + 1))
  echo "== rank $rr -> $node build start $(date +%H:%M:%S)"
  python3 hy4_fp8_stagepack.py --checkpoint /mnt/model-warm/hy4-preview-fp8-official \
    --rank $((10#$rr)) --output-directory . > build_rank$rr.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "BUILD FAIL rank $rr rc=$rc"; exit 1; fi
  lsha=$(cut -d" " -f1 model-fp8-tp16-rank-$rr.safetensors.sha256)
  echo "== rank $rr ship start $(date +%H:%M:%S)"
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 $node \
    "mkdir -p ~/sparkdata/hy4.fp8.tp16/packs/rank-$rr" || { echo "SHIP MKDIR FAIL $node"; exit 1; }
  scp -q model-fp8-tp16-rank-$rr.safetensors model-fp8-tp16-rank-$rr.safetensors.sha256 \
      manifest-rank-$rr.json $node:sparkdata/hy4.fp8.tp16/packs/rank-$rr/ || { echo "SHIP FAIL $node"; exit 1; }
  rsha=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 $node \
    "sha256sum ~/sparkdata/hy4.fp8.tp16/packs/rank-$rr/model-fp8-tp16-rank-$rr.safetensors" | cut -d" " -f1)
  if [ "$rsha" != "$lsha" ]; then echo "SHA FAIL $node got=$rsha want=$lsha"; exit 1; fi
  echo "== rank $rr VERIFIED $rsha $(date +%H:%M:%S)"
  rm -f model-fp8-tp16-rank-$rr.safetensors
done
echo ALL_RANKS_PLACED
