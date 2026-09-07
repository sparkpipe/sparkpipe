#!/bin/bash
# Place the 27B nvfp4a16 TP4 arm: finalize each primary in the canonical
# dir + chattr, then ship to the TP4 replication map nodes
# (rank r on every spark{N} with N%4 == r) with a dual-sha gate and a
# receipt on both ends.
set -u
place() {
  r=$1; primary=$2; shift 2
  nn=$(printf "%02d" "$r")
  psha=$(timeout 120 ssh -o ConnectTimeout=10 "$primary" "
    mkdir -p ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs
    if [ -f ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rank$nn.q38sp ]; then
      echo canonical-already >&2
    else
      mv ~/stagepacks/nvfp4a16-tp4/tp4-rank$nn.q38sp ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/
      mv ~/stagepacks/nvfp4a16-tp4/tp4-rank$nn.q38sp.receipt.json ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/ 2>/dev/null
    fi
    sudo chattr +i ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rank$nn.q38sp
    sha256sum ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rank$nn.q38sp | awk '{print \$1}'
  ") || { echo "PRIMARY FAILED rank$r on $primary"; return 1; }
  [ -n "$psha" ] || { echo "PRIMARY FAILED rank$r (no sha)"; return 1; }
  for sec in "$@"; do
    ssha=$(timeout 1800 ssh -o ConnectTimeout=10 "$sec" "
      mkdir -p ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs
      rsync -a --timeout 900 $primary:sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rank$nn.q38sp ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/
      d=\$(sha256sum ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rank$nn.q38sp | awk '{print \$1}')
      if [ \"\$d\" != \"$psha\" ]; then echo \"SHA MISMATCH \$d\" >&2; exit 1; fi
      printf '{\"kind\": \"sparkpipe.qwen38_27b.stagepack-receipt.v1\", \"rank\": $r, \"tp_degree\": 4, \"ffn_format\": \"nvfp4a16\", \"mtp\": \"none\", \"output_sha256\": \"%s\", \"ship_source\": \"%s\", \"ship_verified\": true}\n' \"\$d\" '$primary' > ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rank$nn.q38sp.receipt.json
      sudo chattr +i ~/sparkdata/qwen38-27b.nvfp4a16.tp4/packs/tp4-rank$nn.q38sp
      echo \$d
    ") || { echo "SHIP FAILED rank$r -> $sec"; return 1; }
    echo "rank$r: $primary(${psha:0:16}) -> $sec(${ssha:0:16}...) OK"
  done
  echo "rank$r PLACED: $primary $*"
}
place 0 spark0 spark4 spark8 sparkc
place 1 spark1 spark5 spark9 sparkd
place 2 spark2 spark6 sparka sparke
place 3 spark3 spark7 sparkb sparkf
echo PLACEMENT-DONE
