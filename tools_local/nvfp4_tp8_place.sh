#!/bin/bash
# Place the nvfp4 TP8 arm: move each primary's verified pack into the
# canonical dir + chattr, then ship the 2x-map secondary copy with a
# dual-sha gate (exit non-zero on any mismatch) and a receipt on both
# ends. Rank r lives on spark{r} + spark{(r+8)%16}; rank 0 was
# relay-built on sparka, so it ships to BOTH spark0 and spark8 (the
# sparka copy is removed only after both digests verify in place).
# Sparke is skipped here (its dsv4 compaction owns the node).
set -u
place() {
  r=$1; primary=$2; shift 2
  nn=$(printf "%02d" "$r")
  psha=$(timeout 120 ssh -o ConnectTimeout=10 "$primary" "
    mkdir -p ~/sparkdata/qwenflash.tp8.nvfp4/packs
    if [ -f ~/sparkdata/qwenflash.tp8.nvfp4/packs/tp8-rank$nn.q4fsp ]; then
      echo canonical-already >&2
    else
      mv ~/stagepacks/nvfp4-tp8/tp8-rank$nn.q4fsp ~/sparkdata/qwenflash.tp8.nvfp4/packs/
      mv ~/stagepacks/nvfp4-tp8/tp8-rank$nn.q4fsp.receipt.json ~/sparkdata/qwenflash.tp8.nvfp4/packs/ 2>/dev/null
    fi
    sudo chattr +i ~/sparkdata/qwenflash.tp8.nvfp4/packs/tp8-rank$nn.q4fsp
    sha256sum ~/sparkdata/qwenflash.tp8.nvfp4/packs/tp8-rank$nn.q4fsp | awk '{print \$1}'
  ") || { echo "PRIMARY FAILED rank$r on $primary"; return 1; }
  [ -n "$psha" ] || { echo "PRIMARY FAILED rank$r (no sha)"; return 1; }
  for sec in "$@"; do
    ssha=$(timeout 1800 ssh -o ConnectTimeout=10 "$sec" "
      mkdir -p ~/sparkdata/qwenflash.tp8.nvfp4/packs
      rsync -a --timeout 900 $primary:sparkdata/qwenflash.tp8.nvfp4/packs/tp8-rank$nn.q4fsp ~/sparkdata/qwenflash.tp8.nvfp4/packs/
      d=\$(sha256sum ~/sparkdata/qwenflash.tp8.nvfp4/packs/tp8-rank$nn.q4fsp | awk '{print \$1}')
      if [ \"\$d\" != \"$psha\" ]; then echo \"SHA MISMATCH \$d\" >&2; exit 1; fi
      printf '{\"kind\": \"sparkpipe.qwen4_flash.stagepack-receipt.v1\", \"rank\": $r, \"tp_degree\": 8, \"expert_format\": \"nvfp4-official\", \"mtp\": \"none\", \"output_sha256\": \"%s\", \"ship_source\": \"%s\", \"ship_verified\": true}\n' \"\$d\" '$primary' > ~/sparkdata/qwenflash.tp8.nvfp4/packs/tp8-rank$nn.q4fsp.receipt.json
      sudo chattr +i ~/sparkdata/qwenflash.tp8.nvfp4/packs/tp8-rank$nn.q4fsp
      echo \$d
    ") || { echo "SHIP FAILED rank$r -> $sec"; return 1; }
    echo "rank$r: $primary($psha) -> $sec(${ssha:0:16}...) OK"
  done
  echo "rank$r PLACED: $primary $*"
}
place 0 sparka spark0 spark8
place 1 spark1 spark9
place 2 spark2 sparka
place 3 spark3 sparkb
place 4 spark4 sparkc
place 5 spark5 sparkd
place 6 spark6
place 7 spark7 sparkf
echo PLACEMENT-DONE
