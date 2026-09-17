#!/bin/bash
# Re-ship the four recovered qwen-max v2 packs (rebuilt from source on
# spark1) to their placement nodes, dual-sha gated, receipts restored,
# chattr ring closed.
set -u
ship() {
  r=$1; dst=$2
  nn=$(printf "%02d" "$r")
  src_sha=$(timeout 900 ssh -o ConnectTimeout=10 spark1 "
    sha256sum ~/max-rebuild/qwen38_max.tp4_pp4.rank$nn.spstage | awk '{print \$1}'") || { echo "SRC FAIL rank$r"; return 1; }
  dsha=$(timeout 2400 ssh -o ConnectTimeout=10 "$dst" "
    sudo chattr -i ~/sparkdata/qwen38_max.tp4pp4/packs/qwen38_max.tp4_pp4.rank$nn.spstage 2>/dev/null
    rsync -a --timeout 1800 spark1:max-rebuild/qwen38_max.tp4_pp4.rank$nn.spstage ~/sparkdata/qwen38_max.tp4pp4/packs/
    rsync -a spark1:max-rebuild/qwen38_max.tp4_pp4.rank$nn.spstage.receipt.json ~/sparkdata/qwen38_max.tp4pp4/packs/ 2>/dev/null
    d=\$(sha256sum ~/sparkdata/qwen38_max.tp4pp4/packs/qwen38_max.tp4_pp4.rank$nn.spstage | awk '{print \$1}')
    if [ \"\$d\" != \"$src_sha\" ]; then echo \"SHA MISMATCH \$d\" >&2; exit 1; fi
    sudo chattr +i ~/sparkdata/qwen38_max.tp4pp4/packs/qwen38_max.tp4_pp4.rank$nn.spstage
    echo \$d") || { echo "SHIP FAILED rank$r -> $dst"; return 1; }
  echo "rank$r: spark1(${src_sha:0:16}) -> $dst(${dsha:0:16}...) OK"
}
ship 2 spark2
ship 7 spark7
ship 11 sparkb
ship 15 sparkf
echo RESHIP-DONE
