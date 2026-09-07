#!/bin/bash
# Ship dsv4-pro TP16 rank masters from sparka's warm CENTRAL to canonical
# nodes: rank r -> spark{hex r}:~/sparkdata/dsv4_pro.tp16/packs/.
# Per placement law: real bytes + sha256 verified at destination + receipt.
# Usage: dsv4pro_tp16_ship.sh [ranks...]   (default: all 0-15)
set -euo pipefail
CENTRAL=/mnt/model-warm/packbuild/dsv4pro-tp16
HEX=(0 1 2 3 4 5 6 7 8 9 a b c d e f)
ranks=("$@")
[[ $# -ge 1 ]] || ranks=($(seq 0 15))
rc=0
for rank in "${ranks[@]}"; do
  host="spark${HEX[$rank]}"
  dir="/home/${host}/sparkdata/dsv4_pro.tp16/packs"
  pack="$CENTRAL/rank${rank}.spstage"
  shafile="$CENTRAL/rank${rank}.sha"
  [[ -s "$pack" ]] || { echo "RANK${rank}-NO-MASTER skip"; rc=1; continue; }
  [[ -s "$shafile" ]] || { sha256sum "$pack" | awk '{print $1}' > "$shafile"; }
  bytes=$(stat -c %s "$pack")
  exp=$(cat "$shafile")
  ssh -o BatchMode=yes "$host" "mkdir -p '$dir'"
  # idempotent: skip if destination already digest-matches
  dsha=$(ssh -o BatchMode=yes "$host" "sha256sum '$dir/rank${rank}.spstage' 2>/dev/null" | awk '{print $1}') || dsha=""
  if [[ "$dsha" == "$exp" ]]; then
    echo "RANK${rank}-ALREADY-PLACED $host"; continue
  fi
  scp -q -o BatchMode=yes "$pack" "$host:$dir/rank${rank}.spstage"
  dsha=$(ssh -o BatchMode=yes "$host" "sha256sum '$dir/rank${rank}.spstage'" | awk '{print $1}')
  if [[ "$dsha" != "$exp" ]]; then
    echo "RANK${rank}-SHIP-SHA-MISMATCH $host exp=$exp got=$dsha"; rc=1; continue
  fi
  scp -q -o BatchMode=yes "$CENTRAL/rank${rank}.receipt.json" \
    "$CENTRAL/rank${rank}.sha" "$host:$dir/"
  echo "RANK${rank}-PLACED $host bytes=$bytes sha=$exp"
done
exit $rc
