#!/bin/bash
# Ship dsv4-pro TP4PP4 rank packs from warm CENTRAL to canonical nodes:
# rank r -> spark{hex r}:~/sparkdata/dsv4_pro.tp4pp4/packs/ with the
# flash-set convention: packs/dsv4_pro.tp4_pp4.rankNN.spstage (NN
# zero-padded) + top-level rankNN.receipt ("rank r host sha timestamp")
# + rankNN.sha. Idempotent: destination digest match => skip.
# Usage: dsv4pro_tp4pp4_ship.sh [ranks...]   (default: all 0-15)
set -euo pipefail
SRC=/mnt/model-warm/packbuild/dsv4pro-tp4pp4
HEX=(0 1 2 3 4 5 6 7 8 9 a b c d e f)
ranks=("$@")
[[ $# -ge 1 ]] || ranks=($(seq 0 15))
rc=0
for rank in "${ranks[@]}"; do
  nn=$(printf '%02d' "$rank")
  host="spark${HEX[$rank]}"
  dir="/home/${host}/sparkdata/dsv4_pro.tp4pp4/packs"
  pack="$SRC/dsv4_pro.tp4_pp4.rank${nn}.spstage"
  shafile="$SRC/dsv4_pro.tp4_pp4.rank${nn}.sha"
  [[ -s "$pack" ]] || { echo "RANK${rank}-NO-MASTER skip"; rc=1; continue; }
  if [[ ! -s "$shafile" ]]; then
    sha256sum "$pack" | awk '{print $1}' > "$shafile"
  fi
  exp=$(cat "$shafile")
  [[ -n "$exp" ]] || { echo "RANK${rank}-NO-SHA skip"; rc=1; continue; }
  ssh -o BatchMode=yes "$host" "mkdir -p '$dir'"
  dsha=$(ssh -o BatchMode=yes "$host" "sha256sum '$dir/dsv4_pro.tp4_pp4.rank${nn}.spstage' 2>/dev/null" | awk '{print $1}') || dsha=""
  if [[ "$dsha" == "$exp" ]]; then
    echo "RANK${rank}-ALREADY-PLACED $host"; continue
  fi
  scp -q -o BatchMode=yes "$pack" "$host:$dir/dsv4_pro.tp4_pp4.rank${nn}.spstage"
  dsha=$(ssh -o BatchMode=yes "$host" "sha256sum '$dir/dsv4_pro.tp4_pp4.rank${nn}.spstage'" | awk '{print $1}')
  if [[ "$dsha" != "$exp" ]]; then
    echo "RANK${rank}-SHIP-SHA-MISMATCH $host exp=$exp got=$dsha"; rc=1; continue
  fi
  scp -q -o BatchMode=yes "$shafile" "$host:$dir/../dsv4_pro.tp4_pp4.rank${nn}.sha"
  ssh -o BatchMode=yes "$host" "printf 'rank %s %s %s %s\n' '$rank' '$host' '$exp' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > /home/${host}/sparkdata/dsv4_pro.tp4pp4/rank${nn}.receipt"
  echo "RANK${rank}-PLACED $host bytes=$(stat -c %s "$pack") sha=$exp"
done
exit $rc
