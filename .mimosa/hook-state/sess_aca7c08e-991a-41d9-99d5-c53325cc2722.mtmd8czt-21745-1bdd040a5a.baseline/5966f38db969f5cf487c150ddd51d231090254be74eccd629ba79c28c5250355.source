#!/bin/bash
# Strip the MTP draft blocks from the placed dsv4-pro TP16 rank packs.
# One pack per node; sudo for the chattr ring.
set -u
NODES="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparde sparkf"
for n in $NODES; do
  (
    timeout 3600 ssh -o ConnectTimeout=10 "$n" '
      f="$HOME/sparkdata/dsv4_pro.tp16/packs/rank*.spstage"
      for f in $HOME/sparkdata/dsv4_pro.tp16/packs/rank*.spstage; do
        [ -f "$f" ] || continue
        case "$f" in *receipt*|*sha*) continue ;; esac
        sudo python3 "$HOME/sparkpipe/tools/stagepack_mtp_strip.py" --pack "$f" --family dsv4 --compact
      done' >> "/tmp/mtp_strip_dsv4_$n.log" 2>&1
    echo "$n finished rc=$?" >> /tmp/mtp_strip_dsv4_fleet.log
  ) &
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
echo DSV4-STRIP-DONE
