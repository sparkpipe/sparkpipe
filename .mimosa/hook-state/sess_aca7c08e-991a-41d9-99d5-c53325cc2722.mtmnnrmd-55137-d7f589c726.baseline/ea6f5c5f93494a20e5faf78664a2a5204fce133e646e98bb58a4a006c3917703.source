#!/bin/bash
# Strip the MTP append from every placed glm5_next g5nsp pack, fleet-wide.
set -u
NODES="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
for n in $NODES; do
  (
    timeout 3600 ssh -o ConnectTimeout=10 "$n" '
      for f in \
        "$HOME/sparkdata/glm5_next.tp16/packs/"*.g5nsp \
        "$HOME/sparkdata/glm5_next.tp4pp4/packs/"*.g5nsp \
        "$HOME/sparkdata/glm5_next.tp8.fp8/packs/"*.g5nsp \
        "$HOME/sparkdata/glm5_next.bf16.tp16/packs/"*.g5nsp ; do
        [ -f "$f" ] || continue
        case "$f" in *receipt*) continue ;; esac
        sudo python3 "$HOME/sparkpipe/tools/stagepack_mtp_strip.py" --pack "$f" --family g5nsp
      done' >> "/tmp/mtp_strip_g5_$n.log" 2>&1
    echo "$n finished rc=$?" >> /tmp/mtp_strip_g5_fleet.log
  ) &
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
echo G5-STRIP-DONE
