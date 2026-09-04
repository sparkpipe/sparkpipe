#!/bin/bash
# Strip the MTP tail from placed flash-family stagepacks, wave 1:
# all nodes EXCEPT sparka/spark1-4 (those are running the nvfp4 -f
# builds; their strip runs after, one IO-heavy job per node).
set -u
NODES="spark0 spark5 spark6 spark7 spark8 spark9 sparkb sparkc sparkd sparke sparkf"
for n in $NODES; do
  (
    timeout 5400 ssh -o ConnectTimeout=10 "$n" '
      for f in \
        "$HOME/sparkdata/qwenflash.tp8.fp8/packs/"*.spstage \
        "$HOME/sparkdata/qwenflash.tp4pp4.fp8/packs/"*.spstage \
        "$HOME/sparkdata/qwen4_flash.tp4/packs_v4/"*.qwen4_flashsp \
        "$HOME/qf_rank4/"*.pack ; do
        [ -f "$f" ] || continue
        case "$f" in *receipt*) continue ;; esac
        sudo python3 "$HOME/sparkpipe/tools/stagepack_mtp_strip.py" --pack "$f" --family qwen4_flash --compact
      done' >> "/tmp/mtp_strip_$n.log" 2>&1
    echo "$n finished rc=$?" >> /tmp/mtp_strip_fleet.log
  ) &
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
echo ALL-DONE
