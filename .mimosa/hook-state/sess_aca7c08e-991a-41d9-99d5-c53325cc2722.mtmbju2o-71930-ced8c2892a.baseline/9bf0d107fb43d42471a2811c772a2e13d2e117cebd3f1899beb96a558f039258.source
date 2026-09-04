#!/bin/bash
# Strip wave 2: sparka/spark1-4 (their nvfp4 -f builds are complete).
set -u
for n in sparka spark1 spark2 spark3 spark4; do
  (
    timeout 5400 ssh -o ConnectTimeout=10 "$n" '
      for f in \
        "$HOME/sparkdata/qwenflash.tp8.fp8/packs/"*.spstage \
        "$HOME/sparkdata/qwenflash.tp4pp4.fp8/packs/"*.spstage \
        "$HOME/sparkdata/qwen4_flash.tp4/packs_v4/"*.qwen4_flashsp ; do
        [ -f "$f" ] || continue
        case "$f" in *receipt*) continue ;; esac
        sudo python3 "$HOME/sparkpipe/tools/stagepack_mtp_strip.py" --pack "$f" --family qwen4_flash --compact
      done' >> "/tmp/mtp_strip_$n.log" 2>&1
    echo "$n finished rc=$?" >> /tmp/mtp_strip_fleet.log
  ) &
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
echo WAVE2-DONE
