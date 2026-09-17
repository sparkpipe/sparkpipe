#!/bin/bash
# Clean rebuild+replace for the remaining 15 tp8.fp8 g5nsp ranks.
# Per node: stop any tp8 mirror rsync, rebuild MTP-less from source,
# verify with glm5_next_pack_verify, replace in place, relock.
set -u
for pair in spark0:0 spark1:1 spark2:2 spark3:3 spark4:4 spark6:6 spark7:7 \
            spark8:8 spark9:9 sparka:10 sparkb:11 sparkc:12 sparkd:13 \
            sparke:14 sparkf:15; do
  n=${pair%%:*}
  r=${pair##*:}
  nn=$(printf "%02d" "$r")
  (
    timeout 7200 ssh -o ConnectTimeout=10 "$n" '
      pkill -f "rsync.*tp8.fp8" 2>/dev/null
      sudo chattr -i "$HOME/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank'"$nn"'.g5nsp" 2>/dev/null
      rm -rf "$HOME/g5-rebuild"
      mkdir -p "$HOME/g5-rebuild"
      python3 "$HOME/sparkpipe/tools/glm5_next_resident_stagepack.py" \
        --source /mnt/model-warm/glm-5.3-flash \
        --first-layer 0 --layer-count 45 \
        --tp-degree 8 --tp-rank '"$r"' \
        --output "$HOME/g5-rebuild/glm5_next_stage.tp8.rank'"$nn"'.g5nsp" &&
      python3 "$HOME/sparkpipe/tools/glm5_next_pack_verify.py" \
        --pack "$HOME/g5-rebuild/glm5_next_stage.tp8.rank'"$nn"'.g5nsp" \
        --source /mnt/model-warm/glm-5.3-flash \
        --tp-rank '"$r"' --tp-degree 8 &&
      sudo cp "$HOME/g5-rebuild/glm5_next_stage.tp8.rank'"$nn"'.g5nsp/glm5_next_stage.tp8.rank'"$nn"'.g5nsp" \
        "$HOME/sparkdata/glm5_next.tp8.fp8/packs/" &&
      sudo chown '"$(whoami)"':'"$(whoami)"' "$HOME/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank'"$nn"'.g5nsp 2>/dev/null
      sudo chattr +i "$HOME/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank'"$nn"'.g5nsp" &&
      echo "RANK'"$nn"' REPLACED-LOCKED"
    ' >> "/tmp/g5_rebuild_$n.log" 2>&1
    echo "$n rc=$?" >> /tmp/g5_rebuild_fleet.log
  ) &
  while [ "$(jobs -r | wc -l)" -ge 5 ]; do sleep 10; done
done
wait
echo REBUILD-FLEET-DONE
