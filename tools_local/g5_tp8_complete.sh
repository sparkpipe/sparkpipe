#!/bin/bash
# For each tp8 rank: wait for the g5-rebuild to land, verify it, replace
# the placed pack, relock. Idempotent — skips nodes already replaced.
set -u
for pair in "spark0 00 0" "spark1 01 1" "spark2 02 2" "spark3 03 3" "spark4 04 4" \
            "spark6 06 6" "spark7 07 7" "spark8 08 8" "spark9 09 9" "sparka 10 10" \
            "sparkb 11 11" "sparkc 12 12" "sparkd 13 13" "sparke 14 14" "sparkf 15 15"; do
  n=${pair%% *}; rest=${pair#* }; nn=${rest%% *}; r=${rest##* }
  (
    timeout 7200 ssh -o ConnectTimeout=10 "$n" '
      d="$HOME/g5-rebuild/glm5_next_stage.tp8.rank'"$nn"'.g5nsp"
      f="$d/glm5_next_stage.tp8.rank'"$nn"'.g5nsp"
      for i in $(seq 1 120); do [ -f "$f" ] && break; sleep 10; done
      [ -f "$f" ] || { echo "RANK'"$nn"' BUILD NEVER LANDED"; exit 1; }
      python3 "$HOME/sparkpipe/tools/glm5_next_pack_verify.py" \
        --pack "$f" --source /mnt/model-warm/glm-5.3-flash \
        --tp-rank '"$r"' --tp-degree 8 &&
      sudo chattr -i "$HOME/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank'"$nn"'.g5nsp" 2>/dev/null
      sudo cp "$f" \
        "$HOME/sparkdata/glm5_next.tp8.fp8/packs/" &&
      sudo chown '"$(whoami)"' "$HOME/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank'"$nn"'.g5nsp &&
      sudo chattr +i "$HOME/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank'"$nn"'.g5nsp &&
      echo "RANK'"$nn"' REPLACED-LOCKED"
    ' >> "/tmp/g5_complete_$n.log" 2>&1
    echo "$n rc=$?" >> /tmp/g5_complete_fleet.log
  ) &
  while [ "$(jobs -r | wc -l)" -ge 5 ]; do sleep 10; done
done
wait
echo COMPLETE-DONE
