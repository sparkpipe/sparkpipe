#!/bin/bash
# Place the rebuilt tp8.fp8 ranks (the packer receipt is the gate).
# The rebuild output is a DIRECTORY containing the pack file; the placed
# pack is a FILE (locked from the original placement).
set -u
for pair in "spark1 1 1" "spark2 2 2" "spark3 3 3" "spark4 4 4" "spark6 6 6" "spark7 7 7" "spark8 8 8" "spark9 9 9" "sparka 10 10" "sparkb 11 11" "sparkc 12 12" "sparkd 13 13" "sparke 14 14" "sparkf 15 15"; do
  n=${pair%% *}; r=${pair##* }
  (
    timeout 2400 ssh -o ConnectTimeout=10 "$n" "
      f=/home/$n/g5-rebuild/glm5_next_stage.tp8.rank$nn.g5nsp/glm5_next_stage.tp8.rank$r.g5nsp
      if [ -f \$f ]; then
        sudo chattr -i /home/$n/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank$nn.g5nsp 2>/dev/null
        sudo cp \$f /home/$n/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank$nn.g5nsp &&
        sudo chattr +i /home/$n/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank$nn.g5nsp &&
        echo RANK$r-PLACED
      else
        echo RANK$r-BUILD-NOT-READY
      fi
    " >> "/tmp/g5_ship_$n.log" 2>&1
    echo "$n rc=$?" >> /tmp/g5_ship_fleet.log
  ) &
  while [ "$(jobs -r | wc -l)" -ge 5 ]; do sleep 10; done
done
wait
echo PLACEMENT-WAVE-DONE
