#!/bin/bash
# For the landed tp8 builds: verify, replace the placed pack, relock.
set -u
for pair in "spark0 00 0" "spark1 01 1" "spark2 02 2" "spark3 03 3" "spark4 04 4" "spark6 06 6" "spark7 07 7"; do
  n=${pair%% *}; rest=${pair#* }; nn=${rest%% *}; r=${rest##* }
  # the packer names the INNER file with the unpadded rank
  (
    timeout 3000 ssh -o ConnectTimeout=10 "$n" "
      sudo python3 ~/sparkpipe/tools/glm5_next_pack_verify.py \
        --pack ~/g5-rebuild/glm5_next_stage.tp8.rank$nn.g5nsp/glm5_next_stage.tp8.rank$r.g5nsp \
        --source /mnt/model-warm/glm-5.3-flash --tp-rank $r --tp-degree 8 &&
      sudo chattr -i ~/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank$nn.g5nsp 2>/dev/null
      sudo cp ~/g5-rebuild/glm5_next_stage.tp8.rank$nn.g5nsp/glm5_next_stage.tp8.rank$r.g5nsp \
        ~/sparkdata/glm5_next.tp8.fp8/packs/ &&
      sudo chattr +i ~/sparkdata/glm5_next.tp8.fp8/packs/glm5_next_stage.tp8.rank$nn.g5nsp &&
      echo RANK$nn-REPLACED-LOCKED
    " >> "/tmp/g5_ship_$n.log" 2>&1
    echo "$n rc=$?" >> /tmp/g5_ship_fleet.log
  ) &
  while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
echo SHIP-WAVE-DONE
