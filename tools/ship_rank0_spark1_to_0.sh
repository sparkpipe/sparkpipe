#!/bin/bash
# Ship the completed rank0 relay build from spark1 to spark0.
set -u
# Wait for the build to finish
while pgrep -f resident_stagepack > /dev/null 2>&1; do sleep 30; done
# Wait for the pack file to appear
for i in $(seq 1 60); do
  [ -f /home/spark1/g5-rank0-relay/glm5_next_stage.tp8.rank00.g5nsp/glm5_next_stage.tp8.rank0.g5nsp ] && break
  sleep 30
done
# Ship spark1 -> spark0
rsync -a --timeout 1800 /home/spark1/g5-rank0-relay/glm5_next_stage.tp8.rank00.g5nsp/glm5_next_stage.tp8.rank0.g5nsp \
  /home/spark1/g5-rank0-relay/glm5_next_stage.tp8.rank00.g5nsp.receipt.json \
  spark0:/home/spark0/sparkdata/glm5_next.tp8.fp8/packs/
echo "RANK0-SHIPPED at $(date)"
