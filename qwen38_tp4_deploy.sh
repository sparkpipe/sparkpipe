#!/usr/bin/env bash
# qwen36 TP4 band deploy: spark0-3, simultaneous start, batch E2E on request
set -euo pipefail
RUNTIME_NAME=qwen38.bf16.tp4
BUILD=/home/sparkd/sparkdata/$RUNTIME_NAME/build
HOSTS=(spark0 spark1 spark2 spark3)
PACKS=/home/sparkd/sparkdata/$RUNTIME_NAME/packs
COMMIT=$(cd /home/sparkd/sparkpipe-qwen && git rev-parse HEAD)
GENERATION=$(date -u +%Y%m%d%H%M%S)
# TCP NCCL pinned to the 100G rail (enp1s0f1np1, 10.10.100.0/24) so the
# ring never selects the mgmt/wifi/tailscale interfaces.
NCCL_ENV="NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN"
if [[ "${1:-}" == "deploy" ]]; then
  i=0
  for host in "${HOSTS[@]}"; do
    rt=/home/$host/sparkdata/$RUNTIME_NAME
    ssh -o BatchMode=yes $host "mkdir -p $rt/bin $rt/lib $rt/config $rt/packs"
    rsync -a $PACKS/tp4-rank$i.qwen36sp $host:$rt/packs/tp4-rank$i.qwen36sp &
    rsync -a $BUILD/sparkpipe_model_residentd $BUILD/sparkpipe_model_batch $host:$rt/bin/ &
    rsync -a $BUILD/model_driver.so $BUILD/model_serving_adapter.so $BUILD/hidden_transport.so $BUILD/libnccl.so.2 $host:$rt/lib/ &
    ssh -o BatchMode=yes $host "cp $rt/lib/hidden_transport.so $rt/lib/libhidden_transport.so" &
    rsync -a $BUILD/qwen36_tp4_rank$i.json $host:$rt/config/qwen36_tp4_rank$i.json &
    rsync -a $BUILD/model_resident.json $host:$rt/config/ &
    wait
    echo "deployed $host rank$i"
    i=$((i+1))
  done
  echo DEPLOY-COMPLETE
fi
if [[ "${1:-}" == "start" ]]; then
  i=0
  for host in "${HOSTS[@]}"; do
    rt=/home/$host/sparkdata/$RUNTIME_NAME
    ssh -o BatchMode=yes $host "cd $rt && export LD_LIBRARY_PATH=$rt/lib:\$LD_LIBRARY_PATH $NCCL_ENV SPARKPIPE_RELEASE_GENERATION=$GENERATION SPARKPIPE_RELEASE_GIT_COMMIT=$COMMIT SPARKPIPE_RELEASE_ID=qwen38-tp4-rank$i SPARK_QWEN36_STAGE_MTP=1 SPARK_QWEN36_STAGE_GDN_SNAPSHOT_SLOTS=8 SPARK_QWEN36_SERVING_SPECULATE=1 SPARK_QWEN36_SERVING_SPECULATIVE_DRAFT_COUNT=2 && setsid -f bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index $i >/tmp/qwen38-tp4-rank$i.log 2>&1 </dev/null" &
    i=$((i+1))
  done
  wait
  echo STARTED-ALL-4
fi
if [[ "${1:-}" == "stop" ]]; then
  for host in "${HOSTS[@]}"; do
    ssh -o BatchMode=yes $host "pkill -f 'sparkpipe_model_residentd --deployment config/model_resident.json' || true" &
  done
  wait
  echo STOPPED-ALL-4
fi
if [[ "${1:-}" == "batch" ]]; then
  ssh -o BatchMode=yes spark0 "cd /home/spark0/sparkdata/$RUNTIME_NAME && export LD_LIBRARY_PATH=\$PWD/lib:\$LD_LIBRARY_PATH && bin/sparkpipe_model_batch --deployment config/model_resident.json --runtime-root \$PWD --batch /tmp/qwen38-batch.json"
fi
