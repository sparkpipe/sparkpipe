#!/usr/bin/env bash
set -u
REV=${1:-HEAD}
VARIANT=${2:-release}
S=${LING_SCRATCH:?LING_SCRATCH is a local scratch directory}
LEAD_FW=/Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-72eaa5fa3
cd /Users/mac/wf/drv-ling
SHA=$(git rev-parse --short=9 $REV) || exit 1
TAG=$SHA; FLAGS=""
if [ "$VARIANT" = debug ]; then TAG=$SHA-debug; FLAGS='NVCCFLAGS=-O3 -gencode arch=compute_121a,code=sm_121a -DDEBUG'; fi
SO="build/libling3_flash_tp16_serving_adapter.so build/libling3_fin_tp16_serving_adapter.so"
COPYFILE_DISABLE=1 git archive --format=tar $SHA > $S/src-$SHA.tar || exit 1
ssh -o BatchMode=yes sparka "mkdir -p ~/build-sprint-ling" < /dev/null || exit 1
scp -q $S/src-$SHA.tar sparka:build-sprint-ling/src-$SHA.tar || exit 1
ssh -o BatchMode=yes sparka "cd ~/build-sprint-ling && rm -rf $TAG && mkdir $TAG && tar -C $TAG -xf src-$SHA.tar && echo $SHA > $TAG/SOURCE_COMMIT && cd $TAG && export PATH=/usr/local/cuda/bin:\$PATH && make -j8 ${FLAGS:+\"$FLAGS\"} $SO > ../build-$TAG.out 2>&1; echo BUILD-RC=\$?; grep -n -E ' error|Error ' ../build-$TAG.out | head -5" < /dev/null
FW=$S/fw-ling-$TAG
rm -rf $FW && mkdir -p $FW && cp $LEAD_FW/sparkpipe_model_residentd $LEAD_FW/weightd_warm $LEAD_FW/libhidden_transport_spark_host_rdma_verbs.so $FW/ || exit 1
ssh -o BatchMode=yes sparka "cd ~/build-sprint-ling/$TAG/build && tar cf - $(echo $SO | sed 's#build/##g')" < /dev/null | tar -C $FW -xf - || exit 1
echo "$SHA" > $FW/SOURCE_COMMIT
ls -la $FW; echo FW-DIR $FW
