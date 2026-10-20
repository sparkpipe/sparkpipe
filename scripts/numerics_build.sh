#!/usr/bin/env bash
set -u
REV=${1:-HEAD}
S=${NUMERICS_SCRATCH:?NUMERICS_SCRATCH is a local scratch directory}
BASE_FW=/Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-2aa246576
OUT=/Users/mac/sparkpipe-coord/lanes/driver-sprint
cd /Users/mac/wf/drv-numerics
SHA=$(git rev-parse --short=9 $REV) || exit 1
ADAPTERS="dsv41_flash gemma4_26b gemma4_31b glm53_flash glm53_full hy4 k3 laguna ling3_fin ling3_flash muse_glimmer qwen38_27b_bf16 qwen38_27b qwen38_flash qwen38_max"
SO=""; for a in $ADAPTERS; do SO="$SO build/lib${a}_tp16_serving_adapter.so"; done
COPYFILE_DISABLE=1 git archive --format=tar $SHA > $S/src-$SHA.tar || exit 1
ssh -o BatchMode=yes sparka "mkdir -p ~/build-sprint-numerics" < /dev/null || exit 1
scp -q $S/src-$SHA.tar sparka:build-sprint-numerics/src-$SHA.tar || exit 1
ssh -o BatchMode=yes sparka "cd ~/build-sprint-numerics && rm -rf $SHA && mkdir $SHA && tar -C $SHA -xf src-$SHA.tar && echo $SHA > $SHA/SOURCE_COMMIT && cd $SHA && export PATH=/usr/local/cuda/bin:\$PATH && nice -n 10 make -j8 $SO build/tp_mesh_hardware_probe > ../build-$SHA.out 2>&1; echo BUILD-RC=\$?; grep -n -E ' error|Error ' ../build-$SHA.out | head -5" < /dev/null
FW=$OUT/fw-numerics-$SHA
rm -rf $FW && mkdir -p $FW && cp $BASE_FW/sparkpipe_model_residentd $BASE_FW/weightd_warm $BASE_FW/libhidden_transport_spark_host_rdma_verbs.so $FW/ || exit 1
ssh -o BatchMode=yes sparka "cd ~/build-sprint-numerics/$SHA/build && tar cf - $(echo $SO | sed 's#build/##g')" < /dev/null | tar -C $FW -xf - || exit 1
echo "$SHA" > $FW/SOURCE_COMMIT
ls $FW | wc -l; echo FW-DIR $FW
