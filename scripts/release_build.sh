#!/usr/bin/env bash
set -u
REPO=${REPO:-/Users/mac/wf/integ}
SHA=$(cd $REPO && git rev-parse --short=9 ${1:-HEAD})
HOST=${BUILD_HOST:-sparkd}
LOCK=/Users/mac/sparkpipe-coord/locks/tp16.lock
OUT=/Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-release-$SHA
S=${RELEASE_WORK:-${TMPDIR:-/tmp}/sparkpipe-release}; mkdir -p "$S"
ADAPTERS="dsv41_flash gemma4_26b gemma4_31b glm53_flash glm53_full hy4 k3 laguna ling3_fin ling3_flash mimo26_flash mimo26_pro muse_glimmer qwen38_27b_bf16 qwen38_27b qwen38_flash qwen38_max"
SO="build/libk3_serving_adapter.so"; for a in $ADAPTERS; do SO="$SO build/lib${a}_tp16_serving_adapter.so"; done
NODE="build/sparkpipe_model_residentd build/weightd_warm build/libhidden_transport_spark_host_rdma_verbs.so build/sparkpipe_weightd build/weightd_receipt build/weightdctl build/sparkpipe_model_batch"
cd $REPO
COPYFILE_DISABLE=1 git archive --format=tar $SHA > $S/src-$SHA.tar || exit 1
scp -q $S/src-$SHA.tar rtx5090:rp-src-$SHA.tar || exit 1
ssh -o BatchMode=yes rtx5090 "rm -rf ~/api-build-$SHA && mkdir ~/api-build-$SHA && tar -C ~/api-build-$SHA -xf ~/rp-src-$SHA.tar && echo $SHA > ~/api-build-$SHA/SOURCE_COMMIT && cd ~/api-build-$SHA && export PATH=/usr/local/cuda/bin:\$PATH && make -j16 build/sparkpipe_model_api build/sparkpipe_tokenize_prompt $SO > ../api-$SHA.out 2>&1; echo API-RC=\$?" < /dev/null &
ssh -o BatchMode=yes $HOST "mkdir -p ~/build-rp" < /dev/null
scp -q $S/src-$SHA.tar $HOST:build-rp/src-$SHA.tar || exit 1
/usr/local/bin/flock -s -w 10800 $LOCK ssh -o BatchMode=yes $HOST "cd ~/build-rp && rm -rf fw-$SHA-src && mkdir fw-$SHA-src && tar -C fw-$SHA-src -xf src-$SHA.tar && echo $SHA > fw-$SHA-src/SOURCE_COMMIT && cd fw-$SHA-src && export PATH=/usr/local/cuda/bin:\$PATH && make -j8 $NODE $SO > ../fw-$SHA.out 2>&1; echo FW-RC=\$?" < /dev/null
wait
ssh -o BatchMode=yes $HOST "grep -n -E ' error|Error ' ~/build-rp/fw-$SHA.out | head -5" < /dev/null
ssh -o BatchMode=yes rtx5090 "grep -n -E ' error|Error ' ~/api-$SHA.out | head -5; cd ~/api-build-$SHA/build && sha256sum sparkpipe_model_api sparkpipe_tokenize_prompt *.so | cut -c1-16,65- > ../SHA16" < /dev/null
rm -rf $OUT && mkdir -p $OUT && ssh $HOST "cd ~/build-rp/fw-$SHA-src/build && tar cf - $(echo $NODE $SO | sed 's#build/##g')" < /dev/null | tar -C $OUT -xf - && cp $REPO/tools/weightd_spine_budget.py $OUT/ && echo $SHA > $OUT/SOURCE_COMMIT && (cd $OUT && shasum -a 256 $(echo $NODE $SO | sed 's#build/##g') | cut -c1-16,65- > SHA16 && shasum -a 256 $(echo $NODE $SO | sed "s#build/##g") weightd_spine_budget.py > SHA256SUMS) && wc -l < $OUT/SHA16 && echo FW-DIR $OUT
