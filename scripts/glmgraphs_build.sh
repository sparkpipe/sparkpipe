#!/usr/bin/env bash
set -u
REPO=/Users/mac/wf/drv-glmgraphs
SHA=$(cd $REPO && git rev-parse --short=9 ${1:-HEAD})
S=/private/tmp/claude-502/-Users-mac-sparkpipe/df54bdc1-84bf-4584-9653-5211cf17258e/scratchpad/glmgraphs
BASE=/Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-2aa246576
FW=/Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-glmgraphs-$SHA
SO=build/libglm53_full_tp16_serving_adapter.so
EAGER=/Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-glmgraphs-eager-2aa246576
SSH="ssh -o BatchMode=yes"
mkdir -p $S
cd $REPO
COPYFILE_DISABLE=1 git archive --format=tar $SHA > $S/src-$SHA.tar || exit 1
scp -q $S/src-$SHA.tar spark8:build-sprint-glmgraphs-src-$SHA.tar || exit 1
scp -q $S/src-$SHA.tar rtx5090:apisrc-glmgraphs-$SHA.tar || exit 1
$SSH rtx5090 "rm -rf ~/api-build-glmgraphs-$SHA && mkdir ~/api-build-glmgraphs-$SHA && tar -C ~/api-build-glmgraphs-$SHA -xf ~/apisrc-glmgraphs-$SHA.tar && echo $SHA > ~/api-build-glmgraphs-$SHA/SOURCE_COMMIT && cd ~/api-build-glmgraphs-$SHA && export PATH=/usr/local/cuda/bin:\$PATH && make -j8 build/sparkpipe_model_api build/sparkpipe_tokenize_prompt $SO > ../api-glmgraphs-$SHA.out 2>&1; echo API-RC=\$?" < /dev/null &
$SSH spark8 "mkdir -p ~/build-sprint-glmgraphs && cd ~/build-sprint-glmgraphs && rm -rf $SHA && mkdir $SHA && tar -C $SHA -xf ~/build-sprint-glmgraphs-src-$SHA.tar && echo $SHA > $SHA/SOURCE_COMMIT && cd $SHA && export PATH=/usr/local/cuda/bin:\$PATH && make -j8 $SO build/weightd_warm > ../fw-$SHA.out 2>&1; echo FW-RC=\$?; make -j8 build/test_model_resident_deployment > ../test-$SHA.out 2>&1 && ./build/test_model_resident_deployment >> ../test-$SHA.out 2>&1; echo DEPLOY-TEST-RC=\$?" < /dev/null
wait
$SSH spark8 "grep -n -E 'error|warning' ~/build-sprint-glmgraphs/fw-$SHA.out | head -8; tail -2 ~/build-sprint-glmgraphs/test-$SHA.out" < /dev/null
$SSH rtx5090 "grep -n -E 'error' ~/api-glmgraphs-$SHA.out | head -5" < /dev/null
rm -rf $FW && mkdir -p $FW || exit 1
cp $BASE/sparkpipe_model_residentd $BASE/libhidden_transport_spark_host_rdma_verbs.so $FW/ || exit 1
scp -q spark8:build-sprint-glmgraphs/$SHA/$SO spark8:build-sprint-glmgraphs/$SHA/build/weightd_warm $FW/ || exit 1
rm -rf $EAGER && mkdir -p $EAGER && cp $BASE/sparkpipe_model_residentd $BASE/libhidden_transport_spark_host_rdma_verbs.so $BASE/libglm53_full_tp16_serving_adapter.so $BASE/SOURCE_COMMIT $FW/weightd_warm $EAGER/ || exit 1
(cd $EAGER && shasum -a 256 sparkpipe_model_residentd weightd_warm libhidden_transport_spark_host_rdma_verbs.so libglm53_full_tp16_serving_adapter.so > SHA256SUMS.txt)
echo $SHA > $FW/SOURCE_COMMIT
(cd $FW && shasum -a 256 sparkpipe_model_residentd weightd_warm libhidden_transport_spark_host_rdma_verbs.so libglm53_full_tp16_serving_adapter.so > SHA256SUMS.txt)
cat $FW/SHA256SUMS.txt
echo FW-DIR $FW API-DIR rtx5090:~/api-build-glmgraphs-$SHA/build
