#!/bin/bash
tree=$1
cd ~/build-rp/$tree || exit 1
export PATH=/usr/local/cuda/bin:$PATH
NODE="build/sparkpipe_model_residentd build/weightd_warm build/libhidden_transport_spark_host_rdma_verbs.so build/sparkpipe_weightd build/weightd_receipt build/weightdctl build/sparkpipe_model_batch"
SO=$(for id in $(python3 -c "import json; print(' '.join(sorted(json.load(open('deployment/lanes.json'))['lanes'])))"); do
  set -- $(python3 tools/lane_config.py release-row "$id") && echo "build/$2" || echo "RELEASE-ROW-FAILED:$id"; done | sort -u | tr '\n' ' ')
case "$SO" in *RELEASE-ROW-FAILED*) echo "$tree: adapter rows failed: $SO"; exit 1;; esac
echo "$tree: adapters $SO" > ../$tree.targets
start=$(date +%s)
make -j8 $NODE $SO > ../$tree.out 2>&1
rc=$?
echo "$tree: make exit $rc after $(( $(date +%s) - start )) s, $(echo $SO | wc -w) adapters"
grep -m5 -E "error|Error" ../$tree.out
