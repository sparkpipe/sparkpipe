#!/bin/sh
cd /home/sparkf/build-glmflash-b1
export PATH=/usr/local/cuda/bin:$PATH
O=/home/sparkf/build-glmflash-b1/results/$1
B=$2
shift 2
mkdir -p $O
nsys profile -f true -o $O/b1 --trace=cuda --cuda-graph-trace=node ./bin/$B --batches 1 --context 1024 --iterations 30 --graph 1 --route-readback 0 "$@" > $O/b1_run.txt 2>&1
nsys stats -f csv -o $O/b1 --report cuda_gpu_trace $O/b1.nsys-rep > /dev/null 2>&1
rm -f $O/b1.sqlite
grep ROOFLINE-GRAPH $O/b1_run.txt
