#!/bin/sh
cd /home/sparke/build-glm-perf
export PATH=/usr/local/cuda/bin:$PATH
O=results/prof2
mkdir -p $O
for b in 1 8; do
nsys profile -f true -o $O/b${b} --trace=cuda --cuda-graph-trace=node ./glm5_next_batch_roofline --batches $b --context 1024 --iterations 30 --graph 1 --route-readback 0 > $O/b${b}_run.txt 2>&1
nsys stats -f csv -o $O/b${b} --report cuda_gpu_trace $O/b${b}.nsys-rep > /dev/null 2>&1
done
ls $O
