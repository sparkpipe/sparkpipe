#!/bin/sh
cd /home/sparke/build-kernels-w5/src
export PATH=/usr/local/cuda/bin:$PATH
O=/home/sparke/build-kernels-w5/results/$1
B=${2:-./build/glm5_next_batch_roofline}
mkdir -p $O
nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv,noheader > $O/gpu_before.txt 2>&1
for b in 1 8; do
nsys profile -f true -o $O/b${b} --trace=cuda --cuda-graph-trace=node $B --batches $b --context 1024 --iterations 30 --graph 1 --route-readback 0 > $O/b${b}_run.txt 2>&1
nsys stats -f csv -o $O/b${b} --report cuda_gpu_trace $O/b${b}.nsys-rep > /dev/null 2>&1
done
ls $O
