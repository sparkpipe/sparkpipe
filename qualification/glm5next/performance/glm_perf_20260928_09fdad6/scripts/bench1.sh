#!/bin/sh
cd /home/sparkf/build-glm-perf
export PATH=/usr/local/cuda/bin:$PATH
O=results/$1
mkdir -p $O
nvidia-smi --query-gpu=utilization.gpu,clocks.sm --format=csv > $O/gpu_before.txt 2>&1
for g in 0 1; do ./build/glm5_next_batch_roofline --batches 1,8 --context 1024 --iterations 40 --graph $g > $O/ctx1024_graph$g.txt 2>&1; done
./build/glm5_next_batch_roofline --batches 1 --context 128 --iterations 40 --graph 1 > $O/ctx128_graph1.txt 2>&1
nsys profile -f true -o $O/b1_graph --trace=cuda --cuda-graph-trace=node ./build/glm5_next_batch_roofline --batches 1 --context 1024 --iterations 20 --graph 1 > $O/nsys_run.txt 2>&1
nsys stats -f csv -o $O/b1 --report cuda_gpu_kern_sum $O/b1_graph.nsys-rep > /dev/null 2>&1
nsys stats -f csv -o $O/b1 --report cuda_gpu_trace $O/b1_graph.nsys-rep > /dev/null 2>&1
ls $O
