#!/bin/sh
export PATH=/usr/local/cuda/bin:$PATH
cd ~/hy4-gpu || exit 1
nvcc -O2 -arch=sm_121 -fmad=false -o hy4_layer_test hy4_layer_test.cu \
  ~/hy4-cmp/hy4_rank_loader.c -I ~/hy4-cmp 2> layer_build_err.txt
if [ $? -ne 0 ]; then
  echo "BUILD FAIL"
  head -40 layer_build_err.txt
  exit 1
fi
echo "BUILD OK"
./hy4_layer_test ~/hy4-allranks ~/hy4-cmp/l1state.t0 2>&1
rc=$?
echo "run rc=$rc"
