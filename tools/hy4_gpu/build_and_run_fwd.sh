#!/bin/sh
export PATH=/usr/local/cuda/bin:$PATH
cd ~/hy4-gpu || exit 1
nvcc -O2 -arch=sm_121 -fmad=false -o hy4_forward_test hy4_forward_test.cu \
  ~/hy4-cmp/hy4_rank_loader.c -I ~/hy4-cmp 2> fwd_build_err.txt
if [ $? -ne 0 ]; then
  echo "BUILD FAIL"
  head -40 fwd_build_err.txt
  exit 1
fi
echo "BUILD OK"
rm -f fwd_test_run.log
setsid ./run_forward_test.sh </dev/null >/dev/null 2>&1 &
echo "RUN LAUNCHED"
