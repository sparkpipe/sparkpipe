#!/bin/sh
export PATH=/usr/local/cuda/bin:$PATH
cd ~/hy4-gpu || exit 1
i=0
while [ $i -lt 28 ]; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -ge 8000 ]; then
    break
  fi
  sleep 30
  i=$((i + 1))
done
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free:-0}" -lt 8000 ]; then
  echo "GPU STILL BUSY free=${free}MB, aborting"
  exit 1
fi
nvcc -O2 -arch=sm_121 -fmad=false -o hy4_forward_test hy4_forward_test.cu \
  ~/hy4-cmp/hy4_rank_loader.c -I ~/hy4-cmp 2> fwd_build_err.txt
if [ $? -ne 0 ]; then
  echo "BUILD FAIL"
  head -40 fwd_build_err.txt
  exit 1
fi
echo "BUILD OK free=${free}MB"
rm -f fwd_test_run.log
setsid ./run_forward_test.sh </dev/null >/dev/null 2>&1 &
echo "RUN LAUNCHED"
