#!/bin/sh
export PATH=/usr/local/cuda/bin:$PATH
cd ~/hy4-gpu || exit 1
nvcc -O2 -arch=sm_121 -fmad=false -o hy4_hc_test hy4_hc_test.cu 2> hc_build_err.txt
if [ $? -ne 0 ]; then
  echo "BUILD FAIL"
  head -40 hc_build_err.txt
  exit 1
fi
echo "BUILD OK"
gguf=$(ls ~/hy4-allranks/rank-00/model-*.gguf | head -1)
echo "pack: $gguf"
./hy4_hc_test "$gguf" 2>&1
rc=$?
echo "run rc=$rc"
