#!/bin/sh
export PATH=/usr/local/cuda/bin:$PATH
cd ~/hy4-gpu || exit 1
nvcc -O2 -arch=sm_121 -fmad=false -o hy4_qchain_test hy4_qchain_test.cu 2> qchain_build_err.txt
if [ $? -ne 0 ]; then
  echo "BUILD FAIL"
  head -40 qchain_build_err.txt
  exit 1
fi
echo "BUILD OK"
gguf=$(ls ~/hy4-allranks/rank-00/model-*.gguf | head -1)
echo "pack: $gguf"
./hy4_qchain_test "$gguf" 2>&1
rc=$?
echo "run rc=$rc"
