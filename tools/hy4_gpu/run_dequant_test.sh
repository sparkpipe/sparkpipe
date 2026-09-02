#!/bin/sh
export PATH=/usr/local/cuda/bin:$PATH
cd ~/hy4-gpu || exit 1
nvcc -O2 -arch=sm_121 -fmad=false -o hy4_dequant_test hy4_dequant_test.cu 2> build_err.txt
if [ $? -ne 0 ]; then
  echo "BUILD FAIL"
  head -40 build_err.txt
  exit 1
fi
echo "BUILD OK"
gguf=$(ls ~/sparkdata/hy4.ud-iq1m.tp16/packs/rank-*/model-*.gguf | head -1)
echo "pack: $gguf"
./hy4_dequant_test "$gguf" 2>&1
rc=$?
echo "run rc=$rc"
