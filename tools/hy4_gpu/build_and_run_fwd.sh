#!/bin/sh
export PATH=/usr/local/cuda/bin:$PATH
cd "$HOME/hy4-gpu" || exit 1
D="$HOME/hy4-anchor"
if [ -f "$D/gpu/gpu.state.done" ]; then
  echo "FWD DONE ALREADY"
  exit 0
fi
if [ ./hy4_forward_test -nt ./hy4_forward_test.cu ]; then
  echo "BIN FRESH"
else
  nvcc -O2 -arch=sm_121 -fmad=false -o hy4_forward_test hy4_forward_test.cu \
    "$HOME/hy4-cmp/hy4_rank_loader.c" -I "$HOME/hy4-cmp" 2> fwd_build_err.txt
  if [ $? -ne 0 ]; then
    echo "BUILD FAIL"
    head -40 fwd_build_err.txt
    exit 1
  fi
fi
n=$(ls -d "$HOME/hy4-allranks"/rank-* 2>/dev/null | wc -l)
if [ "$n" -ne 16 ]; then
  echo "RANK DIRS $n != 16"
  exit 1
fi
mkdir -p "$D/gpu"
if [ ! -f "$D/gpu/gpu.state" ]; then
  rm -f "$D/gpu"/gpu_L*_t*
  rm -f fwd_anchor.log
fi
export HY4_CKPT="$D/gpu/gpu.state"
export HY4_DUMP_PFX="$D/gpu/gpu"
./hy4_forward_test "$HOME/hy4-allranks" -1 >> fwd_anchor.log 2>&1
rc=$?
echo "FWD RC=$rc"
exit $rc
