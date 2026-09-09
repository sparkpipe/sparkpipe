#!/bin/sh
export PATH=/usr/local/cuda/bin:$PATH
cd "$HOME/hy4-gpu" || exit 1
for e in /proc/[0-9]*/exe; do
  t=$(readlink "$e" 2>/dev/null) || continue
  case "$t" in *hy4_forward_test*)
    echo "LIVE INSTANCE REFUSES: $t"
    exit 1 ;;
  esac
done
nvcc -O2 -arch=sm_121 -fmad=false -o hy4_forward_test hy4_forward_test.cu \
  "$HOME/hy4-cmp/hy4_rank_loader.c" -I "$HOME/hy4-cmp" 2> fwd_build_err.txt
if [ $? -ne 0 ]; then
  echo "BUILD FAIL"
  head -40 fwd_build_err.txt
  exit 1
fi
echo "BUILD OK"
n=$(ls -d "$HOME/hy4-allranks"/rank-* 2>/dev/null | wc -l)
if [ "$n" -ne 16 ]; then
  echo "RANK DIRS $n != 16"
  exit 1
fi
mkdir -p "$HOME/hy4-anchor/gpu"
rm -f "$HOME/hy4-gpu/fwd_anchor.log"
setsid ./run_forward_test.sh </dev/null >/dev/null 2>&1 &
echo "RUN LAUNCHED"
