#!/bin/sh
cd "$HOME/hy4-cmp" || exit 1
D="$HOME/hy4-anchor"
if [ -f "$D/cpu/cpu.state.done" ]; then
  echo "GEN DONE ALREADY"
  exit 0
fi
if [ ./hy4_generate -nt ./hy4_generate.c ]; then
  echo "BIN FRESH"
else
  cc -O2 -o hy4_generate hy4_generate.c hy4_rank_loader.c -I . -lm 2> gen_build_err.txt
  if [ $? -ne 0 ]; then
    echo "BUILD FAIL"
    head -40 gen_build_err.txt
    exit 1
  fi
fi
n=$(ls -d "$HOME/hy4-allranks"/rank-* 2>/dev/null | wc -l)
if [ "$n" -ne 16 ]; then
  echo "RANK DIRS $n != 16"
  exit 1
fi
mkdir -p "$D/cpu"
if [ ! -f "$D/cpu/cpu.state" ]; then
  rm -f "$D/cpu"/cpu_L*_t*
  rm -f "$HOME/hy4-cmp/gen_anchor.log"
  printf '802 5466 19405 63357\n' > "$D/prompt.txt"
fi
export HY4_CKPT="$D/cpu/cpu.state"
./hy4_generate "$HOME/hy4-allranks" 1 "$D/prompt.txt" "$D/cpu/cpu" >> "$HOME/hy4-cmp/gen_anchor.log" 2>&1
rc=$?
echo "GEN RC=$rc"
exit $rc
