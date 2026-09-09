#!/bin/sh
cd "$HOME/hy4-cmp" || exit 1
for e in /proc/[0-9]*/exe; do
  t=$(readlink "$e" 2>/dev/null) || continue
  case "$t" in *hy4_generate*)
    echo "LIVE INSTANCE REFUSES: $t"
    exit 1 ;;
  esac
done
cc -O2 -o hy4_generate hy4_generate.c hy4_rank_loader.c -I . -lm 2> gen_build_err.txt
if [ $? -ne 0 ]; then
  echo "BUILD FAIL"
  head -40 gen_build_err.txt
  exit 1
fi
echo "BUILD OK"
n=$(ls -d "$HOME/hy4-allranks"/rank-* 2>/dev/null | wc -l)
if [ "$n" -ne 16 ]; then
  echo "RANK DIRS $n != 16"
  exit 1
fi
mkdir -p "$HOME/hy4-anchor/cpu"
printf '802 5466 19405 63357\n' > "$HOME/hy4-anchor/prompt.txt"
rm -f "$HOME/hy4-cmp/gen_anchor.log"
setsid ./hy4_generate "$HOME/hy4-allranks" 1 "$HOME/hy4-anchor/prompt.txt" \
  "$HOME/hy4-anchor/cpu/cpu" </dev/null >> "$HOME/hy4-cmp/gen_anchor.log" 2>&1 &
echo "GEN LAUNCHED"
