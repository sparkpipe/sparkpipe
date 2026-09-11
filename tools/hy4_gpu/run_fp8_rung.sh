#!/bin/sh
cd "$HOME/hy4-gpu" || exit 1
P="$HOME/sparkdata/hy4.fp8.tp16/packs/rank-02/model-fp8-tp16-rank-02.safetensors"
M="$HOME/hy4-fp8inc"
D="$HOME/hy4-fp8rung6"
if [ -f "$D/ALL.done" ]; then
  echo "FP8 RUNG DONE ALREADY"
  exit 0
fi
for p in /proc/[0-9]*/exe; do
  t=$(readlink "$p" 2>/dev/null) || continue
  case "$t" in */hy4_fp8_rung) echo "LIVE INSTANCE $t"; exit 1;; esac
done
if [ hy4_fp8_rung -nt hy4_fp8_rung.cu ] && \
   [ hy4_fp8_rung -nt spark_hy4_resident_decode_stage_cuda.cu ]; then
  echo "BIN FRESH"
else
  export PATH=/usr/local/cuda/bin:$PATH
  nvcc -O2 -arch=sm_121 -fmad=false -o hy4_fp8_rung hy4_fp8_rung.cu \
    spark_hy4_resident_decode_stage_cuda.cu -I "$M" 2> fp8rung_build_err.txt
  if [ $? -ne 0 ]; then
    echo "BUILD FAIL"
    head -40 fp8rung_build_err.txt
    exit 1
  fi
fi
mkdir -p "$D/done"
python3 fp8_rung_manifest.py "$P" "$D/fp8_rung.manifest" 2 16 || exit 1
export HY4_FP8_CKPT="$D/ckpt"
export HY4_FP8_TSV="$D/fp8_rung.tsv"
./hy4_fp8_rung "$P" "$D/fp8_rung.manifest" "$D" >> fp8rung6.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q FP8_RUNG_DONE fp8rung6.log; then
  date > "$D/ALL.done"
fi
echo "FP8RUNG RC=$rc"
exit $rc
