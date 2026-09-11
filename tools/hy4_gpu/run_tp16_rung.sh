#!/bin/sh
ROOT="$HOME/hy4-tp16"
cd "$ROOT" || exit 1
export PATH=/usr/local/cuda/bin:$PATH
CUDA=/usr/local/cuda
SOCKET=/tmp/hy4-tp16-weightd.sock
if [ -f "$ROOT/TP16.done" ]; then
  echo "TP16 RUNG DONE ALREADY"
  exit 0
fi
for p in /proc/[0-9]*/exe; do
  t=$(readlink "$p" 2>/dev/null) || continue
  case "$t" in */hy4_tp16_rung) echo "LIVE INSTANCE $t"; exit 1;; esac
done
rm -f "$SOCKET"
make build/sparkpipe_weightd > tp16_daemon_build.log 2>&1
if [ $? -ne 0 ]; then
  echo "DAEMON BUILD FAIL"
  tail -30 tp16_daemon_build.log
  exit 1
fi
echo "DAEMON BUILD OK"
GCCFLAGS="-O2 -std=c11 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -I $ROOT/include -I $CUDA/include"
gcc $GCCFLAGS -c ring/transport/tp_device_collective.c -o tp16_tdc.o 2> tp16_build_err.txt || { echo "ENGINE BUILD FAIL"; head -40 tp16_build_err.txt; exit 1; }
NVCCFLAGS="-O2 -arch=sm_121 -fmad=false -I $ROOT -I $ROOT/include -I $ROOT/model-families/hy4/include -I $ROOT/modules/hy4_resident_decode_stage/include -I $ROOT/modules/hy4_resident_decode_stage/source -I $CUDA/include"
nvcc $NVCCFLAGS -c tools/hy4_gpu/hy4_tp16_rung.cu -o tp16_rung.o 2>> tp16_build_err.txt || { echo "RUNG BUILD FAIL"; head -40 tp16_build_err.txt; exit 1; }
nvcc $NVCCFLAGS -c modules/hy4_resident_decode_stage/source/spark_hy4_resident_decode_stage_cuda.cu -o tp16_hy4cuda.o 2>> tp16_build_err.txt || { echo "HY4CUDA BUILD FAIL"; head -40 tp16_build_err.txt; exit 1; }
g++ -O2 tp16_rung.o tp16_hy4cuda.o tp16_tdc.o build/libsparkpipe_runtime.a build/libsparkpipe_core.a -L $CUDA/lib64 -lcudart -lcuda -lpthread -o hy4_tp16_rung 2>> tp16_build_err.txt || { echo "LINK FAIL"; head -40 tp16_build_err.txt; exit 1; }
echo "RUNG BUILD OK"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > tp16_gpu_before.txt 2>&1
"$ROOT/build/sparkpipe_weightd" --socket "$SOCKET" --device-bytes-max 1073741824 > tp16_weightd.log 2>&1 &
DAEMON_PID=$!
i=0
while [ ! -S "$SOCKET" ] && [ $i -lt 50 ]; do
  sleep 0.2
  i=$((i+1))
done
if [ ! -S "$SOCKET" ]; then
  echo "DAEMON SOCKET TIMEOUT"
  cat tp16_weightd.log
  kill $DAEMON_PID 2>/dev/null
  exit 1
fi
SPARK_WEIGHTD_SOCKET="$SOCKET" HY4_TP16_CUTOFF=240 HY4_TP16_TSV="$ROOT/tp16_rung.tsv" \
  ./hy4_tp16_rung
RC=$?
kill $DAEMON_PID 2>/dev/null
wait $DAEMON_PID 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > tp16_gpu_after.txt 2>&1
echo "GPU_BEFORE:"; cat tp16_gpu_before.txt
echo "GPU_AFTER:"; cat tp16_gpu_after.txt
echo "TP16RUNG RC=$RC"
if [ $RC -eq 0 ]; then
  date > "$ROOT/TP16.done"
fi
exit $RC
