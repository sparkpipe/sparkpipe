#!/bin/sh
ROOT="$HOME/hy4-tp16"
cd "$ROOT" || exit 1
export PATH=/usr/local/cuda/bin:$PATH
CUDA=/usr/local/cuda
SOCKET=/tmp/hy4-tp16-serving-weightd.sock
MODULE_ARCHIVE=build/modules/hy4_resident_decode_stage/libhy4_resident_decode_stage.a
if [ -f "$ROOT/TP16_SERVING.done" ]; then
  echo "TP16 SERVING RUNG DONE ALREADY"
  exit 0
fi
for p in /proc/[0-9]*/exe; do
  t=$(readlink "$p" 2>/dev/null) || continue
  case "$t" in */hy4_tp16_serving_rung) echo "LIVE INSTANCE $t"; exit 1;; esac
done
rm -f "$SOCKET"
make build/sparkpipe_weightd > serving_daemon_build.log 2>&1
if [ $? -ne 0 ]; then
  echo "DAEMON BUILD FAIL"
  tail -30 serving_daemon_build.log
  exit 1
fi
echo "DAEMON BUILD OK"
make build/libsparkpipe_transport.a build/libsparkpipe_runtime.a \
  build/libsparkpipe_model_common.a build/libsparkpipe_core.a \
  > serving_lib_build.log 2>&1
if [ $? -ne 0 ]; then
  echo "LIB BUILD FAIL"
  tail -30 serving_lib_build.log
  exit 1
fi
make -C modules/hy4_resident_decode_stage archive >> serving_lib_build.log 2>&1
if [ $? -ne 0 ]; then
  echo "MODULE ARCHIVE FAIL"
  tail -30 serving_lib_build.log
  exit 1
fi
echo "MODULE ARCHIVE OK"
GCCFLAGS="-O2 -std=c11 -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -I $ROOT/include -I $ROOT/model-families/hy4/include -I $ROOT/modules/hy4_resident_decode_stage/include -I $ROOT/modules/hy4_resident_decode_stage/source -I $CUDA/include"
gcc $GCCFLAGS -c tools/hy4_gpu/hy4_tp16_serving_rung.c -o tp16_serving_rung.o 2> serving_build_err.txt || { echo "RUNG BUILD FAIL"; head -40 serving_build_err.txt; exit 1; }
g++ -O2 -Wl,--no-undefined tp16_serving_rung.o $MODULE_ARCHIVE \
  build/libsparkpipe_transport.a build/libsparkpipe_runtime.a \
  build/libsparkpipe_model_common.a build/libsparkpipe_core.a \
  -L $CUDA/lib64 -lcudart -lcuda -lpthread -ldl -lm -o hy4_tp16_serving_rung 2>> serving_build_err.txt || { echo "RUNG LINK FAIL"; head -40 serving_build_err.txt; exit 1; }
echo "RUNG BUILD OK"
cc -shared -fPIC -Wl,--no-undefined $MODULE_ARCHIVE \
  build/libsparkpipe_transport.a build/libsparkpipe_runtime.a \
  build/libsparkpipe_model_common.a build/libsparkpipe_core.a \
  -L $CUDA/lib64 -lcudart -lcuda -lpthread -ldl -lm \
  -o build/hy4_resident_decode_stage.so 2>> serving_build_err.txt || { echo "MODULE SO LINK FAIL"; head -40 serving_build_err.txt; exit 1; }
nm -D --defined-only build/hy4_resident_decode_stage.so > serving_so_symbols.txt 2>&1
grep -q "T SparkHy4ResidentDecodeStageInitialize" serving_so_symbols.txt || { echo "MODULE SO MISSING INITIALIZE"; exit 1; }
grep -q "T SparkTpDeviceCollectiveCreate" serving_so_symbols.txt || { echo "MODULE SO MISSING COLLECTIVE PATH"; exit 1; }
ls -l build/hy4_resident_decode_stage.so
echo "MODULE SO LINK OK"
if [ "$TP16_SERVING_LINK_ONLY" = "1" ]; then
  echo "TP16 SERVING LINK ONLY OK"
  exit 0
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > serving_gpu_before.txt 2>&1
"$ROOT/build/sparkpipe_weightd" --socket "$SOCKET" --device-bytes-max 1073741824 > serving_weightd.log 2>&1 &
DAEMON_PID=$!
i=0
while [ ! -S "$SOCKET" ] && [ $i -lt 50 ]; do
  sleep 0.2
  i=$((i+1))
done
if [ ! -S "$SOCKET" ]; then
  echo "DAEMON SOCKET TIMEOUT"
  cat serving_weightd.log
  kill $DAEMON_PID 2>/dev/null
  exit 1
fi
SPARK_WEIGHTD_SOCKET="$SOCKET" HY4_TP16_SERVING_TSV="$ROOT/tp16_serving_rung.tsv" \
  HY4_TP16_SERVING_WORKDIR="$ROOT" ./hy4_tp16_serving_rung
RC=$?
kill $DAEMON_PID 2>/dev/null
wait $DAEMON_PID 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > serving_gpu_after.txt 2>&1
echo "GPU_BEFORE:"; cat serving_gpu_before.txt
echo "GPU_AFTER:"; cat serving_gpu_after.txt
echo "TP16 SERVING RUNG RC=$RC"
if [ $RC -eq 0 ]; then
  date > "$ROOT/TP16_SERVING.done"
fi
exit $RC
