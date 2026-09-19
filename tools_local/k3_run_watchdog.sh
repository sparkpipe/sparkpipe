#!/usr/bin/env bash
# K3 lane fixture watchdog: waits for run A, then runs B and byte-compares.
set -u
cd /home/spark7/lane-k3-t1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export K3_LAYER_TIMING=1

while pgrep -f "tools/t1_reference_decode[r]" >/dev/null; do
  sleep 30
done

if [ ! -f out_a/k3/MANIFEST.json ]; then
  echo "WATCHDOG: run A produced no MANIFEST.json" > watchdog_status.log
  exit 1
fi
echo "WATCHDOG: run A done, launching run B" > watchdog_status.log

rm -rf out_b run_b.log
setsid nohup python3 tools/t1_reference_decoder.py \
  --family k3 --checkpoint /home/spark7/lane-k3-t1/kimi-k3-local \
  --header model-families/k3/include/sparkpipe/llm_defines.h \
  --prompts prompts.json --output out_b > run_b.log 2>&1 < /dev/null &

B_PID=$!
while kill -0 "$B_PID" 2>/dev/null; do sleep 30; done

if cmp -s out_a/k3/capital_of_france.t1r out_b/k3/capital_of_france.t1r && \
   cmp -s out_a/k3/count_up.t1r out_b/k3/count_up.t1r; then
  echo "WATCHDOG: run A and run B fixtures BYTE-IDENTICAL" > watchdog_status.log
else
  echo "WATCHDOG: DIVERGENCE between run A and run B fixtures" > watchdog_status.log
fi
