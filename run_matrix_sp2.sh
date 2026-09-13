#!/bin/bash
# SparkPipe benchmark matrix driver - qwen38.fp8.tp1 @ spark3
set -u
ROOT=/home/spark3/sparkdata/qwen38.fp8.tp1
CELLS=/tmp/sp2_cells
OUTDIR=/tmp/sp2_cells/out
SUMMARY=/tmp/sp2_cells/matrix_summary.tsv
cd "$ROOT" || exit 1
mkdir -p "$OUTDIR"
printf 'cell\trc\twall_ms\tsubmitted\tadmitted\trejected\tbatch_status_line\n' > "$SUMMARY"

CELLS_LIST="ctx512_b1 ctx512_b4 ctx512_b16 ctx512_b1_prefill ctx2048_b1 ctx2048_b4 ctx2048_b16 ctx2048_b1_prefill"

for cell in $CELLS_LIST; do
  echo "=== CELL $cell start $(date -Is) ==="
  pkill -f 'sparkpipe_model_resident[d]' || true
  sleep 5
  bash /tmp/launch_sp3.sh >/dev/null 2>&1
  up=0
  for i in $(seq 1 120); do
    if ss -ltn 2>/dev/null | grep -q ':17480 '; then up=1; break; fi
    sleep 2
  done
  if [ "$up" != "1" ]; then
    printf '%s\tPORT_TIMEOUT\t\t\t\t\t\n' "$cell" >> "$SUMMARY"
    continue
  fi
  sleep 3
  echo "=== CELL $cell daemon up $(date -Is), running batch ==="
  t0=$(date +%s%N)
  ./bin/sparkpipe_model_batch \
      --deployment config/model_resident.json \
      --runtime-root . \
      --batch "$CELLS/${cell}.sparkbatch.json" \
      > "$OUTDIR/${cell}.out" 2> "$OUTDIR/${cell}.err"
  rc=$?
  t1=$(date +%s%N)
  wall_ms=$(( (t1 - t0) / 1000000 ))
  pipe_line=$(grep -m1 'batch_pipeline' "$OUTDIR/${cell}.err" || true)
  stat_line=$(grep -m1 'batch_status' "$OUTDIR/${cell}.err" || true)
  sub=$(printf '%s' "$pipe_line" | sed -n 's/.*submitted=\([0-9]*\).*/\1/p')
  adm=$(printf '%s' "$pipe_line" | sed -n 's/.*admitted=\([0-9]*\).*/\1/p')
  rej=$(printf '%s' "$pipe_line" | sed -n 's/.*rejected=\([0-9]*\).*/\1/p')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cell" "$rc" "$wall_ms" "${sub:-NA}" "${adm:-NA}" "${rej:-NA}" "${stat_line:-NA}" >> "$SUMMARY"
  echo "=== CELL $cell done rc=$rc wall_ms=${wall_ms}ms $(date -Is) ==="
done
echo MATRIX_DONE
echo "--- SUMMARY ---"
cat "$SUMMARY"
