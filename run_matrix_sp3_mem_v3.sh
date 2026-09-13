#!/bin/bash
# SparkPipe pass-2: full matrix + per-driver memory capture (residentd & batch client)
set -u
ROOT=/home/spark3/sparkdata/qwen38.fp8.tp1
CELLS=/tmp/sp2_cells
OUTDIR=/tmp/sp2_cells/out
MEMDIR=/tmp/sp2_cells/mem
SUMMARY=/tmp/sp2_cells/pass3_summary_v3.tsv
cd "$ROOT" || exit 1
mkdir -p "$OUTDIR" "$MEMDIR"
printf 'cell\trc\twall_ms\tsubmitted\tadmitted\trejected\tstatus_line\trd_peak_rss_mb\trd_hwm_mb\tbat_peak_rss_mb\tbat_hwm_mb\trd_gpu_mib\tbat_gpu_mib\tguard_waits\tattempts\n' > "$SUMMARY"

CELLS_LIST="${CELLS_OVERRIDE:-ctx512_b1 ctx512_b4 ctx512_b16 ctx512_b1_prefill ctx2048_b1 ctx2048_b4 ctx2048_b16 ctx2048_b1_prefill}"

gpu_of_pid() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v p="$1" '$1==p{print $2; f=1} END{if(!f) print 0}'
}

for cell in $CELLS_LIST; do
  echo "=== CELL $cell start $(date -Is) ==="
  # contention guard: let any foreign batch client finish (max 15 min)
  gw=0
  for i in $(seq 1 90); do
    pgrep -f 'bin/sparkpipe_model_batc[h]' >/dev/null 2>&1 || break
    gw=$((gw+1)); sleep 10
  done
  [ "$gw" -gt 0 ] && echo "=== CELL $cell waited ${gw}0s for foreign batch client ==="

  pkill -f 'sparkpipe_model_resident[d]' || true
  sleep 5
  bash /tmp/launch_sp3.sh >/dev/null 2>&1
  up=0
  for i in $(seq 1 120); do
    ss -ltn 2>/dev/null | grep -q ':17480 ' && { up=1; break; }
    sleep 2
  done
  if [ "$up" != "1" ]; then
    printf '%s\tPORT_TIMEOUT\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t%s\t0\n' "$cell" "$gw" >> "$SUMMARY"
    continue
  fi
  sleep 3

  ok=0; attempt=0
  while [ "$ok" != "1" ] && [ "$attempt" -lt 5 ]; do
    attempt=$((attempt+1))
    [ "$attempt" -gt 1 ] && sleep 20
    mcsv="$MEMDIR/${cell}.csv"
    [ -f "$mcsv" ] && mv "$mcsv" "$MEMDIR/${cell}_a$((attempt-1)).csv"
    echo "--- $cell attempt $attempt $(date -Is)"
    pkill -f 'sparkpipe_model_resident[d]' || true
    sleep 5
    bash /tmp/launch_sp3.sh >/dev/null 2>&1
    up=0
    for i in $(seq 1 120); do
      ss -ltn 2>/dev/null | grep -q ':17480 ' && { up=1; break; }
      sleep 2
    done
    if [ "$up" != "1" ]; then echo "--- $cell attempt $attempt daemon-port-timeout"; continue; fi
    sleep 3
    rpid=$(pgrep -f 'bin/sparkpipe_model_resident[d]' | head -1)
    echo "ts,kind,pid,rss_kb,hwm_kb,gpu_mib" > "$mcsv"
    (
      while :; do
        ts=$(date +%s)
        if [ -n "$rpid" ] && [ -d "/proc/$rpid" ]; then
          rss=$(awk '/^VmRSS/{print $2}' "/proc/$rpid/status" 2>/dev/null)
          hwm=$(awk '/^VmHWM/{print $2}' "/proc/$rpid/status" 2>/dev/null)
          echo "$ts,rd,$rpid,${rss:-0},${hwm:-0},$(gpu_of_pid "$rpid")" >> "$mcsv"
        fi
        bpid=$(pgrep -f 'bin/sparkpipe_model_batc[h]' | head -1)
        if [ -n "$bpid" ] && [ -d "/proc/$bpid" ]; then
          brss=$(awk '/^VmRSS/{print $2}' "/proc/$bpid/status" 2>/dev/null)
          bhwm=$(awk '/^VmHWM/{print $2}' "/proc/$bpid/status" 2>/dev/null)
          echo "$ts,bat,$bpid,${brss:-0},${bhwm:-0},$(gpu_of_pid "$bpid")" >> "$mcsv"
        fi
        sleep 1
      done
    ) &
    sampler=$!
    t0=$(date +%s%N)
    ./bin/sparkpipe_model_batch \
        --deployment config/model_resident.json \
        --runtime-root . \
        --batch "$CELLS/${cell}.sparkbatch.json" \
        > "$OUTDIR/${cell}.out" 2> "$OUTDIR/${cell}.err"
    rc=$?
    t1=$(date +%s%N)
    wall_ms=$(( (t1 - t0) / 1000000 ))
    kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
    pipe_line=$(grep -m1 'batch_pipeline' "$OUTDIR/${cell}.err" || true)
    stat_line=$(grep -m1 'batch_status' "$OUTDIR/${cell}.err" || true)
    sub=$(printf '%s' "$pipe_line" | sed -n 's/.*submitted=\([0-9]*\).*/\1/p')
    adm=$(printf '%s' "$pipe_line" | sed -n 's/.*admitted=\([0-9]*\).*/\1/p')
    rej=$(printf '%s' "$pipe_line" | sed -n 's/.*rejected=\([0-9]*\).*/\1/p')
    if [ "$rc" = "0" ] && printf '%s' "$stat_line" | grep -q 'status=0'; then ok=1; fi
  done
  rd_rss_max=$(awk -F, '$2=="rd"{if($4>m)m=$4} END{print int(m/1024)}' "$mcsv")
  rd_hwm=$(awk -F, '$2=="rd"{if($5>m)m=$5} END{print int(m/1024)}' "$mcsv")
  bat_rss_max=$(awk -F, '$2=="bat"{if($4>m)m=$4} END{print int(m/1024)}' "$mcsv")
  bat_hwm=$(awk -F, '$2=="bat"{if($5>m)m=$5} END{print int(m/1024)}' "$mcsv")
  rd_gpu=$(awk -F, '$2=="rd"{if($6>m)m=$6} END{print int(m)}' "$mcsv")
  bat_gpu=$(awk -F, '$2=="bat"{if($6>m)m=$6} END{print int(m)}' "$mcsv")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$cell" "$rc" "$wall_ms" "${sub:-NA}" "${adm:-NA}" "${rej:-NA}" "${stat_line:-NA}" \
    "${rd_rss_max:-0}" "${rd_hwm:-0}" "${bat_rss_max:-0}" "${bat_hwm:-0}" "${rd_gpu:-0}" "${bat_gpu:-0}" "$gw" >> "$SUMMARY"
  echo "=== CELL $cell FINAL rc=$rc wall_ms=${wall_ms}ms attempts=$attempt $(date -Is) ==="
done
echo MATRIX_DONE
echo "--- PASS2 SUMMARY ---"
cat "$SUMMARY"