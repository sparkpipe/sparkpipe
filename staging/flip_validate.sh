#!/bin/bash
set -u
RR="$HOME/sparkdata/glm5_next.tp16"
LOG="$RR/residentd.log"
rm -f "$LOG" "$RR/flip_validate.out"
cd "$RR"
env LD_LIBRARY_PATH="$RR/lib" nohup ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index 0 > residentd.log 2>&1 < /dev/null &
RPID=$!
sleep 8
for i in $(seq 1 24); do
    if grep -qE "hidden_spark_rdma_fabric_ready|hidden_spark_rdma_control_listen|SCHEMA|schema_error|invalid|usage_error|initialize.*busy|io_error" "$LOG" 2>/dev/null; then break; fi
    sleep 5
done
{
echo "== verdict markers =="
grep -cE "hidden_spark_rdma_fabric_ready" "$LOG" 2>/dev/null
grep -m3 -E "SCHEMA|schema|invalid_argument|usage|io_error|busy" "$LOG" 2>/dev/null | cut -c1-140
echo "== tail =="
tail -5 "$LOG" 2>/dev/null | cut -c1-140
} | tee "$RR/flip_validate.out"
for p in $(pgrep -f "bin/sparkpipe_model_[r]esidentd"); do c=$(readlink /proc/$p/cwd 2>/dev/null); [ "$c" = "$RR" ] && kill -TERM $p; done
sleep 3
echo "validate done $(date +%T)"
