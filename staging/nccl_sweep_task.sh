#!/bin/bash
# 16-rank NCCL allreduce size sweep (glm5next lane, glm53-flash dev).
# Runs on spark5 (queue task nodes[0]); fans out via burst_fanout5.sh.
# Spec: bytes:iters:burst   (burst=0 sync-per-iter latency mode, 1 deep-enqueue)
# One "op" in bench output = one iter = cA+cB collective pair (dual=1),
# so per-collective = per_op_us / 2.
set -u
DEADLINE=$(( $(date +%s) + 700 ))
LAT="8192:2000:0 65536:1000:0"
SWEEP="8192:2000:1 32768:2000:1 65536:1000:1 131072:1000:1 262144:1000:1 524288:500:1 1048576:500:1 4194304:200:1 16777216:100:1"
SPECS="${1:-$LAT $SWEEP}"
STAMP=$(date +%H%M%S)
OUT="$HOME/nccl_sweep_$STAMP.log"
RES="$HOME/nccl_sweep_${STAMP}.results"
: > "$RES"
echo "sweep start $(date) specs=[$SPECS]" >> "$OUT"

collect_one() {
    local r=$1 h B BU D log t0 n line l ok
    h=$(printf "spark%x" "$r")
    B=$2; BU=$3; D=$4
    log="/tmp/burst_${B}_${BU}_${D}.log"
    t0=$(date +%s)
    line=""
    while [ $(( $(date +%s) - t0 )) -lt 80 ]; do
        l=$(timeout 6 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" \
            "grep -h per_op_us '$log' 2>/dev/null | tail -1" 2>/dev/null)
        if [ -n "$l" ]; then line="$l"; break; fi
        sleep 4
    done
    if [ -z "$line" ]; then
        l=$(timeout 6 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" \
            "tail -3 '$log' 2>/dev/null" 2>/dev/null)
        echo "rank=$r NO-RESULT tail: $l" >> "$RES"
    else
        echo "rank=$r $line" >> "$RES"
    fi
}

for spec in $SPECS; do
    B=${spec%%:*}; rest=${spec#*:}; I=${rest%%:*}; BU=${rest##*:}
    echo "--- size bytes=$B iters=$I burst=$BU $(date +%T) ---" >> "$OUT"
    bash "$HOME/burst_fanout6.sh" "$B" "$BU" 1 "$I" >> "$OUT" 2>&1
    for r in $(seq 0 15); do collect_one "$r" "$B" "$BU" 1 >> "$RES" 2>&1 & done
    wait
    ok=$(grep -c " OK" "$RES" 2>/dev/null || true)
    echo "collected: $(grep -c 'rank=' "$RES") lines, OK=$ok" >> "$OUT"
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        echo "SWEEP DEADLINE HIT after bytes=$B — stopping (rerun remainder)" >> "$OUT"
        break
    fi
done

{
    echo "=== RESULTS $(date) ==="
    sort -t= -k2 -n "$RES"
    echo "=== TABLE (per-collective = per_op_us/2; busbw = 2*(15/16)*S/t; toks = 1e6/(95*t)) ==="
    echo "bytes  batch  burst  per_coll_us  busbw_GBps  tok/s_floor"
    for spec in $SPECS; do
        B=${spec%%:*}; rest=${spec#*:}; BU=${rest##*:}
        v=$(grep "^rank=0 " "$RES" | grep "bytes=$B burst=$BU " | tail -1 | sed 's/.*per_op_us=\([0-9.]*\).*/\1/')
        [ -z "$v" ] && continue
        pc=$(awk -vp="$v" 'BEGIN{printf "%.1f", p/2}')
        bb=$(awk -vb="$B" -p="$pc" 'BEGIN{printf "%.2f", 2*(15.0/16)*b/p/1000}')
        tk=$(awk -vp="$pc" 'BEGIN{printf "%.0f", 1000000/(95*p)}')
        bat=$(( B / 8192 ))
        echo "$B  B$bat  $BU  $pc  $bb  $tk"
    done
} >> "$OUT" 2>&1
echo "sweep end $(date)" >> "$OUT"
