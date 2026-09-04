#!/bin/bash
# doorbell bench driver v2 (runs on spark5 as queue task).
# args: "rows:mode:iters:algo ..." algo: rd | d2a
set -u
STAMP=$(date +%H%M%S)
LOG="$HOME/mb_db_sweep_$STAMP.log"
DEADLINE=$(( $(date +%s) + 780 ))
SPECS="${1:-1:0:2000:rd}"
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
log "db_sweep_start specs=[$SPECS]"
B=$(md5sum "$HOME/mb_doorbell" | cut -d' ' -f1)
MISMATCH=""
for h in $ALL; do
    r=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" "md5sum \$HOME/mb_doorbell 2>/dev/null | cut -d' ' -f1" 2>/dev/null)
    [ "$r" = "$B" ] || MISMATCH="$MISMATCH $h"
done
if [ -n "$MISMATCH" ]; then
    log "mb_preflight FAILED (mixed binaries:$MISMATCH) - refusing to launch"
    exit 1
fi
log "mb_preflight ok ($B)"

teardown() {
    local h
    for h in $ALL; do
        timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" \
            "systemctl --user stop 'mbdb-*' 2>/dev/null; pkill -TERM -f mb_doorbell 2>/dev/null; sleep 1; pkill -TERM -f mb_doorbell 2>/dev/null; true" >/dev/null 2>&1
    done
}

run_one() {
    local ROWS=$1 MODE=$2 ITERS=$3 ALGO=$4 h r n ok line results
    local AVAL=rd
    [ "$ALGO" = "d2a" ] && AVAL=d2a
    [ "$ALGO" = "ring" ] && AVAL=ring
    log "db_run rows=$ROWS mode=$MODE iters=$ITERS algo=$ALGO"
    teardown
    for r in $(seq 0 15); do
        h=$(printf "spark%x" "$r")
        timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" \
            "rm -f /tmp/mb_db_r${r}.log; systemd-run --user --collect --unit=mbdb-$STAMP-$r --working-directory=\$HOME --setenv=BENCH_ALGO=$AVAL --setenv=MB_PROFILE=1 --setenv=BENCH_CREDITS=${BENCH_CREDITS:-64} --setenv=MB_LANES=${MB_LANES:-1} bash \$HOME/mb_db.sh $r $ITERS $ROWS $MODE" \
            >>"$LOG" 2>&1 &
        while [ $(jobs -r | wc -l) -ge 4 ]; do wait -n; done
    done
    wait
    results=""
    ok=0
    for n in $(seq 1 20); do
        ok=0
        for r in $(seq 0 15); do
            h=$(printf "spark%x" "$r")
            line=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" \
                "grep -h 'per_op_us' /tmp/mb_db_r${r}.log 2>/dev/null | tail -1" 2>/dev/null)
            if [ -n "$line" ]; then
                ok=$((ok+1))
                results="$results
rank=$r $line"
            fi
        done
        [ $ok -ge 16 ] && break
        sleep 6
    done
    echo "$results" >> "$LOG"
    if [ $ok -ge 16 ]; then
        log "db_result rows=$ROWS mode=$MODE algo=$ALGO ranks=16 median=$(echo "$results" | grep -o 'per_op_us=[0-9.]*' | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
    else
        log "db_failed rows=$ROWS mode=$MODE algo=$ALGO ranks=$ok/16"
    fi
}

for spec in $SPECS; do
    ROWS=$(echo "$spec" | cut -d: -f1); MODE=$(echo "$spec" | cut -d: -f2); ITERS=$(echo "$spec" | cut -d: -f3); ALGO=$(echo "$spec" | cut -d: -f4)
    run_one "$ROWS" "$MODE" "$ITERS" "$ALGO"
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "db_deadline"; break; }
done
teardown
log "db_sweep_end"
grep -h "rank=" "$LOG" | grep per_op_us | sort -t= -k4 -n
