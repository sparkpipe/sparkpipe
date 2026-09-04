#!/bin/bash
# meshbench sweep v7 — hill-climb build. Adds: MB_ROOT (root node, default
# spark5 per the 09-02 receipt), rotated rank map r=((hex-host-hex-root+16)%16),
# MB_LABEL (run tag), MB_ENV (extra NCCL exports shipped to every rank via
# /tmp/mb_extra.env). Single-comm only (dual-comm is fatal on NCCL 2.28).
# args: "bytes:iters:burst ..."
set -u
ROOTN="${MB_ROOT:-spark5}"
LABEL="${MB_LABEL:-base}"
EXTRA="${MB_ENV:-}"
STAMP=$(date +%H%M%S)
LOG="$HOME/mb_${LABEL}_${STAMP}.log"
DEADLINE=$(( $(date +%s) + 780 ))
SIZES="${1:-8192:2000:1}"
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"

hx() { printf "%d" "0x${1#spark}"; }
rank_of() { echo $(( ( $(hx "$1") - $(hx "$ROOTN") + 16 ) % 16 )); }
node_of_rank() { local r=$1 n rr; for n in $ALL; do rr=$(rank_of "$n"); [ "$rr" -eq "$r" ] && { echo "$n"; return; }; done; }

log() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
log "mb_sweep_start v7 root=$ROOTN label=$LABEL extra=[$EXTRA] specs=[$SIZES]"

ship_env() {
    local h
    for h in $ALL; do
        timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" \
            "if [ -n '$EXTRA' ]; then printf 'export %s\n' '$EXTRA' > /tmp/mb_extra.env; else rm -f /tmp/mb_extra.env; fi" \
            >/dev/null 2>&1 &
        while [ $(jobs -r | wc -l) -ge 4 ]; do wait -n; done
    done
    wait
}

teardown() {
    local h p n count termd
    termd=0
    for h in $ALL; do
        if timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" \
            "systemctl --user stop 'mbp-*' 'mbroot-*' 2>/dev/null; if pgrep -f nccl_burs[t] >/dev/null; then for p in \$(pgrep -f nccl_burs[t]); do kill -TERM \$p 2>/dev/null; done; echo T; fi" \
            2>/dev/null | grep -q T; then
            termd=1
        fi
    done
    if [ $termd -eq 1 ]; then
        count=1
        for n in $(seq 1 15); do
            count=0
            for h in $ALL; do
                c=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" \
                    "pgrep -c -f nccl_burs[t] || true" 2>/dev/null)
                count=$(( count + ${c:-0} ))
            done
            [ $count -eq 0 ] && break
            sleep 3
        done
        if [ $count -ne 0 ]; then
            log "mb_teardown force_kill residue=$count"
            for h in $ALL; do
                timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" \
                    "for p in \$(pgrep -f nccl_burs[t]); do kill -9 \$p 2>/dev/null; done; true" \
                    >/dev/null 2>&1
            done
            sleep 60
        else
            log "mb_teardown clean term-exit; settle 25s"
            sleep 25
        fi
    fi
}

run_size() {
    local B=$1 I=$2 BU=$3 TAG="$LABEL-$1-$2-$3" attempt port b10 b11 h r n line ok results
    for attempt in 1 2; do
        log "mb_run size=$B iters=$I burst=$BU attempt=$attempt tag=$TAG"
        ship_env
        teardown
        timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROOTN" \
            "systemd-run --user --collect --unit=mbroot-$STAMP-a$attempt --working-directory=\$HOME bash \$HOME/mb_root.sh 16 $I $B $BU 0" \
            >>"$LOG" 2>&1
        for n in $(seq 1 10); do
            timeout 8 scp -q "$ROOTN":/tmp/nccl_id.bin "$HOME/mb_id_$STAMP.bin" 2>/dev/null
            [ -s "$HOME/mb_id_$STAMP.bin" ] && break
            sleep 1
        done
        if [ ! -s "$HOME/mb_id_$STAMP.bin" ]; then log "mb_run_failed size=$B reason=no_id attempt=$attempt"; continue; fi
        b10=$(od -An -tu1 -j10 -N1 "$HOME/mb_id_$STAMP.bin" | tr -d ' ')
        b11=$(od -An -tu1 -j11 -N1 "$HOME/mb_id_$STAMP.bin" | tr -d ' ')
        port=$(( b10 * 256 + b11 ))
        ok=1
        for n in $(seq 1 20); do
            if timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "$ROOTN" \
                "ss -tln | grep -q ':${port} '" 2>/dev/null; then ok=0; break; fi
            sleep 3
        done
        if [ $ok -ne 0 ]; then log "mb_run_failed size=$B reason=root_no_listener port=$port attempt=$attempt"; continue; fi
        log "mb_root_ready port=$port"
        for r in $(seq 1 15); do
            h=$(node_of_rank "$r")
            timeout 10 scp -q "$HOME/mb_id_$STAMP.bin" "$h:/tmp/mb_id_$STAMP.bin" 2>/dev/null
        done
        for r in $(seq 1 15); do
            h=$(node_of_rank "$r")
            timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" \
                "systemd-run --user --collect --unit=mbp-$STAMP-a$attempt-$r --working-directory=\$HOME bash \$HOME/mb_peer.sh $r 16 $I $B $BU 0 /tmp/mb_id_$STAMP.bin $STAMP-$LABEL" \
                >>"$LOG" 2>&1
        done
        results=""
        ok=0
        for n in $(seq 1 15); do
            ok=0
            line=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "$ROOTN" "grep per_op_us /tmp/mb_root.log 2>/dev/null | grep \"bytes=$B burst=$BU \" | tail -1" 2>/dev/null)
            [ -n "$line" ] && ok=$((ok+1)) && results="$results
rank=0 $line"
            for r in $(seq 1 15); do
                h=$(node_of_rank "$r")
                line=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "$h" "grep -h per_op_us /tmp/mb_${STAMP}-${LABEL}_r${r}.log 2>/dev/null | tail -1" 2>/dev/null)
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
            log "mb_result label=$LABEL size=$B burst=$BU ranks=16 attempt=$attempt"
            return 0
        fi
        log "mb_run_failed size=$B reason=incomplete ranks=$ok/16 attempt=$attempt"
    done
    return 1
}

for spec in $SIZES; do
    B=${spec%%:*}; rest=${spec#*:}; I=${rest%%:*}; BU=${rest##*:}
    run_size "$B" "$I" "$BU"
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then log "mb_deadline_hit after size=$B"; break; fi
done
teardown
log "mb_sweep_end"
grep -h "rank=" "$LOG" | grep per_op_us | sort -t= -k4 -n
rm -f "$HOME/mb_id_$STAMP.bin"
