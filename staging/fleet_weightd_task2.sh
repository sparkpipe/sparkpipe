#!/bin/bash
# weightd fast-load: robust restart + 5-cycle verification, round 2.
# The round-1 restart raced live old daemons (TERM only lands between poll
# steps; a mid-load old daemon survives 10s). Here: 60s TERM grace, spawn
# only after confirmed death, prove the new binary by its ck128 cold-load
# line, then 5 cycles per node with per-node verdicts.
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
NODES="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
LOG=/tmp/fleet_weightd2.log
: > "$LOG"

rank_of() {
    case "$1" in
        spark0) echo 0;;  spark1) echo 1;;  spark2) echo 2;;  spark3) echo 3;;
        spark4) echo 4;;  spark5) echo 5;;  spark6) echo 6;;  spark7) echo 7;;
        spark8) echo 8;;  spark9) echo 9;;  sparka) echo 10;; sparkb) echo 11;;
        sparkc) echo 12;; sparkd) echo 13;; sparke) echo 14;; sparkf) echo 15;;
    esac
}

echo "=== RESTART2 $(date -u +%H:%M:%S) ===" >> "$LOG"
for h in $NODES; do
    (
        out=$(timeout 180 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "
            rr=\$HOME/sparkdata/glm5_next.tp16
            for p in \$(pgrep -f 'bin/sparkpipe_weight[d]'); do
                exe=\$(readlink /proc/\$p/exe 2>/dev/null)
                case \"\$exe\" in
                    \$rr/bin/sparkpipe_weightd*) kill -TERM \$p;;
                esac
            done
            dead=0
            for i in \$(seq 1 30); do
                pgrep -f 'bin/sparkpipe_weight[d]' >/dev/null || { dead=1; break; }
                sleep 2
            done
            [ \"\$dead\" = 1 ] || { echo TERM-STUCK; exit 1; }
            cd \$rr && nohup ./bin/sparkpipe_weightd --socket /tmp/spark_weightd.sock >/tmp/weightd.out 2>&1 &
            for i in \$(seq 1 10); do
                grep -q 'spark_weightd ready' /tmp/weightd.out 2>/dev/null && break
                sleep 1
            done
            grep -q 'spark_weightd ready' /tmp/weightd.out && echo ready || echo UNREADY
        " 2>&1)
        echo "$h restart2: $out" >> "$LOG"
    ) &
done
wait
ready=$(grep -c 'restart2: ready' "$LOG")
echo "ready=$ready/16" >> "$LOG"
[ "$ready" -eq 16 ] || { echo RESTART-FAILED >> "$LOG"; grep restart2 "$LOG"; exit 1; }

echo "=== PROOF $(date -u +%H:%M:%S) ===" >> "$LOG"
for h in $NODES; do
    r=$(rank_of "$h")
    (
        out=$(timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "
            export SPARK_WEIGHTD_SOCKET=/tmp/spark_weightd.sock
            p=\$HOME/sparkdata/glm5_next.tp16/packs/glm5_next_stage.tp16.rank${r}.g5nsp
            export SPARK_WEIGHTD_PACK_SHA256=\$(awk '{print \$1}' \"\$p.sha256\")
            cd \$HOME/sparkdata/glm5_next.tp16
            ~/weightdctl reclaim >/dev/null 2>&1
            out=\$(SPARK_WEIGHTDCTL_NO_IMPORT=1 ~/weightdctl load \"\$p\" glm5_next_stage '')
            grep -q 'cold=1' <<<\"\$out\" || { echo \"NO-ATTACH: \$out\"; exit 1; }
            grep 'cold-load' /tmp/weightd.out | tail -1
        " 2>&1)
        echo "$h proof: $out" >> "$LOG"
    ) &
done
wait

echo "=== CYCLES2 $(date -u +%H:%M:%S) ===" >> "$LOG"
WALL_START=$(date +%s%N)
for h in $NODES; do
    r=$(rank_of "$h")
    (
        timeout 240 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "
            export SPARK_WEIGHTD_SOCKET=/tmp/spark_weightd.sock
            p=\$HOME/sparkdata/glm5_next.tp16/packs/glm5_next_stage.tp16.rank${r}.g5nsp
            export SPARK_WEIGHTD_PACK_SHA256=\$(awk '{print \$1}' \"\$p.sha256\")
            cd \$HOME/sparkdata/glm5_next.tp16
            for c in 1 2 3 4 5; do
                ~/weightdctl reclaim >/dev/null 2>&1
                a=\$(date +%s%N)
                out=\$(SPARK_WEIGHTDCTL_NO_IMPORT=1 ~/weightdctl load \"\$p\" glm5_next_stage '')
                b=\$(date +%s%N)
                tag=\$(printf '%s' \"\$out\" | grep -o 'cold=[01]')
                [ \"\$tag\" = 'cold=1' ] || echo \"$h cycle\$c NOT-COLD: \$out\"
                out2=\$(SPARK_WEIGHTDCTL_NO_IMPORT=1 ~/weightdctl load \"\$p\" glm5_next_stage '')
                d=\$(date +%s%N)
                tag2=\$(printf '%s' \"\$out2\" | grep -o 'cold=[01]')
                [ \"\$tag2\" = 'cold=0' ] || echo \"$h cycle\$c NOT-WARM: \$out2\"
                echo \"$h cycle\$c cold_ms=\$(( (b - a) / 1000000 )) warm_ms=\$(( (d - b) / 1000000 ))\"
            done
        " >> "$LOG" 2>&1
    ) &
done
wait
WALL_END=$(date +%s%N)

echo "=== VERDICT $(date -u +%H:%M:%S) ===" >> "$LOG"
fail=0
for h in $NODES; do
    total_ms=$(grep -o "^$h cycle[1-5] cold_ms=[0-9]*" "$LOG" | grep -o '[0-9]*$' | paste -sd+ - | bc)
    anomalies=$(grep -c "^$h cycle[1-5] NOT-" "$LOG")
    proof=$(grep -c "^$h proof: weightd cold-load.*checksum=ck128" "$LOG")
    verdict=ok
    { [ -n "$total_ms" ] && [ "$total_ms" -le 60000 ]; } || { verdict=FAIL-SLOW; fail=1; }
    [ "$anomalies" -eq 0 ] || { verdict="$verdict+FAIL-CYCLE"; fail=1; }
    [ "$proof" -eq 1 ] || { verdict="$verdict+NO-PROOF"; fail=1; }
    echo "$h total_5cycle_ms=${total_ms:-none} ck128_proof=$proof $verdict"
done | tee -a "$LOG"
echo "OVERALL: $([ $fail -eq 0 ] && echo PASS || echo FAIL) cycles_wall_ms=$(( (WALL_END - WALL_START) / 1000000 ))" | tee -a "$LOG"
