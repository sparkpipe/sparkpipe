#!/bin/bash
# weightd fast-load fleet rollout + 5-cycle verification.
# Runs on nodes[0] (spark5). Per node: stage binaries, stamp .ck128 sidecar,
# restart daemon (fresh socket), then 5 cycles of [reclaim -> cold load ->
# warm verify]. Gate: 5 cycles <= 60s total per node, warm attach cold=0.
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
NODES="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
BUILD=/home/spark5/weightd-fastbuild
LOG=/tmp/fleet_weightd.log
: > "$LOG"

rank_of() {
    case "$1" in
        spark0) echo 0;;  spark1) echo 1;;  spark2) echo 2;;  spark3) echo 3;;
        spark4) echo 4;;  spark5) echo 5;;  spark6) echo 6;;  spark7) echo 7;;
        spark8) echo 8;;  spark9) echo 9;;  sparka) echo 10;; sparkb) echo 11;;
        sparkc) echo 12;; sparkd) echo 13;; sparke) echo 14;; sparkf) echo 15;;
    esac
}

phase_banner() { echo "=== $1 $(date -u +%H:%M:%S) ===" >> "$LOG"; }

phase_banner STAGE
for h in $NODES; do
    (
        timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "mkdir -p ~/weightd-fastbuild/bin" &&
        timeout 180 scp -q -o BatchMode=yes -o ConnectTimeout=8 \
            spark5:$BUILD/sparkpipe_weightd spark5:$BUILD/weightdctl spark5:$BUILD/ck128_stamp \
            "$h":weightd-fastbuild/bin/ &&
        echo "$h staged" >> "$LOG"
    ) &
done
wait
staged=$(grep -c ' staged$' "$LOG")
echo "staged=$staged/16" >> "$LOG"
[ "$staged" -eq 16 ] || { echo "STAGE-FAILED" >> "$LOG"; grep -v staged "$LOG" | tail -20; exit 1; }

phase_banner STAMP
for h in $NODES; do
    r=$(rank_of "$h")
    (
        timeout 180 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "
            p=\$HOME/sparkdata/glm5_next.tp16/packs/glm5_next_stage.tp16.rank${r}.g5nsp
            if [ -s \"\$p.ck128\" ]; then echo 'ck128 existing'; else ~/weightd-fastbuild/bin/ck128_stamp \"\$p\" 2>&1 | tail -1; fi
        " >> "$LOG" 2>&1
        echo "$h stamp-done" >> "$LOG"
    ) &
done
wait
echo "stamped=$(grep -c 'stamp-done' "$LOG")/16" >> "$LOG"

phase_banner RESTART
for h in $NODES; do
    (
        out=$(timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "
            rr=\$HOME/sparkdata/glm5_next.tp16
            for p in \$(pgrep -f 'bin/sparkpipe_weight[d]'); do
                exe=\$(readlink /proc/\$p/exe 2>/dev/null)
                case \"\$exe\" in
                    \$rr/bin/sparkpipe_weightd*) kill -TERM \$p;;
                esac
            done
            for i in 1 2 3 4 5 6 7 8 9 10; do
                pgrep -f 'bin/sparkpipe_weight[d]' >/dev/null || break
                sleep 1
            done
            cp -f \$rr/bin/sparkpipe_weightd \$rr/bin/sparkpipe_weightd.prev 2>/dev/null || true
            cp -f ~/weightd-fastbuild/bin/sparkpipe_weightd \$rr/bin/sparkpipe_weightd
            cp -f ~/weightd-fastbuild/bin/weightdctl \$HOME/weightdctl
            cd \$rr && nohup ./bin/sparkpipe_weightd --socket /tmp/spark_weightd.sock >/tmp/weightd.out 2>&1 &
            for i in 1 2 3 4 5 6 7 8 9 10; do
                grep -q 'spark_weightd ready' /tmp/weightd.out 2>/dev/null && break
                sleep 1
            done
            grep -q 'spark_weightd ready' /tmp/weightd.out && echo ready || echo UNREADY
        " 2>&1)
        echo "$h restart: $out" >> "$LOG"
    ) &
done
wait
ready=$(grep -c 'restart: ready' "$LOG")
echo "ready=$ready/16" >> "$LOG"
[ "$ready" -eq 16 ] || { echo "RESTART-FAILED" >> "$LOG"; grep restart "$LOG"; exit 1; }

phase_banner ROOFLINE
for h in $NODES; do
    r=$(rank_of "$h")
    (
        timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "
            dd if=\$HOME/sparkdata/glm5_next.tp16/packs/glm5_next_stage.tp16.rank${r}.g5nsp \
               of=/dev/null bs=64M count=128 2>&1 | tail -1
        " >> "$LOG" 2>&1
        echo "$h dd-done" >> "$LOG"
    ) &
done
wait

phase_banner CYCLES
CYCLE_START=$(date +%s%N)
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
                [ \"\$tag\" = 'cold=1' ] || echo \"\$h cycle\$c NOT-COLD: \$out\"
                out2=\$(SPARK_WEIGHTDCTL_NO_IMPORT=1 ~/weightdctl load \"\$p\" glm5_next_stage '')
                d=\$(date +%s%N)
                tag2=\$(printf '%s' \"\$out2\" | grep -o 'cold=[01]')
                [ \"\$tag2\" = 'cold=0' ] || echo \"\$h cycle\$c NOT-WARM: \$out2\"
                echo \"\$h cycle\$c cold_ms=\$(( (b - a) / 1000000 )) warm_ms=\$(( (d - b) / 1000000 ))\"
            done
        " >> "$LOG" 2>&1
    ) &
done
wait
CYCLE_END=$(date +%s%N)
echo "cycles_wall_ms=$(( (CYCLE_END - CYCLE_START) / 1000000 ))" >> "$LOG"

phase_banner VERDICT
fail=0
for h in $NODES; do
    total_ms=$(grep -o "$h cycle[1-5] cold_ms=[0-9]*" "$LOG" | grep -o '[0-9]*$' | paste -sd+ - | bc)
    warm_bad=$(grep -c "$h cycle[1-5] NOT-" "$LOG" || true)
    verdict=ok
    [ -n "$total_ms" ] && [ "$total_ms" -le 60000 ] || { verdict=FAIL-SLOW; fail=1; }
    [ "$warm_bad" -eq 0 ] || { verdict=FAIL-CYCLE; fail=1; }
    echo "$h total_5cycle_ms=${total_ms:-none} $verdict" | tee -a "$LOG"
done
echo "OVERALL: $([ $fail -eq 0 ] && echo PASS || echo FAIL) wall_ms=$(( (CYCLE_END - CYCLE_START) / 1000000 ))" | tee -a "$LOG"
grep -E 'cold-load' /tmp/weightd.out 2>/dev/null | tail -5
