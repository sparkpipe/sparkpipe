#!/bin/bash
set -u
run_group() {
    local D=$1 S=$2 G=$3
    local P=$(( 58399 + G ))
    local r h
    for (( r = D - 1; r >= 0; r-- )); do
        h=$(printf "spark%x" $(( S + r )))
        timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
            "pkill -TERM mock_allreduce 2>/dev/null; sleep 1; rm -f /tmp/ma_run_$P.log; setsid nohup taskset -c 2 \$HOME/mock_allreduce $r $D $S $P > /tmp/ma_run_$P.log 2>&1 & echo ok" \
            >/dev/null 2>&1
        sleep 1
    done
    timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 spark9 \
        "pkill -TERM ma_broker 2>/dev/null; sleep 1; setsid nohup taskset -c 2 \$HOME/ma_broker $D $P > /tmp/ma_broker_$P.log 2>&1 & echo ok" \
        >/dev/null 2>&1
}

for h in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do
    (timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" 'pkill -TERM mock_allreduce 2>/dev/null; pkill -TERM ma_broker 2>/dev/null; true') &
done
wait
run_group 16 0 0
sleep 45
run_group 8 0 1
run_group 8 8 2
sleep 35
run_group 4 0 3
run_group 4 4 4
run_group 4 8 5
run_group 4 12 6
sleep 45
for g in 0 1 2 3 4 5 6; do
    P=$(( 58399 + G ))
    case $g in
        0) S=0; D=16 ;;
        1) S=0; D=8 ;;
        2) S=8; D=8 ;;
        *) S=$(( (g - 3) * 4 )); D=4 ;;
    esac
    P=$(( 58399 + g ))
    echo "== group$g tp$D start$S"
    for (( r = 0; r < D; r++ )); do
        h=$(printf "spark%x" $(( S + r )))
        timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" \
            "grep -E 'ALLREDUCE|TOKS' /tmp/ma_run_$P.log 2>/dev/null" 2>/dev/null
        sleep 1
    done
done
