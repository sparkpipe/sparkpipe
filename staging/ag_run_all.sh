#!/bin/bash
set -u
for h in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do
    (timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" 'systemctl --user stop "mag-*" "mab*" 2>/dev/null; pkill -TERM mock_allgather 2>/dev/null; pkill -TERM mock_allreduce 2>/dev/null; true') &
done
wait
run_group() {
    local D=$1 S=$2 G=$3
    local P=$(( 58399 + G ))
    local r h
    timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 spark9 \
        "rm -f /tmp/ma_broker_$P.log; systemd-run --user --collect --unit=mab$P --working-directory=\$HOME bash \$HOME/ma_broker_run.sh $D $P" >/dev/null 2>&1
    sleep 2
    for r in $(seq 0 $((D-1))); do
        h=$(printf "spark%x" $(( S + r )))
        (timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
            "rm -f /tmp/ma_run_$P.log; systemd-run --user --collect --unit=mag$P-$r --working-directory=\$HOME bash \$HOME/ag_run.sh $r $D $S $P") >/dev/null 2>&1 &
    done
    wait
}
run_group 16 0 0
sleep 40
run_group 8 0 1
run_group 8 8 2
sleep 30
run_group 4 0 3
run_group 4 4 4
run_group 4 8 5
run_group 4 12 6
sleep 40
for g in 0 1 2 3 4 5 6; do
    case $g in
        0) S=0; D=16 ;;
        1) S=0; D=8 ;;
        2) S=8; D=8 ;;
        *) S=$(( (g-3)*4 )); D=4 ;;
    esac
    P=$(( 58399 + g ))
    echo "== group$g tp$D"
    for (( r=0; r<D; r++ )); do
        h=$(printf "spark%x" $(( S + r )))
        timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" \
            "grep -E 'ALLGATHER|VERIFY|probe failed|arrival timeout' /tmp/ma_run_$P.log 2>/dev/null | head -2" 2>/dev/null
        sleep 1
    done
done
