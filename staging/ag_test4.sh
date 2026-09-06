#!/bin/bash
set -u
P=58402
for h in spark0 spark1 spark2 spark3; do
    (timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" 'pkill -TERM mock_allgather 2>/dev/null; systemctl --user stop "mag-*" "mab*" 2>/dev/null; true') &
done
(ssh -o BatchMode=yes -o ConnectTimeout=8 -n spark9 'pkill -TERM ma_broker 2>/dev/null; true') &
wait
ssh -o BatchMode=yes -o ConnectTimeout=8 -n spark9 "pkill -TERM ma_broker 2>/dev/null; sleep 1; rm -f /tmp/ma_broker_$P.log; systemd-run --user --collect --unit=mab$P --working-directory=\$HOME \$HOME/ma_broker 4 $P 1024" >/dev/null 2>&1
sleep 2
for r in 0 1 2 3; do
    h=$(printf "spark%x" "$r")
    (timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "rm -f /tmp/ma_run_$P.log; systemd-run --user --collect --unit=mag$P-$r --working-directory=\$HOME bash \$HOME/ag_run.sh $r 4 0 $P" >/dev/null 2>&1) &
done
wait
sleep 30
for r in 0 1 2 3; do
    h=$(printf "spark%x" "$r")
    timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "head -4 /tmp/ma_run_$P.log 2>/dev/null" 2>/dev/null
    sleep 1
done
