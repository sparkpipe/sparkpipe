#!/bin/bash
set -u
P=58399
systemctl --user stop mab$P 2>/dev/null
rm -f /tmp/ma_broker_$P.log
systemd-run --user --collect --unit=mab$P --working-directory=$HOME \
    bash $HOME/ma_broker_run.sh 16 $P >/dev/null 2>&1
sleep 2
for r in $(seq 0 15); do
    h=$(printf "spark%x" "$r")
    (timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
        "systemctl --user stop mag$P-$r 2>/dev/null; rm -f /tmp/ma_run_$P.log; systemd-run --user --collect --unit=mag$P-$r --working-directory=\$HOME bash \$HOME/ma_run.sh $r 16 0 $P" >/dev/null 2>&1) &
done
wait
sleep 30
for r in $(seq 0 15); do
    h=$(printf "spark%x" "$r")
    echo -n "r$r: "
    timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
        "grep -oE 'expect=[0-9a-f]+' /tmp/ma_run_$P.log 2>/dev/null | tail -1; grep -c ALLREDUCE /tmp/ma_run_$P.log 2>/dev/null | tr -d '\n'; echo \" ok\"" 2>/dev/null
    sleep 1
done
