#!/bin/bash
set -u
for r in $(seq 0 15); do
    h=$(printf "spark%x" "$r")
    timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        "rm -f /tmp/ma_run.log; systemd-run --user --collect --unit=max-$r --working-directory=\$HOME bash \$HOME/ma_run.sh $r" \
        >/dev/null 2>&1 &
    while [ $(jobs -r | wc -l) -ge 4 ]; do wait -n; done
done
wait
sleep 40
for r in $(seq 0 15); do
    h=$(printf "spark%x" "$r")
    timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        "cat /tmp/ma_run.log 2>/dev/null" > /tmp/ma_host_$r.log 2>/dev/null
    sleep 1
done
cat /tmp/ma_host_*.log 2>/dev/null | grep -E "ALLREDUCE|TOKS|VERIFY|timeout|mismatch|failed" | sort
