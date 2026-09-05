#!/bin/bash
set -u
for r in $(seq 0 15); do
    h=$(printf "spark%x" "$r")
    timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        "rm -f /tmp/fc_run.log; systemd-run --user --collect --unit=fcx-$r --working-directory=\$HOME bash \$HOME/fc_run.sh $r" \
        >/dev/null 2>&1 &
    while [ $(jobs -r | wc -l) -ge 4 ]; do wait -n; done
done
wait
sleep 45
for r in $(seq 0 15); do
    h=$(printf "spark%x" "$r")
    timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        "cat /tmp/fc_run.log 2>/dev/null" > /tmp/fc_host_$r.log 2>/dev/null
    sleep 1
done
cat /tmp/fc_host_*.log 2>/dev/null | grep -E "^LINK|^RANK|MISMATCH|wc status|failed" | sort
