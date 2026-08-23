#!/usr/bin/env bash
# Session watcher: poll spark6 production run + fleet reachability every 5 min.
# Exits when SHA256SUMS.tp16 lands, production exits, or ssh dies repeatedly.
LOG=/Users/mac/dsh.sparkpipe/tmp/k3_prod_watch.log
fails=0
while true; do
    ts=$(date -u +%FT%TZ)
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark6 '
        sz=$(stat -c %s /home/spark6/k3tp16prod/k3.full.tilek32.pack.payload 2>/dev/null || echo 0)
        jrnl=$(tail -1 /home/spark6/k3tp16prod/k3.full.tilek32.pack.payload.journal 2>/dev/null | head -c 120)
        shard=$(find /home/spark6/k3tp16prod -maxdepth 1 -type f -name "*.rank*.pack" 2>/dev/null | wc -l)
        sums=$([ -s /home/spark6/k3tp16prod/SHA256SUMS.tp16 ] && echo yes || echo no)
        alive=$(pgrep -cf "k3_pack.py|k3_shard.py|k3_tp16_pack_production" || true)
        echo "$sz | shards=$shard | sums=$sums | procs=$alive | $jrnl"
    ' 2>/dev/null)
    if [ -z "$out" ]; then
        fails=$((fails+1))
        echo "$ts SSH-FAIL $fails" >> "$LOG"
        [ "$fails" -ge 4 ] && { echo "$ts GIVEUP ssh" >> "$LOG"; exit 1; }
    else
        fails=0
        up=$(tools/devcycle/fleet_status.sh 2>/dev/null | grep -c -v unreachable)
        echo "$ts $out fleet_up=$up" >> "$LOG"
        case "$out" in
            *"sums=yes"*|*"procs=0"*) echo "$ts DONE" >> "$LOG"; exit 0;;
        esac
    fi
    sleep 300
done
