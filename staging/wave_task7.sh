#!/bin/bash
# Wave verification of the mesh-robust transport: deploy the freshly built
# residentd (tp-nccl idle-window + visibility), launch all 16 with warm
# weightds, capture the new tp-nccl log lines, probe once, report.
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
NODES="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
LOG=/tmp/wave_task7.log
: > "$LOG"

for h in $NODES; do
    (
        out=$(timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" '
            rr=$HOME/sparkdata/glm5_next.tp16
            for pat in "bin/sparkpipe_model_[r]esidentd" "bin/sparkpipe_model_[a]pi"; do
                for p in $(pgrep -f "$pat"); do
                    c=$(readlink /proc/$p/cwd 2>/dev/null)
                    [ "$c" = "$rr" ] && kill -TERM $p
                done
            done
            for i in $(seq 1 20); do
                pgrep -f "bin/sparkpipe_model_[r]esidentd" >/dev/null || break
                sleep 1
            done
            cp -f $rr/bin/sparkpipe_model_residentd $rr/bin/sparkpipe_model_residentd.prev 2>/dev/null || true
            :
        ' 2>&1)
        echo "$h cleaned" >> "$LOG"
    ) &
done
wait

for h in $NODES; do
    (
        ok=0
        for attempt in 1 2; do
            timeout 180 scp -q -o BatchMode=yes -o ConnectTimeout=8 \
                spark5:glm53-gates/build/sparkpipe_model_residentd \
                "$h":sparkdata/glm5_next.tp16/bin/sparkpipe_model_residentd.new && { ok=1; break; }
            sleep 5
        done
        if [ "$ok" = 1 ]; then
            timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
                "mv \$HOME/sparkdata/glm5_next.tp16/bin/sparkpipe_model_residentd.new \$HOME/sparkdata/glm5_next.tp16/bin/sparkpipe_model_residentd" \
                && echo "$h deployed" >> "$LOG"
        fi
    ) &
done
wait
deployed=$(grep -c ' deployed' "$LOG")
echo "deployed=$deployed/16" >> "$LOG"
[ "$deployed" -eq 16 ] || { echo DEPLOY-FAILED >> "$LOG"; exit 1; }

for h in $NODES; do
    (
        out=$(timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" '
            rr=$HOME/sparkdata/glm5_next.tp16
            cd $rr
            mv residentd.log residentd.log.prev-$(date +%s) 2>/dev/null || true
            env NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1 NCCL_IB_GID_INDEX=3 \
                LD_LIBRARY_PATH=$rr/lib nohup ./bin/sparkpipe_model_residentd \
                --deployment model_resident.json --rank-index '"$(case $h in spark0) echo 0;; spark1) echo 1;; spark2) echo 2;; spark3) echo 3;; spark4) echo 4;; spark5) echo 5;; spark6) echo 6;; spark7) echo 7;; spark8) echo 8;; spark9) echo 9;; sparka) echo 10;; sparkb) echo 11;; sparkc) echo 12;; sparkd) echo 13;; sparke) echo 14;; sparkf) echo 15;; esac)"' \
                > residentd.log 2>&1 < /dev/null &
            echo launched
        ' 2>&1)
        echo "$h launch: $out" >> "$LOG"
    ) &
done
wait

deadline=$(( $(date +%s) + 300 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    ready=0
    for h in $NODES; do
        timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" \
            "grep -q 'model_residentd ready' \$HOME/sparkdata/glm5_next.tp16/residentd.log 2>/dev/null" \
            && ready=$((ready+1))
    done
    echo "ready: $ready/16 $(date -u +%H:%M:%S)" >> "$LOG"
    [ "$ready" -eq 16 ] && break
    sleep 10
done

echo "=== TP-NCCL LINES (spark0 + spark8) ===" >> "$LOG"
for h in spark0 spark8; do
    echo "-- $h:" >> "$LOG"
    timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        "grep 'tp-nccl' \$HOME/sparkdata/glm5_next.tp16/residentd.log 2>/dev/null | head -8" >> "$LOG" 2>&1
done
grep -E 'ready: |tp-nccl' "$LOG" | tail -30
