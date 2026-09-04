#!/bin/bash
# 16-rank NCCL burst fan-out v3, runs ON spark5. Root (rank0, spark0) first,
# id relay, then WAIT FOR ROOT'S BOOTSTRAP LISTENER before launching peers
# (root's CUDA+NCCL init can take 60-90s on spark0; peers launched early
# exhaust their 35-retry connect budget and the whole mesh dies).
# args: bytes burst dual iters
set -u
B=$1 BU=$2 D=$3 I=$4
NODES="spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
ALL="spark0 $NODES"

# clean strays, fresh id
for h in $ALL; do
    timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
        "for p in \$(pgrep -f nccl_burs[t]); do kill -TERM \$p; done; rm -f /tmp/burst_${B}_${BU}_${D}.log; true" &
done
wait

# root = rank 0 on spark0 (generates id in-process, stays alive)
timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 spark0 \
    "BURST_RANK=0 BURST_NRANKS=16 nohup bash \$HOME/burst_node.sh root $B $BU $D $I >/dev/null 2>&1 &"
for i in $(seq 1 15); do
    sleep 1
    timeout 8 scp -q spark0:/tmp/nccl_id.bin /tmp/nccl_id.bin 2>/dev/null && [ -s /tmp/nccl_id.bin ] && break
done
[ -s /tmp/nccl_id.bin ] || { echo NO-ID; exit 1; }

# wait for root's bootstrap LISTEN (10.10.100.10 = spark0 fabric) before peers
WAIT=0
while [ $WAIT -lt 50 ]; do
    L=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 spark0 \
        "ss -tln 2>/dev/null | grep -c '10.10.100.10:'" 2>/dev/null)
    [ "${L:-0}" -ge 1 ] && break
    WAIT=$((WAIT+1))
    sleep 3
done
if [ "${L:-0}" -lt 1 ]; then echo ROOT-NO-LISTENER after 150s; exit 2; fi
echo "root listening after $((WAIT*3))s"

for r in $(seq 1 15); do
    h=$(printf "spark%x" "$r")
    timeout 10 scp -q /tmp/nccl_id.bin "$h:/tmp/nccl_id.bin" 2>/dev/null &
done
wait
for r in $(seq 1 15); do
    h=$(printf "spark%x" "$r")
    timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
        "BURST_RANK=$r BURST_NRANKS=16 nohup bash \$HOME/burst_node.sh $B $BU $D $I >/dev/null 2>&1 &" &
done
wait

# gap check + relaunch (peers only; root handled by the listener wait)
sleep 8
for round in 1 2 3; do
    MISS=""
    for r in $(seq 1 15); do
        h=$(printf "spark%x" "$r")
        n=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "pgrep -c -f nccl_burs[t] || true" 2>/dev/null | head -1)
        [ "${n:-0}" = "0" ] && MISS="$MISS $h:$r"
    done
    [ -z "$MISS" ] && break
    for m in $MISS; do
        h=${m%%:*}; r=${m##*:}
        timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
            "BURST_RANK=$r BURST_NRANKS=16 nohup bash \$HOME/burst_node.sh $B $BU $D $I >/dev/null 2>&1 &"
    done
    sleep 5
done
echo "fanout done round=$round miss='${MISS:-none}'"
