#!/usr/bin/env bash
# registrar_fleet_probe.sh — per-host state line for the glm5_next fleet.
# One line per host: running procs (pid + rank + cwd), ready lines,
# open-timeout lines in the current residentd.log. Used for the registrar
# lane's baseline and post-wave receipts.
set -u
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
for h in "${HOSTS[@]}"; do
    line="$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'bash -s' <<'EOF'
rr="$HOME/sparkdata/glm5_next.tp16"
log="$rr/residentd.log"
pids=$(pgrep -f "sparkpipe_model_residentd --deployment" 2>/dev/null || true)
procs=0
detail=""
for p in $pids; do
    rank=$(ps -o args= -p "$p" | sed -nE "s/.*rank-index ([0-9]+).*/\1/p")
    cwd=$(readlink "/proc/$p/cwd" 2>/dev/null)
    procs=$((procs+1))
    detail="$detail pid=$p rank=${rank:-?} cwd=${cwd:-?}"
done
if [ -f "$log" ]; then
    ot=$(grep -c "rdma_open_timeout" "$log" 2>/dev/null || echo 0)
    rd=$(grep -c "model_residentd ready" "$log" 2>/dev/null || echo 0)
    last=$(tail -1 "$log" | cut -c1-90)
    echo "procs=$procs$detail ready_lines=$rd open_timeouts=$ot last=$last"
else
    echo "procs=$procs$detail log=missing"
fi
EOF
)" 2>&1
    printf '%-8s %s\n' "$h" "$line"
done
