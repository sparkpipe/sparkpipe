#!/usr/bin/env bash
# term_by_cwd_sweep.sh — operator TERM sweep, exact pid only: discover
# residentd processes by exact exe basename + exact deployment cwd (the
# registrar's discovery, never a pattern kill), TERM them, wait for exit.
# Any pid that survives TERM past --term-wait is reported STALE-IMMUNE for
# the operator; this script NEVER sends SIGKILL.
set -u
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
CWD_SUFFIX="sparkdata/glm5_next.tp16"

sweep_host() {
    local h="$1"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" 'bash -s' <<'EOF'
suffix="'"$CWD_SUFFIX"'"
term_one() {
    local pid="$1"
    local exe cwd
    exe=$(readlink "/proc/$pid/exe" 2>/dev/null) || return 0
    [ "$(basename "$exe")" = "sparkpipe_model_residentd" ] || return 0
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null) || return 0
    case "$cwd" in *"$suffix") ;; *) return 0 ;; esac
    local state
    state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)
    [ "$state" = "Z" ] && return 0
    echo "TERM $pid"
    kill -TERM "$pid" 2>/dev/null
}
for round in 1 2 3 4 5 6 7 8; do
    found=0
    immune=""
    for d in /proc/[0-9]*; do
        pid=$(basename "$d")
        line=$(term_one "$pid")
        case "$line" in TERM*) found=$((found+1));; esac
    done
    [ "$found" -eq 0 ] && { echo "clear"; exit 0; }
    sleep 5
    # recount survivors
    alive=0
    for d in /proc/[0-9]*; do
        pid=$(basename "$d")
        exe=$(readlink "/proc/$pid/exe" 2>/dev/null) || continue
        [ "$(basename "$exe")" = "sparkpipe_model_residentd" ] || continue
        cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null) || continue
        case "$cwd" in *"$suffix") ;; *) continue ;; esac
        state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)
        [ "$state" = "Z" ] && continue
        alive=$((alive+1))
        immune="$immune $pid"
    done
    [ "$alive" -eq 0 ] && { echo "clear"; exit 0; }
    echo "still_alive:$alive pids:$immune"
    exit 3
done
echo "sweep_exhausted"
exit 4
EOF
}

for h in "${HOSTS[@]}"; do
    echo "$h: $(sweep_host "$h" | tr '\n' ' ')"
done
