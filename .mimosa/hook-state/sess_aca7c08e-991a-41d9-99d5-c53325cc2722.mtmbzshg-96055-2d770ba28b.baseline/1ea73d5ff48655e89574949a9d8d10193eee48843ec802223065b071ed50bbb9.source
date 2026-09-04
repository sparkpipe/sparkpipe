#!/usr/bin/env bash
# glm5_next fixed-pack build monitor (controller side).
# Polls the 15 rank rebuilds (rank r builds on sparke-hex-digit r; rank 0's
# pack is already complete on spark0). Tracks:
#   - pack size growth (liveness),
#   - builder process state (D = uninterruptible ceph sleep; a D streak
#     >= D_LIMIT consecutive polls (~30 min at the default interval) is
#     flagged as WEDGED-D for the lane to act on,
#   - the completion receipt line ("N tensors, M bytes") in the build log.
# Exit 0 when every tracked rank has its receipt. Never signals the nodes.
#
# usage: glm5_next_pack_monitor.sh [interval_seconds] [D_limit]
set -u
INTERVAL="${1:-300}"
D_LIMIT="${2:-6}"
RANKS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
declare -A D_STREAK
declare -A DONE

host_for_rank() { printf "spark%x" "$1"; }
pack_path() { echo "\$HOME/glm53_packs_fixed/glm5_next_stage.tp16.rank$1.g5nsp"; }

while true; do
    stamp=$(date -u +%FT%TZ)
    remaining=0
    line="$stamp"
    for r in $RANKS; do
        h=$(host_for_rank "$r")
        out=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "
            p=\$(ls $(pack_path "$r") 2>/dev/null) && sz=\$(stat -c%s \"\$p\") || sz=0;
            pid=\$(pgrep -f '^python3 .*resident_stagepack.py .*--tp-rank $r ' | head -1);
            if [ -n \"\$pid\" ]; then st=\$(ps -o stat= -p \$pid | cut -c1-1); et=\$(ps -o etimes= -p \$pid); else st=X; et=-; fi;
            rec=\$(grep -o 'rank$r.g5nsp: [0-9]* tensors, [0-9]* bytes' /tmp/packbuild-r$r.log 2>/dev/null | head -1);
            echo \"\$sz|\$st|\$et|\$rec\";" 2>/dev/null | tail -1)
        sz=$(echo "$out" | cut -d'|' -f1)
        st=$(echo "$out" | cut -d'|' -f2)
        et=$(echo "$out" | cut -d'|' -f3)
        rec=$(echo "$out" | cut -d'|' -f4)
        if [ -n "$rec" ]; then
            DONE[$r]=1
            line="$line | r$r:DONE($rec)"
            continue
        fi
        remaining=$((remaining + 1))
        if [ "$st" = "D" ]; then
            D_STREAK[$r]=$(( ${D_STREAK[$r]:-0} + 1 ))
        else
            D_STREAK[$r]=0
        fi
        note=""
        if [ "${D_STREAK[$r]}" -ge "$D_LIMIT" ]; then
            note="!WEDGED-D(${D_STREAK[$r]})"
        fi
        line="$line | r$r:$sz B st=$st et=${et}s $note"
        [ "$st" = "X" ] && line="$line !NO-PROC"
    done
    echo "$line"
    if [ "$remaining" -eq 0 ]; then
        echo "$stamp ALL-15-DONE"
        exit 0
    fi
    sleep "$INTERVAL"
done
