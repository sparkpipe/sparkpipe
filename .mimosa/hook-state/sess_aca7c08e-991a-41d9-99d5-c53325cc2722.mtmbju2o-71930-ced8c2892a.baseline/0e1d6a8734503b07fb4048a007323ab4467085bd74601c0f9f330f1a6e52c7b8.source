#!/usr/bin/env bash
#
# fleet_status.sh — one line per Spark: which residentd is running (rank,
# cwd) or free. Run before every measured window; include with receipts.
#
# usage: tools/devcycle/fleet_status.sh
set -u

HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)

for h in "${HOSTS[@]}"; do
    out="$(ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" '
        p=$(pgrep -f "^bin/sparkpipe_model_residentd" | head -1)
        if [[ -n "$p" ]]; then
            rank=$(ps -o args= -p "$p" | sed -nE "s/.*rank-index ([0-9]+).*/\\1/p")
            cwd=$(readlink /proc/$p/cwd 2>/dev/null)
            printf "residentd rank=%s cwd=%s" "$rank" "$cwd"
        else
            printf "free"
        fi' 2>/dev/null)"
    printf '%-8s %s
' "$h" "${out:-unreachable}"
done
