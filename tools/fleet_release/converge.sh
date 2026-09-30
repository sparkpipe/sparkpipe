#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
target=${1:-new}
minutes=${2:-$RELEASE_CONVERGE_MIN}
case "$target" in
    new) rules=(--expect exe="$NEW_RESIDENTD" --expect drv="$NEW_DRIVER" --expect applied="${NEW_MANIFEST_SHA:0:16}" --expect wd="$NEW_WEIGHTD")
         for e in $RELEASE_CONVERGE_EXPECT; do [ "$e" = none ] || rules+=(--expect "$e"); done;;
    old) rules=(--expect exe="$OLD_RESIDENTD" --expect drv="$OLD_DRIVER" --expect applied="${SERVED_MANIFEST_SHA:0:16}" --expect wd="$OLD_WEIGHTD");;
    *) echo "usage: converge.sh [new|old] [minutes]"; exit 2;;
esac
start_log converge
start=$(drain_start)
echo "converge $target $(date -u +%FT%TZ): wait up to $minutes min for $NODE_COUNT/$NODE_COUNT (hold at $(date -u -r "$start" +%H:%M:%SZ 2>/dev/null || date -u -d "@$start" +%H:%M:%SZ))"
nodes --expect agent=active --expect hold=no --expect eng_n=1 --expect wd_n=1 --expect "ready>=1" --expect "mem_gib>=$RELEASE_MEM_FLOOR_GIB" "${rules[@]}" --wait-min "$minutes" --interval 15 --quiet-poll --log "$log.nodes"
rc=$?
echo "elapsed since hold: $(( $(date -u +%s) - start )) s"
[ $rc -eq 0 ] || die "read 'journalctl --user -u fleet-agent' and residentd.log (ERRSITE lines; read include/sparkpipe/spark_status.h for the status) on the BAD nodes above; roll back (rollback.sh all) if an engine is not ready 10 min after unhold"
if [ "$target" = new ]; then
    grep -h " verify=" "$log.nodes" | tail -n "$NODE_COUNT" | grep -o "verify=[a-z0-9]*" | sort | uniq -c | sed 's/^/pack verify mode: /'
    python3 "$KIT/engine_logs.py" --checks "$RELEASE_CHECKS" --section converge_logs || die "engine log gate (converge_logs) failed"
fi
echo "CONVERGED $target $NODE_COUNT/$NODE_COUNT $(date -u +%H:%M:%SZ): residentd $([ "$target" = new ] && echo "$NEW_RESIDENTD driver $NEW_DRIVER weightd $NEW_WEIGHTD" || echo "$OLD_RESIDENTD driver $OLD_DRIVER weightd $OLD_WEIGHTD"). Next: $([ "$target" = new ] && echo api_install.sh || echo "rollback.sh api"). log $log"
