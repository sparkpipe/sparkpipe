#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
mode=${1:-release}
case "$mode" in
    release) gate=(--expect agent=active --expect hold=no --expect layout="$RELEASE_LAYOUT" --expect eng_n=1 --expect exe="$OLD_RESIDENTD" --expect drv="$OLD_DRIVER" --expect wd="$OLD_WEIGHTD" --expect wd_n=1 --expect applied="${SERVED_MANIFEST_SHA:0:16}")
             weightd_changes && gate+=(--expect others=none --expect stage_wd="$NEW_WEIGHTD" --expect receipt=rc0);;
    any) gate=(--expect agent=active --expect layout="$RELEASE_LAYOUT" --expect wd_n=1);;
    *) echo "usage: hold.sh [release|any]"; exit 2;;
esac
start_log hold
echo "hold ($mode) $(date -u +%FT%TZ): agent.hold on the $NODE_COUNT $RELEASE_ROOT_NAME roots; each fleet-agent drains its residentd. The model is DOWN from here until unhold.sh + converge.sh + api_install.sh."
rotation_gate || die "pause the rotation first"
nodes "${gate[@]}" --log "$log.nodes-gate" > /dev/null || { nodes_bad "$log.nodes-gate"; die "node gate ($mode) failed"; }
echo "OK   node gate $NODE_COUNT/$NODE_COUNT"
mkdir -p "$LOGS"
date -u +%s > "$LOGS/drain-start"
echo "HOLD START $(date -u +%H:%M:%SZ)"
on_nodes "test -d ~/sparkdata/$RELEASE_ROOT_NAME && touch ~/sparkdata/$RELEASE_ROOT_NAME/agent.hold && echo held"
nodes --expect hold=yes --expect eng_n=0 --wait-min 3 --interval 5 --quiet-poll --log "$log.nodes-down" | tail -n 1
[ "${PIPESTATUS[0]}" -eq 0 ] || die "not every node is held with its residentd down after 3 min (read $log.nodes-down); agents drain with TERM then kill -9 after 15 s"
echo "HOLD DONE $(date -u +%H:%M:%SZ): $NODE_COUNT/$NODE_COUNT held, 0 residentds of $RELEASE_ROOT_NAME. Next: $(weightd_changes && echo "weightd.sh publish" || echo publish.sh) (release) or rollback.sh. log $log"
