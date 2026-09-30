#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
target=${1:-new}
case "$target" in
    new) want=$NEW_MANIFEST_SHA wd=$NEW_WEIGHTD;;
    old) want=$SERVED_MANIFEST_SHA wd=$OLD_WEIGHTD;;
    *) echo "usage: unhold.sh [new|old]"; exit 2;;
esac
start_log unhold
echo "unhold ($target) $(date -u +%FT%TZ): remove agent.hold on $NODE_COUNT/$NODE_COUNT; each agent starts its residentd from the applied root ${want:0:16}"
served=$(hub "sha256sum < ~/release/$RELEASE_ROOT_NAME/MANIFEST | cut -d' ' -f1")
[ "$served" = "$want" ] || die "served MANIFEST is ${served:0:16}, want ${want:0:16}"
nodes --expect hold=yes --expect eng_n=0 --expect applied="${want:0:16}" --expect rootok=yes --expect wd="$wd" --expect wd_n=1 --expect "mem_gib>=$RELEASE_MEM_UNHOLD_GIB" --log "$log.nodes-gate" > /dev/null \
    || { nodes_bad "$log.nodes-gate"; die "node gate: every node must be held, down, on ${want:0:16} with a verified root, weightd $wd, MemAvailable >= $RELEASE_MEM_UNHOLD_GIB GiB"; }
echo "OK   node gate $NODE_COUNT/$NODE_COUNT"
on_nodes "rm -f ~/sparkdata/$RELEASE_ROOT_NAME/agent.hold && echo released"
echo "UNHOLD SENT $(date -u +%H:%M:%SZ). Next: converge.sh $target. log $log"
