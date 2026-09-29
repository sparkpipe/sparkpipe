#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
start_log publish
echo "publish $S7 $(date -u +%FT%TZ): root MANIFEST ${NEW_MANIFEST_SHA:0:16} ($EXPECT_CHANGED changed entries) while every node is held. weightd $NEW_WEIGHTD. Engines start only at unhold.sh."
rotation_gate || die "pause the rotation first"
read -r announced served <<< "$(hub "echo \$(tr -d '[:space:]' < ~/release/core/WEIGHTSD_BIN) \$(sha256sum < ~/release/$RELEASE_ROOT_NAME/MANIFEST | cut -d' ' -f1)")"
echo "hub: WEIGHTSD_BIN=$announced served=${served:0:16}"
[ "$announced" = "$NEW_WEIGHTD" ] || die "WEIGHTSD_BIN is $announced, want $NEW_WEIGHTD$(weightd_changes && echo " (run weightd.sh publish, announce and wait first)")"
[ "$served" = "$SERVED_MANIFEST_SHA" ] || die "served root MANIFEST is ${served:0:16}, not production ${SERVED_MANIFEST_SHA:0:16}"

echo "== gate: $NODE_COUNT/$NODE_COUNT held, no residentd of the root, production MANIFEST applied, weightd $NEW_WEIGHTD running"
nodes --expect agent=active --expect hold=yes --expect eng_n=0 --expect applied="${SERVED_MANIFEST_SHA:0:16}" --expect wd="$NEW_WEIGHTD" --expect wd_n=1 --log "$log.nodes-gate" > /dev/null \
    || { nodes_bad "$log.nodes-gate"; die "node gate failed: run hold.sh first$(weightd_changes && echo ", then weightd.sh publish/announce/wait")"; }
echo "OK   node gate $NODE_COUNT/$NODE_COUNT"

echo "== root publish $(date -u +%H:%M:%SZ)"
hub_script publish_root.sh $HUB_VARS 2>&1 | tee "$LOGS/root-publish-$ts.out"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "root publish failed (nodes stay held, nothing started). Read $LOGS/root-publish-$ts.out. If it failed after rollback-saved, run rollback.sh root before any retry"
grep -q "^PUBLISHED " "$LOGS/root-publish-$ts.out" || die "root publish did not print PUBLISHED"

echo "== wait for every agent to apply the new MANIFEST (held: files sync and verify, nothing starts)"
nodes --expect hold=yes --expect eng_n=0 --expect applied="${NEW_MANIFEST_SHA:0:16}" --expect rootok=yes --wait-min 5 --interval 5 --quiet-poll --log "$log.nodes-applied" | tail -n 1
[ "${PIPESTATUS[0]}" -eq 0 ] || die "not every node applied ${NEW_MANIFEST_SHA:0:16} with a verified root in 5 min (read $log.nodes-applied; 'journalctl --user -u fleet-agent' on the BAD nodes). Nodes stay held. Retry the wait, or rollback.sh engines"
echo "PUBLISH DONE $(date -u +%H:%M:%SZ): root ${NEW_MANIFEST_SHA:0:16} published, applied and verified (sha256sum -c) $NODE_COUNT/$NODE_COUNT, still held. Next: unhold.sh new. log $log"
