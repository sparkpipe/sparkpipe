#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
need BUILD_HOST BUILD_DIR SOURCE_REPO
start_log precheck
fail=0
ok() { echo "OK   $*"; }
bad() { echo "FAIL $*"; fail=$((fail + 1)); }
echo "precheck $S7 (read-only) $(date -u +%FT%TZ): root $RELEASE_ROOT_NAME on $NODE_COUNT nodes, weightd $OLD_WEIGHTD -> $NEW_WEIGHTD"

echo "== rotation"
rotation_gate || fail=$((fail + 1))

echo "== source"
if git -C "$SOURCE_REPO" merge-base --is-ancestor "$SHA" origin/main; then ok "$S7 is on origin/main (last fetched: $(git -C "$SOURCE_REPO" rev-parse --short origin/main))"; else bad "$S7 is not an ancestor of the fetched origin/main"; fi
ci=$(cd "$SOURCE_REPO" && sh tools/sparkpipe_github_pat.sh gh api "repos/sparkpipe/sparkpipe/commits/$SHA/check-runs" --jq '.check_runs[] | "\(.name)=\(.conclusion)"' 2>&1 | sort | tr '\n' ' ')
case "$ci" in *"compile-sm121a=success"*"host-tests=success"*|*"host-tests=success"*"compile-sm121a=success"*) ok "GitHub CI on $S7: $ci";; *) bad "GitHub CI on $S7 not green: $ci";; esac

echo "== hub $RELEASE_HUB"
hub_script precheck.sh $HUB_VARS || fail=$((fail + 1))

echo "== build host $BUILD_HOST ($BUILD_DIR)"
b=$($RELEASE_SSH "$BUILD_HOST" "cd $BUILD_DIR && echo head=\$(git rev-parse HEAD) clean=\$(git diff --quiet HEAD && echo yes || echo no) tar=\$(cut -c1-16 build/$RELEASE_ARTIFACT.tar.gz.sha256)")
echo "INFO $b"
for kv in "head=$SHA" "clean=yes"; do
    case " $b " in *" $kv "*) ok "$kv";; *) bad "build host: want $kv";; esac
done

echo "== nodes (production: residentd $OLD_RESIDENTD, driver $OLD_DRIVER, weightd $OLD_WEIGHTD, MANIFEST ${SERVED_MANIFEST_SHA:0:16} applied)"
nodes --expect agent=active --expect hold=no --expect layout="$RELEASE_LAYOUT" --expect applied="${SERVED_MANIFEST_SHA:0:16}" --expect rootok=yes --expect eng_n=1 --expect exe="$OLD_RESIDENTD" --expect drv="$OLD_DRIVER" --expect wd="$OLD_WEIGHTD" --expect wd_inst="$OLD_WEIGHTD" --expect wd_n=1 --expect "ready>=1" --expect "mem_gib>=$RELEASE_MEM_FLOOR_GIB" --log "$log.nodes" || fail=$((fail + 1))
others=$(grep -h "others=" "$log.nodes" | grep -v "others=none" | sed 's/^[A-Z]* *\([^ ]*\) .*others=\([^ ]*\).*/\1:\2/' | sort -u | tr '\n' ' ')
[ -n "$others" ] && echo "INFO other residentds running: $others (they may run through a root-only release; a weightd change needs them stopped; perf.sh needs them idle)"
if weightd_changes; then
    echo "== weightd bundle staged (weightd.sh stage + receipts first)"
    nodes --expect stage_wd="$NEW_WEIGHTD" --expect receipt=rc0 --log "$log.nodes-bundle" > /dev/null || { nodes_bad "$log.nodes-bundle"; bad "bundle $NEW_WEIGHTD not staged with valid receipts on every node (weightd.sh stage, receipts-adopt, receipts-verify)"; }
fi
echo "INFO PERF_HOLDER: $(cat "${PERF_HOLDER_FILE:-/dev/null}" 2>/dev/null | cut -c1-120)"

if [ "$fail" -eq 0 ]; then echo "PRECHECK PASS $(date -u +%H:%M:%SZ) log $log"; exit 0; fi
echo "PRECHECK FAIL ($fail) log $log"
exit 1
