#!/usr/bin/env bash
# fleet_sync.sh — keep every node's runtime root in sync with a reference
# directory. The reference IS the manifest: build into it, the loop
# converges the fleet within one sleep. Syncs only the software subtrees;
# logs, packs and kvcache in the roots are untouched.
#
# usage: tools/fleet_sync.sh REFERENCE_ROOT RUNTIME_ROOT_NAME [once]
set -uo pipefail
REF="${1:?reference root}"
NAME="${2:?runtime root name (under ~/sparkdata/)}"
MODE="${3:-loop}"
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
RSYNC="rsync -a --info=name0"
PARTS=(lib bin stages config model_resident.json model_package.json)

sync_once() {
    local h p
    for h in "${HOSTS[@]}"; do
        for p in "${PARTS[@]}"; do
            [ -e "$REF/$p" ] || continue
            $RSYNC "$REF/$p" "$h:sparkdata/$NAME/" &
        done
    done
    wait
}

if [ "$MODE" = once ]; then
    sync_once
    echo "synced once"
    exit 0
fi
echo "sync loop: $REF -> sparkdata/$NAME on ${#HOSTS[@]} hosts (ctrl-c to stop)"
while true; do
    sync_once
    sleep 5
done
