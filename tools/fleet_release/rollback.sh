#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
part=${1:-}
case "$part" in
    all|engines|root|api) ;;
    *) echo "usage: rollback.sh all|engines|root|api"
       echo "  all: engines (hold if needed, root back, weightd back if it changed, wait applied, unhold old, converge old; skipped when already production) then api"
       exit 2;;
esac
start_log "rollback-$part"
K="$KIT"
hubstate() {
    hub "echo \$(sha256sum < ~/release/$RELEASE_ROOT_NAME/MANIFEST | cut -d' ' -f1) \$(sha256sum < ~/$RELEASE_API_CHANNEL/bin/sparkpipe_model_api | cut -c1-16) \$(sha256sum < ~/$RELEASE_API_CHANNEL/runtime/lib/model_serving_adapter.so | cut -c1-16) \$(sha256sum < ~/$RELEASE_API_CHANNEL/model_resident.json | cut -c1-16) \$(tr -d '[:space:]' < ~/release/core/WEIGHTSD_BIN)"
}

engines_production() {
    nodes --expect hold=no --expect eng_n=1 --expect "ready>=1" --expect exe="$OLD_RESIDENTD" --expect drv="$OLD_DRIVER" --expect wd="$OLD_WEIGHTD" --expect applied="${SERVED_MANIFEST_SHA:0:16}" --log "$log.nodes-now" > /dev/null
}

do_root() {
    echo "== root rollback to MANIFEST ${SERVED_MANIFEST_SHA:0:16} $(date -u +%H:%M:%SZ)"
    hub_script rollback_root.sh $HUB_VARS || die "root rollback failed"
}

do_engines() {
    if engines_production; then
        echo "engines: already production $NODE_COUNT/$NODE_COUNT (not held, $OLD_RESIDENTD/$OLD_DRIVER on ${SERVED_MANIFEST_SHA:0:16}, weightd $OLD_WEIGHTD); no restart needed"
        return 0
    fi
    if ! nodes --expect hold=yes --expect eng_n=0 --log "$log.nodes-held" > /dev/null; then
        bash "$K/hold.sh" any || die "hold (rollback) failed"
    fi
    read -r served api adapter dep announced <<< "$(hubstate)"
    [ "$served" = "$SERVED_MANIFEST_SHA" ] || do_root
    if weightd_changes && [ "$announced" != "$OLD_WEIGHTD" ]; then
        bash "$K/weightd.sh" rollback || die "weightd rollback failed (nodes stay held)"
    fi
    echo "== wait for every agent to apply ${SERVED_MANIFEST_SHA:0:16} (held)"
    nodes --expect hold=yes --expect eng_n=0 --expect applied="${SERVED_MANIFEST_SHA:0:16}" --expect rootok=yes --wait-min 5 --interval 5 --quiet-poll --log "$log.nodes-applied" | tail -n 1
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "nodes did not apply the production MANIFEST in 5 min (still held; read $log.nodes-applied)"
    bash "$K/unhold.sh" old || die "unhold old failed"
    bash "$K/converge.sh" old || die "engines did not converge back to $OLD_RESIDENTD/$OLD_DRIVER"
}

do_api() {
    echo "== API rollback $(date -u +%H:%M:%SZ) (restores the backup channel unless api, adapter, model_resident.json and SHA256SUMS are already production)"
    hub_script rollback_api.sh $HUB_VARS || die "API rollback failed"
}

case "$part" in
    root)
        nodes --expect hold=yes --expect eng_n=0 --log "$log.nodes-held" > /dev/null || die "rollback.sh root alone runs only while every node is held and down (agents restart engines on each file change otherwise); use rollback.sh all"
        do_root;;
    engines) do_engines;;
    api) do_api;;
    all)
        echo "rollback all $(date -u +%FT%TZ): hold (if not held) -> root back to ${SERVED_MANIFEST_SHA:0:16} (if published) -> weightd back to $OLD_WEIGHTD (if announced) -> wait applied -> unhold old -> converge old -> API channel back to $OLD_API/$OLD_ADAPTER/$CHANNEL_DEPLOYMENT"
        do_engines
        do_api;;
esac
read -r served api adapter dep announced <<< "$(hubstate)"
echo "ROLLBACK $part DONE $(date -u +%H:%M:%SZ): served=${served:0:16} api=$api adapter=$adapter channel=$dep weightd=$announced (production: ${SERVED_MANIFEST_SHA:0:16}, $OLD_API, $OLD_ADAPTER, $CHANNEL_DEPLOYMENT, $OLD_WEIGHTD)"
if [ "$part" = all ] && [ "$served $api $adapter $dep $announced" != "$SERVED_MANIFEST_SHA $OLD_API $OLD_ADAPTER $CHANNEL_DEPLOYMENT $OLD_WEIGHTD" ]; then
    echo "ROLLBACK INCOMPLETE: hub is not production"
    exit 1
fi
