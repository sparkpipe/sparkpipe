#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
start_log api-install
echo "api_install $S7 $(date -u +%FT%TZ): x86 api $NEW_API + adapter $NEW_ADAPTER + model_resident.json $NEW_CHANNEL_DEPLOYMENT (= $CHANNEL_DEPLOYMENT + $CHANNEL_ADD) into $RELEASE_HUB:~/$RELEASE_API_CHANNEL, one $RELEASE_API_UNIT restart"
nodes --expect eng_n=1 --expect "ready>=1" --expect exe="$NEW_RESIDENTD" --expect drv="$NEW_DRIVER" --expect wd="$NEW_WEIGHTD" --log "$log.nodes" > /dev/null \
    || { nodes_bad "$log.nodes"; die "engines are not $NODE_COUNT/$NODE_COUNT on $S7; run converge.sh new first"; }
echo "OK   engines $NODE_COUNT/$NODE_COUNT on $S7"
hub_script api_install.sh $HUB_VARS || die "hub install failed (read the lines above; rollback.sh api restores the backup channel: binary, adapter and model_resident.json together)"
echo "API INSTALLED $(date -u +%H:%M:%SZ) ($(( $(date -u +%s) - $(drain_start) )) s since the hold). Next: smoke.sh. log $log"
