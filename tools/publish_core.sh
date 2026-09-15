#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?agent|weightd}"
TREE="$HOME/sparkpipe-build"
RELEASE="$HOME/release/core"
mkdir -p "$RELEASE/bin" "$HOME/release"
exec 9>"$HOME/release/.core.publish.lock"
flock 9
case "$MODE" in
    agent) install -m 755 "$TREE/tools/fleet_node_agent.sh" "$RELEASE/bin/fleet_node_agent.sh" ;;
    weightd) install -m 755 "$TREE/build/sparkpipe_weightd" "$RELEASE/bin/sparkpipe_weightd" ;;
    *) echo "mode must be agent or weightd" >&2
       exit 2 ;;
esac
cd "$RELEASE"
find bin -type f | sort | xargs sha256sum > MANIFEST.tmp
mv MANIFEST.tmp MANIFEST
wd_sha=none
agent_sha=none
[ -f bin/sparkpipe_weightd ] && wd_sha=$(sha256sum < bin/sparkpipe_weightd | cut -c1-12)
[ -f bin/fleet_node_agent.sh ] && agent_sha=$(sha256sum < bin/fleet_node_agent.sh | cut -c1-12)
echo "published core ($MODE) -> $RELEASE (weightd $wd_sha, agent $agent_sha)"
