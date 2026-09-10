#!/usr/bin/env bash
set -euo pipefail
TREE="$HOME/sparkpipe-build"
RELEASE="$HOME/release/core"
mkdir -p "$RELEASE/bin"
install -m 755 "$TREE/build/sparkpipe_weightd" "$RELEASE/bin/sparkpipe_weightd"
install -m 755 "$TREE/tools/fleet_node_agent.sh" "$RELEASE/bin/fleet_node_agent.sh"
cd "$RELEASE"
find bin -type f | sort | xargs sha256sum > MANIFEST.tmp
mv MANIFEST.tmp MANIFEST
echo "published core -> $RELEASE (weightd $(sha256sum < bin/sparkpipe_weightd | cut -c1-12), agent $(sha256sum < bin/fleet_node_agent.sh | cut -c1-12))"
