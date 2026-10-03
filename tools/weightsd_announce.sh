#!/usr/bin/env bash
set -euo pipefail
RELEASE="${1:-$HOME/release/core}"
BIN="$RELEASE/bin/sparkpipe_weightd"
STABLE="$RELEASE/WEIGHTSD_BIN"
[ -f "$BIN" ] || { echo "no candidate at $BIN (run tools/fleet_release/weightd.sh publish first)" >&2
    exit 1; }
sha=$(sha256sum < "$BIN" | cut -c1-16)
if [ "$(cat "$STABLE" 2>/dev/null | tr -d '[:space:]')" = "$sha" ]; then
    echo "already announced: $STABLE = $sha"
    exit 0
fi
printf '%s\n' "$sha" > "$STABLE.tmp"
mv "$STABLE.tmp" "$STABLE"
echo "announced weightd stable $sha -> $STABLE"
echo "every agent installs $sha and restarts weightd once no engine depends on it; verify with tools/fleet_release/weightd.sh wait"
