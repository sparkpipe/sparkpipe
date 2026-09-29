set -uo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$KIT/../.." && pwd)
if [ -z "${RELEASE_ENV:-}" ] || [ ! -f "$RELEASE_ENV" ]; then
    echo "RELEASE_ENV must name this release's env file (template: tools/fleet_release/release.env.example)"
    exit 2
fi
. "$RELEASE_ENV"

need() {
    local v
    for v in "$@"; do
        if [ -z "${!v:-}" ]; then echo "$RELEASE_ENV: $v is not set"; exit 2; fi
        case "${!v}" in *PENDING*) echo "$RELEASE_ENV: $v is still PENDING"; exit 2;; esac
    done
}

need RELEASE_NODES RELEASE_HUB RELEASE_ROOT_NAME RELEASE_RANK_PACK RELEASE_DRIVER RELEASE_LAYOUT \
    RELEASE_WEIGHTD_STAGE RELEASE_PROBE_EXTRA RELEASE_CHECKS RELEASE_ARTIFACT \
    RELEASE_API_CHANNEL RELEASE_API_UNIT RELEASE_API_PORT RELEASE_ADAPTER_BUILD \
    RELEASE_ROTATION_PAUSE_FILE RELEASE_ROTATION_UNIT RELEASE_MEM_UNHOLD_GIB RELEASE_MEM_FLOOR_GIB \
    RELEASE_CONVERGE_EXPECT RELEASE_CONVERGE_MIN RELEASE_API_REQUIRED_STRING RELEASE_API_REQUIRED_LOG \
    RELEASE_BUILD_FILES RELEASE_STAGE_WORLD LOGS CHANNEL_ADD STAGE_CONFIG_ADD \
    SHA S7 PREV7 SERVED_MANIFEST_SHA NEW_MANIFEST_SHA EXPECT_CHANGED \
    OLD_RESIDENTD NEW_RESIDENTD OLD_DRIVER NEW_DRIVER OLD_WEIGHTD NEW_WEIGHTD \
    OLD_API NEW_API OLD_ADAPTER NEW_ADAPTER CHANNEL_DEPLOYMENT NEW_CHANNEL_DEPLOYMENT TOKENIZER_SHA
[ -f "$RELEASE_CHECKS" ] || { echo "RELEASE_CHECKS $RELEASE_CHECKS is not a file"; exit 2; }
[ "$RELEASE_PROBE_EXTRA" = none ] || [ -f "$RELEASE_PROBE_EXTRA" ] || { echo "RELEASE_PROBE_EXTRA $RELEASE_PROBE_EXTRA is not a file (or none)"; exit 2; }

RELEASE_SSH=${RELEASE_SSH:-ssh -o BatchMode=yes -o ConnectTimeout=10}
export RELEASE_NODES RELEASE_ROOT_NAME RELEASE_RANK_PACK RELEASE_DRIVER RELEASE_WEIGHTD_STAGE RELEASE_PROBE_EXTRA RELEASE_SSH
NODE_COUNT=$(echo $RELEASE_NODES | wc -w | tr -d ' ')
ASSEMBLE="release-assemble-$S7"
API_BUILD="api-build-$S7"

start_log() {
    STEP=$1
    mkdir -p "$LOGS"
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    log="$LOGS/$STEP-$ts.log"
    exec > >(tee "$log") 2>&1
}

die() {
    echo "${STEP:-release} STOPPED: $*"
    [ -n "${log:-}" ] && echo "log $log"
    exit 1
}

nodes() {
    python3 "$KIT/nodes.py" "$@"
}

nodes_bad() {
    grep BAD "$1" | tail -n "$NODE_COUNT" | cut -c1-300
}

hub() {
    $RELEASE_SSH "$RELEASE_HUB" "$@"
}

env_args() {
    local v out=""
    for v in "$@"; do out="$out $v=$(printf %q "${!v}")"; done
    echo "$out"
}

hub_script() {
    local script=$1
    shift
    $RELEASE_SSH "$RELEASE_HUB" "$(env_args "$@") bash -s" < "$KIT/hub/$script"
}

HUB_VARS="S7 PREV7 SHA RELEASE_ROOT_NAME RELEASE_ARTIFACT RELEASE_API_CHANNEL RELEASE_API_UNIT RELEASE_API_PORT RELEASE_ADAPTER_BUILD RELEASE_DRIVER RELEASE_API_REQUIRED_STRING RELEASE_API_REQUIRED_LOG RELEASE_BUILD_FILES RELEASE_STAGE_WORLD NEW_WEIGHTD OLD_WEIGHTD SERVED_MANIFEST_SHA NEW_MANIFEST_SHA EXPECT_CHANGED NEW_RESIDENTD NEW_DRIVER OLD_API NEW_API OLD_ADAPTER NEW_ADAPTER CHANNEL_DEPLOYMENT NEW_CHANNEL_DEPLOYMENT TOKENIZER_SHA CHANNEL_ADD STAGE_CONFIG_ADD"

on_nodes() {
    local h
    for h in $RELEASE_NODES; do
        $RELEASE_SSH "$h" "$1" 2>&1 | sed "s/^/$h /" &
    done
    wait
}

rotation_gate() {
    if [ "$RELEASE_ROTATION_PAUSE_FILE" = none ]; then
        echo "INFO rotation gate off (RELEASE_ROTATION_PAUSE_FILE=none)"
        return 0
    fi
    local r
    r=$(hub "test -e ~/$RELEASE_ROTATION_PAUSE_FILE && echo paused || echo running; systemctl --user show -p ActiveState --value $RELEASE_ROTATION_UNIT 2>/dev/null || echo unknown")
    set -- $r
    if [ "${1:-}" != paused ]; then
        echo "FAIL rotation is not paused (~/$RELEASE_ROTATION_PAUSE_FILE missing on $RELEASE_HUB): pause it before any release step"
        return 1
    fi
    case "${2:-unknown}" in
        inactive|failed) echo "OK   rotation paused, $RELEASE_ROTATION_UNIT ${2} (no tick running)";;
        *) echo "FAIL rotation paused but $RELEASE_ROTATION_UNIT is ${2:-unknown}: a tick is still running; wait for it to exit"; return 1;;
    esac
}

drain_start() {
    cat "$LOGS/drain-start" 2>/dev/null || date -u +%s
}

weightd_changes() {
    [ "$OLD_WEIGHTD" != "$NEW_WEIGHTD" ]
}
