#!/usr/bin/env bash
#
# bench_claim.sh — mkdir-based claim guard for measured benchmark windows.
#
# Why: on 2026-08-25 two agent sessions ran measured O512/prefill sweeps on
# spark3 concurrently. Each restarts residentd between cells, so daemon
# restarts killed the other session's in-flight connects (batch_status=4,
# terminal=0, sub-second failures), and any overlap invalidates both
# measurements. COORDINATION.md already requires exclusive measured windows;
# this tool makes the exclusivity machine-checkable.
#
# Protocol: claim BEFORE your pre-window daemon restart, beat every few
# minutes during long cells, release AFTER your last measurement.
#
# Claims are directories under /tmp on the target Spark (shared across
# sessions). Acquisition is an atomic mkdir. A held claim is stale — and may
# be reaped — once its directory mtime exceeds the TTL (default 900 s).
#
# usage:
#   bench_claim.sh claim   <host> <window-name> [ttl_s]      default ttl 900
#   bench_claim.sh beat    <host> <window-name>              refresh TTL
#   bench_claim.sh release <host> <window-name>
#   bench_claim.sh status  <host>
#   bench_claim.sh wait    <host> <window-name> [timeout_s]  default 1800
#
# exit codes: 0 ok/free/released · 1 busy/timeout · 2 usage
set -u

DIR_BASE="/tmp/sparkpipe_bench_claims"

usage() { echo "usage: $0 claim|beat|release|status|wait host window-name [arg]" >&2; exit 2; }

CMD="$1"; HOST="$2"
case "$CMD" in
    status) [ $# -eq 2 ] || usage ;;
    *) { [ $# -eq 3 ] || [ $# -eq 4 ]; } || usage ;;
esac
NAME=""
[ $# -ge 3 ] && NAME="$3"
ARG=""
[ $# -ge 4 ] && ARG="$4"

D="$DIR_BASE/$HOST"'__'"$NAME"

ssh_run() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "$@"; }

case "$CMD" in
claim)
    TTL="$ARG"
    [ -n "$TTL" ] || TTL=900
    ssh_run "mkdir -p '$DIR_BASE'"
    # holder identity = this local process pid (owner bookkeeping only)
    if ssh_run "mkdir '$D' 2>/dev/null"; then
        ssh_run "echo $$ > '$D/pid'"
        ssh_run "date -Is >> '$D/since'"
        echo "CLAIMED $D"
        exit 0
    fi
    AGE=$(ssh_run 'echo $(( $(date +%s) - $(stat -c %Y '"$D"' 2>/dev/null || date +%s) ))')
    case "$AGE" in ''|*[!0-9]*) AGE=999999 ;; esac
    if [ "$AGE" -gt "$TTL" ]; then
        # expired: single reaper wins the rm+mkdir race, loser reports busy
        if ssh_run "rm -rf '$D' && mkdir '$D' 2>/dev/null"; then
            ssh_run "echo $$ > '$D/pid'"
            ssh_run "date -Is >> '$D/since'"
            echo "REAPED-STALE-AND-CLAIMED $D stale_age_s=$AGE"
            exit 0
        fi
        echo "BUSY lost-stale-reap-race"
        exit 1
    fi
    HOLDER=$(ssh_run "cat '$D/pid' 2>/dev/null")
    SINCE=$(ssh_run "tail -1 '$D/since' 2>/dev/null")
    [ -n "$HOLDER" ] || HOLDER="?"
    [ -n "$SINCE" ] || SINCE="?"
    echo "BUSY holder_pid=$HOLDER since=$SINCE age_s=$AGE ttl_s=$TTL"
    exit 1
    ;;
beat)
    if ssh_run "[ -d '$D' ] && touch '$D'"; then
        echo "BEAT-OK $D"
    else
        echo "NO-CLAIM $D"
        exit 1
    fi
    ;;
release)
    if ssh_run "[ -d '$D' ] && rm -rf '$D'"; then
        echo "RELEASED $D"
    else
        echo "NOT-HELD $D already-gone-or-nonempty"
    fi
    ;;
status)
    ssh_run "mkdir -p '$DIR_BASE'"
    CLAIMS=$(ssh_run "ls -1 '$DIR_BASE'" | grep -- "^$HOST")
    for c in $CLAIMS; do
        d="$DIR_BASE/$c"
        P=$(ssh_run "cat '$d/pid' 2>/dev/null")
        S=$(ssh_run "tail -1 '$d/since' 2>/dev/null")
        A=$(ssh_run 'test -d '"$d"' && echo $(( $(date +%s) - $(stat -c %Y '"$d"') )) || echo gone')
        [ -n "$P" ] || P="?"
        [ -n "$S" ] || S="?"
        echo "$c pid=$P since=$S age_s=$A"
    done
    ;;
wait)
    TIMEOUT="$ARG"
    [ -n "$TIMEOUT" ] || TIMEOUT=1800
    DEADLINE=$(( SECONDS + TIMEOUT ))
    while [ "$SECONDS" -lt "$DEADLINE" ]; do
        if "$0" claim "$HOST" "$NAME" >/dev/null 2>&1; then
            echo "ACQUIRED after $SECONDS s"
            exit 0
        fi
        sleep 15
    done
    echo "WAIT-TIMEOUT after $TIMEOUT s"
    exit 1
    ;;
*)
    usage
    ;;
esac
