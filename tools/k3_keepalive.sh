#!/usr/bin/env bash
# k3_keepalive.sh — supervisor for a K3 stage-pack build on this node.
#
# The sparke pack build died 3+ times unattended (agent-session deaths,
# neighbor-daemon OOMs, a probe-loop syntax error). This loop ends that
# pattern: k3_pack.py journals every tensor, so a restart RESUMES from the
# journal instead of starting over. Deployed as /home/sparke/k3build/
# keepalive.sh (a copy of this committed file) and run under nohup.
#
# Hard rules honored (docs/AGENT_LANE_BRIEFS/README.md):
#   - the packer pid is captured AT SPAWN and is the ONLY pid this script
#     ever signals; TERM only, never KILL; no pkill/pgrep anywhere.
#   - a live-but-stalled packer is REPORTED (state + journal age), never
#     restarted: two writers on one pack = corruption. Restart happens
#     only when the captured pid is confirmed gone.
#   - a restart is deferred while the ceph warm source refuses 8 MB direct
#     reads or the node has <20 GiB memory headroom (the probe gate from
#     the proven probe_and_resume.sh, generalized).
#   - SIGTERM to this supervisor is relayed as TERM to the captured child;
#     the supervisor never escalates.
#
# usage: nohup bash keepalive.sh <first> <count> <pack_path> >> keepalive.log 2>&1 &
set -u
FIRST="${1:?usage: keepalive.sh FIRST COUNT PACK_PATH}"
COUNT="${2:?usage: keepalive.sh FIRST COUNT PACK_PATH}"
PACK="${3:?usage: keepalive.sh FIRST COUNT PACK_PATH}"
SRC=/mnt/model-warm/kimi-k3
BASE="$(cd "$(dirname "$0")" && pwd)"
SRCDIR="${K3_SRC_DIR:-/home/sparke/k3finish-src}"   # committed-branch checkout
STALL_MIN=45

log() { echo "KEEPALIVE $(date +%F_%T) $*"; }

child=""
shutdown() {
    log "SIGTERM received: TERMing captured child ${child:-none} and exiting"
    if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
        kill -TERM "$child"
    fi
    exit 0
}
trap shutdown TERM INT

pack_done() {
    [ -s "$PACK" ] || return 1
    [ -e "${PACK}.journal" ] && return 1     # journal is unlinked on clean finish
    local s1 s2
    s1=$(stat -c %s "$PACK" 2>/dev/null) || return 1
    sleep 5
    s2=$(stat -c %s "$PACK" 2>/dev/null) || return 1
    [ "$s1" = "$s2" ]
}

probe_ok() {  # the warm source must actually serve before a restart is worth it
    local s
    for s in model-00012-of-000096.safetensors \
             model-00040-of-000096.safetensors \
             model-00060-of-000096.safetensors \
             model-00070-of-000096.safetensors; do
        timeout 8 dd if="$SRC/$s" of=/dev/null iflag=direct bs=1M count=8 skip=8000 2>/dev/null || return 1
    done
    return 0
}

node_headroom_ok() {
    [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 20 ]
}

journal_state() {
    if [ -e "${PACK}.journal" ]; then
        echo "journal present ($(wc -l < "${PACK}.journal") tensors)"
    else
        echo "journal none"
    fi
}

log "START first=$FIRST count=$COUNT pack=$PACK src=$SRCDIR"
restarts=0
child=""
spawn() {
    cd "$SRCDIR" || { log "FATAL: source checkout missing at $SRCDIR"; exit 1; }
    env PYTHONDONTWRITEBYTECODE=1 python3 tools/k3_pack.py \
        "$SRC" "$PACK" "$FIRST" "$COUNT" >> "$BASE/pack_${FIRST}_${COUNT}.log" 2>&1 &
    child=$!
    log "SPAWN pid=$child restarts=$restarts $(journal_state)"
}

while true; do
    if pack_done; then
        log "DONE pack complete ($(du -h "$PACK" | cut -f1)); supervisor exiting"
        exit 0
    fi
    if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
        # Alive: watch for a stall, report only.
        if [ -e "${PACK}.journal" ]; then
            age=$(( ($(date +%s) - $(stat -c %Y "${PACK}.journal")) / 60 ))
            if [ "$age" -ge "$STALL_MIN" ]; then
                state=$(awk '/^State:/{print $2}' "/proc/$child/status" 2>/dev/null || echo '?')
                log "STALL journal idle ${age}m pid=$child state=$state — reporting only, not restarting"
                sleep 300
                continue
            fi
        fi
        sleep 30
        continue
    fi
    # Dead or not yet spawned: restart from the journal (it resumes).
    restarts=$((restarts + 1))
    if ! probe_ok; then
        log "WAIT warm source not serving; restart #$restarts deferred"
        sleep 240
        continue
    fi
    if ! node_headroom_ok; then
        log "WAIT node headroom <20 GiB; restart #$restarts deferred"
        sleep 120
        continue
    fi
    log "RESTART #$restarts (previous pid ${child:-none} gone; $(journal_state))"
    spawn
    sleep 30
done
