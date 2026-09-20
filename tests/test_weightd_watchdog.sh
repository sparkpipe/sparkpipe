#!/usr/bin/env bash
# Watchdog unit tests for tools/fleet_node_agent.sh ensure_weightd.
#
# The fleet vortex was the watchdog killing weightd mid-bake: a young
# process still attaching its packs looked "unresponsive" (no socket yet),
# got kill -9'd, and the crash loop repeated. The fix under test:
#   1. a process younger than the grace period is never touched
#   2. a stale process is PROBED (3x) before any kill
#   3. a live listener on the socket keeps the process alive
#   4. restarts are backoff-limited (no restart storm)
#
# Linux-only (the agent reads /proc/$pid/stat and /proc/uptime). Runs
# against REAL processes: the fake weightd is a copy of the sleep binary
# so /proc/<pid>/exe matches the on-disk sha and the update-recycle path
# stays out of the way.
set -euo pipefail
cd "$(dirname "$0")/.."
AGENT=tools/fleet_node_agent.sh

if [ "$(uname -s)" != "Linux" ]; then
    echo "SKIP: watchdog test requires Linux /proc"
    exit 0
fi

# Safety: the agent's clear path kills anything matching sparkpipe_weightd
# system-wide. Never run on a machine with a live fleet weightd. (The
# bracket in the pattern keeps this very check from matching itself.)
if pgrep -f "[s]parkpipe_weightd" >/dev/null 2>&1; then
    echo "SKIP: a sparkpipe_weightd process is already running; refusing to interfere"
    exit 0
fi

SB=$(mktemp -d /tmp/watchdog_test.XXXXXX)
trap 'rm -rf "$SB"; [ -n "${FAKE_PID:-}" ] && kill -9 "$FAKE_PID" 2>/dev/null; [ -n "${LISTENER_PID:-}" ] && kill -9 "$LISTENER_PID" 2>/dev/null; exit 0' EXIT
PASS=0
FAIL=0
ok()  { PASS=$((PASS + 1)); echo "PASS: $1"; }
bad() { FAIL=$((FAIL + 1)); echo "FAIL: $1"; }

export HOME="$SB/home"
mkdir -p "$HOME/sparkdata/weightd" "$HOME/.ssh" "$SB/bin" /tmp/weightd-mesh
# The fake weightd is a copy of the python3 binary: /proc/<pid>/exe matches
# the on-disk sha (no update-recycle), argv[0] is the weightd path (pgrep -f
# matches), and a multicall-binary copy (busybox/coreutils sleep) would
# dispatch on argv[0] and die instantly.
cp "$(command -v python3)" "$HOME/sparkdata/weightd/sparkpipe_weightd"
chmod +x "$HOME/sparkdata/weightd/sparkpipe_weightd"
fake_weightd() { "$HOME/sparkdata/weightd/sparkpipe_weightd" -c "import time; time.sleep(300)" & }

# Start attempts are recorded, never executed: the setsid stub hands its
# args to the PATH-resolved nohup stub, which records and exits.
printf '#!/bin/sh\necho "nohup $*" >> "%s/starts"\n' "$SB" > "$SB/bin/nohup"
printf '#!/bin/sh\n"$@"\n' > "$SB/bin/setsid"
chmod +x "$SB/bin/nohup" "$SB/bin/setsid"
export PATH="$SB/bin:$PATH"
: > "$SB/starts"

# Extract the agent's function definitions (everything before main).
# keep every function definition, skip the top-level statements
# (the agent interleaves: ensure_root is defined after the banner block)
awk '/^echo "\$\$" > "\$PID_FILE"/ {skip=1; next}
     skip==1 && /^ensure_root\(\)/ {skip=0}
     skip==1 {next}
     /^while true; do/ {exit}
     {print}' "$AGENT" > "$SB/extract.sh"
hostname() { echo spark3; }
RANK=0
MESH_INTERFACE=stub0
MESH_SGID_INDEX=0
source "$SB/extract.sh" testroot

SOCK=/tmp/spark_weightd.sock
rm -f "$SOCK"

# --- case 1: a young process inside the grace window is never touched ---
fake_weightd
FAKE_PID=$!
sleep 1
ensure_weightd
if kill -0 "$FAKE_PID" 2>/dev/null; then ok "young weightd inside grace survives"; else bad "young weightd was killed inside grace"; fi

# --- case 2: grace expired + no listener -> probed, killed, restarted ---
# The fake is seconds old; shrink the grace to 0 so it counts as stale.
SPARK_AGENT_WEIGHTD_GRACE_S=0 ensure_weightd
sleep 1
if kill -0 "$FAKE_PID" 2>/dev/null; then bad "stale unresponsive weightd was NOT killed"; else ok "stale unresponsive weightd killed after probes"; fi
if [ -s "$SB/starts" ]; then ok "replacement weightd start attempted"; else bad "no restart after clearing a stale weightd"; fi
unset FAKE_PID

# --- case 3: a live listener on the socket keeps an old process alive ---
fake_weightd
FAKE_PID=$!
python3 - <<'PY' &
import socket
s = socket.socket(socket.AF_UNIX)
s.bind("/tmp/spark_weightd.sock")
s.listen(4)
import time
time.sleep(60)
PY
LISTENER_PID=$!
sleep 1
SPARK_AGENT_WEIGHTD_GRACE_S=0 ensure_weightd
if kill -0 "$FAKE_PID" 2>/dev/null; then ok "responsive weightd survives past grace"; else bad "responsive weightd was killed"; fi
kill -9 "$FAKE_PID" 2>/dev/null; unset FAKE_PID
kill -9 "$LISTENER_PID" 2>/dev/null; unset LISTENER_PID
rm -f "$SOCK"

# --- case 4: restart backoff suppresses a storm ---
: > "$SB/starts"
unset BACKOFF NEXT_OK
declare -A BACKOFF NEXT_OK
LAST_ANY_RESTART=0
ensure_weightd
ensure_weightd
starts=$(wc -l < "$SB/starts")
if [ "$starts" = "1" ]; then ok "second immediate restart suppressed by backoff"; else bad "restart storm: $starts starts for two calls"; fi

# --- case 5: install_core accepts a full-length announced sha ---
# The hub's WEIGHTSD_BIN may carry the full 64-char sha256 while sha16 of
# the on-disk binary is 16 chars; the agent must compare like lengths or
# the install gate is a permanent no-op (the fleet ran stale weightds).
mkdir -p "$HOME/sparkdata/core/bin" "$SB/release/core"
printf 'fake-new-weightd-binary' > "$HOME/sparkdata/core/bin/sparkpipe_weightd"
chmod +x "$HOME/sparkdata/core/bin/sparkpipe_weightd"
printf 'stale-old-weightd' > "$HOME/sparkdata/weightd/sparkpipe_weightd"
full_sha=$(sha256sum < "$HOME/sparkdata/core/bin/sparkpipe_weightd" | cut -c1-64)
printf '%s\n' "$full_sha" > "$SB/release/core/WEIGHTSD_BIN"
RELEASE_HTTP="file://$SB/release"
install_core
want=$(sha256sum < "$HOME/sparkdata/core/bin/sparkpipe_weightd" | cut -c1-16)
got=$(sha256sum < "$HOME/sparkdata/weightd/sparkpipe_weightd" | cut -c1-16)
if [ "$got" = "$want" ]; then ok "install_core installs on a 64-char announced sha"; else bad "install_core no-op on 64-char announced sha ($got != $want)"; fi

# --- case 6: a mid-bake weightd (no socket yet, CPU advancing) is never killed ---
# case 5 installed a text stub over the fake binary; restore it.
cp "$(command -v python3)" "$HOME/sparkdata/weightd/sparkpipe_weightd"
# The vortex was the watchdog killing weightd mid-bake: the socket probe
# always fails during the bake because the server loop does not accept
# while baking. CPU advancement is the liveness signal.
"$HOME/sparkdata/weightd/sparkpipe_weightd" -c "
import time
t = time.time() + 45
x = 0
while time.time() < t:
    x += 1
" &
FAKE_PID=$!
sleep 1
SPARK_AGENT_WEIGHTD_GRACE_S=0 ensure_weightd
if kill -0 "$FAKE_PID" 2>/dev/null; then ok "mid-bake weightd survives (CPU advancing, no socket)"; else bad "mid-bake weightd was killed"; fi
kill -9 "$FAKE_PID" 2>/dev/null; unset FAKE_PID

# --- case 7: a weightd restart recycles the engine attached to it ---
# The engine's attachment (socket + mesh memfd) is scoped to the weightd
# process; a weightd restart orphans every engine. The agent must detect
# an engine older than its weightd and recycle it.
mkdir -p "$HOME/sparkdata/testroot/bin" "$HOME/sparkdata/testroot/stages/stage_000"
cp "$(command -v python3)" "$HOME/sparkdata/testroot/bin/sparkpipe_model_residentd"
chmod +x "$HOME/sparkdata/testroot/bin/sparkpipe_model_residentd"
echo "model_residentd ready rank=0" > "$HOME/sparkdata/testroot/residentd.log"
cp "$(command -v python3)" "$HOME/sparkdata/weightd/sparkpipe_weightd"
# fake engine, cwd = the runtime root (root_state keys on cwd)
( cd "$HOME/sparkdata/testroot" && "$HOME/sparkdata/testroot/bin/sparkpipe_model_residentd" -c "import time; time.sleep(300)" ) &
ENG=$!
# fake weightd
"$HOME/sparkdata/weightd/sparkpipe_weightd" -c "import time; time.sleep(300)" &
FAKE_PID=$!
sleep 1
ensure_root testroot || true
if kill -0 "$ENG" 2>/dev/null; then ok "engine stable while weightd is stable"; else bad "engine killed while weightd stable"; fi
# restart the weightd: kill it, start a new one (younger)
kill -9 "$FAKE_PID" 2>/dev/null
"$HOME/sparkdata/weightd/sparkpipe_weightd" -c "import time; time.sleep(300)" &
FAKE_PID=$!
sleep 1
ensure_root testroot || true
sleep 1
if kill -0 "$ENG" 2>/dev/null; then bad "engine NOT recycled after a weightd restart"; else ok "weightd restart recycles the orphaned engine"; fi
kill -9 "$FAKE_PID" 2>/dev/null; unset FAKE_PID
unset ENG

echo "test_weightd_watchdog: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
