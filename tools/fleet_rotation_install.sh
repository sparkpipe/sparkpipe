#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${FLEET_ROTATION_CONFIG:-$HERE/deployment/fleet_rotation/rotation.json}"
HUB="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hub"])' "$CONFIG")"
PORT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["schedule_port"])' "$CONFIG")"
FLEET="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["fleet"]))' "$CONFIG")"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
DIR=fleet-rotation
TOOL="python3 \$HOME/$DIR/bin/fleet_rotation.py --config \$HOME/$DIR/rotation.json"
WAIT_POLLS=210

usage() {
  cat >&2 <<USAGE
usage: $0 check | install | status | dry-run [UTC-TIME] | rollback
  check     read-only: hub reachable, hub ssh to every node from a user unit, port $PORT free, config valid on the hub
  install   copy tool + config to $HUB:~/$DIR, write the units, pause the rotation, enable --now the timer and the schedule responder
  status    timer state and the rotation's 'now' view
  dry-run   the installed tool's tick in --dry-run (prints every command, runs none) at UTC-TIME (default now)
  rollback  pause, disable the timer, wait for a running tick (up to its 100 min TimeoutStartSec), stop the schedule responder, converge to the config's rollback_models
USAGE
  exit 2
}

on_hub() { $SSH "$HUB" "$@"; }

check() {
  python3 "$HERE/tools/fleet_rotation.py" --config "$CONFIG" check-config
  on_hub "command -v python3 >/dev/null && echo hub-python-ok"
  on_hub "systemctl --user reset-failed fleet-rotation-sshcheck 2>/dev/null; systemd-run --user --wait --pipe --quiet --unit=fleet-rotation-sshcheck -p RuntimeMaxSec=180 bash -c 'n=0; for h in $FLEET; do ssh -o BatchMode=yes -o ConnectTimeout=8 \$h true && n=\$((n+1)) || echo unreachable \$h; done; echo hub-ssh \$n/$(echo "$FLEET" | wc -w | tr -d ' ')'" | tee /dev/stderr | grep -q "hub-ssh $(echo "$FLEET" | wc -w | tr -d ' ')/" || { echo "check FAIL: the hub cannot reach every node from a user unit" >&2; exit 1; }
  if on_hub "systemctl --user is-active -q fleet-rotation-schedule"; then
    echo "schedule responder already active on :$PORT"
  else
    on_hub "! ss -ltn | grep -q ':$PORT '" || { echo "check FAIL: port $PORT busy on $HUB" >&2; exit 1; }
  fi
  echo "check PASS"
}

install() {
  check
  on_hub "mkdir -p ~/$DIR/bin ~/.config/systemd/user"
  scp -q "$HERE/tools/fleet_rotation.py" "$HUB:$DIR/bin/fleet_rotation.py"
  scp -q "$CONFIG" "$HUB:$DIR/rotation.json"
  on_hub "cd ~/$DIR && sha256sum bin/fleet_rotation.py rotation.json > INSTALLED.sha256 && git_rev=$(git -C "$HERE" rev-parse --short HEAD) && echo \"installed \$(date -u +%FT%TZ) from \$git_rev\" >> INSTALL.log && $TOOL check-config"
  on_hub "test -e ~/$DIR/ROTATION_PAUSE || echo \"install \$(date -u +%FT%TZ): waiting for the supervised first run (resume to start)\" > ~/$DIR/ROTATION_PAUSE"
  on_hub "cat > ~/.config/systemd/user/fleet-rotation.service" <<'UNIT'
[Unit]
Description=SparkPipe fleet rotation tick (docs/FLEET_ROTATION.md)
[Service]
Type=oneshot
WorkingDirectory=%h/fleet-rotation
ExecStart=/usr/bin/python3 %h/fleet-rotation/bin/fleet_rotation.py --config %h/fleet-rotation/rotation.json tick
TimeoutStartSec=100min
StandardOutput=append:%h/fleet-rotation/tick.out
StandardError=append:%h/fleet-rotation/tick.out
UNIT
  on_hub "cat > ~/.config/systemd/user/fleet-rotation.timer" <<'UNIT'
[Unit]
Description=SparkPipe fleet rotation tick every 5 minutes (slot changes at the top of each UTC hour)
[Timer]
OnCalendar=*:00/5:20
AccuracySec=5s
Persistent=false
[Install]
WantedBy=timers.target
UNIT
  on_hub "cat > ~/.config/systemd/user/fleet-rotation-schedule.service" <<'UNIT'
[Unit]
Description=SparkPipe fleet rotation schedule responder (/schedule.json, /v1/models)
After=network.target
[Service]
ExecStart=/usr/bin/python3 %h/fleet-rotation/bin/fleet_rotation.py --config %h/fleet-rotation/rotation.json serve-schedule
Restart=on-failure
RestartSec=5
MemoryMax=256M
[Install]
WantedBy=default.target
UNIT
  on_hub "systemctl --user daemon-reload && systemctl --user enable --now fleet-rotation-schedule.service && systemctl --user enable --now fleet-rotation.timer && systemctl --user list-timers fleet-rotation.timer --no-pager"
  status
  echo "installed PAUSED. First supervised run: see docs/FLEET_ROTATION.md (resume with: $SSH $HUB '$TOOL resume')"
}

status() {
  on_hub "systemctl --user is-active fleet-rotation.timer fleet-rotation-schedule.service fleet-rotation.service; $TOOL now; tail -n 5 ~/$DIR/ALERT 2>/dev/null || true"
}

dry_run() {
  local at="${1:-}"
  on_hub "$TOOL --dry-run ${at:+--at $at} tick"
}

rollback() {
  on_hub "echo \"rollback \$(date -u +%FT%TZ)\" > ~/$DIR/ROTATION_PAUSE.tmp && mv -f ~/$DIR/ROTATION_PAUSE.tmp ~/$DIR/ROTATION_PAUSE; systemctl --user disable --now fleet-rotation.timer"
  local i=0 tick
  while :; do
    tick="$(on_hub "systemctl --user show -p ActiveState --value fleet-rotation.service")" || { echo "rollback: cannot read the tick unit state on $HUB" >&2; exit 1; }
    case "$tick" in activating|active|deactivating|reloading) ;; *) break ;; esac
    [ "$i" -ge "$WAIT_POLLS" ] && { echo "a tick is still $tick after $((WAIT_POLLS / 2)) min; stop it by hand (systemctl --user stop fleet-rotation.service) and rerun rollback" >&2; exit 1; }
    [ "$i" -eq 0 ] && echo "waiting for the running tick ($tick) to finish"
    sleep 30; i=$((i + 1))
  done
  on_hub "systemctl --user disable --now fleet-rotation-schedule.service"
  on_hub "$TOOL converge --rollback"
  status
}

case "${1:-}" in
  check) check ;;
  install) install ;;
  status) status ;;
  dry-run) dry_run "${2:-}" ;;
  rollback) rollback ;;
  *) usage ;;
esac
