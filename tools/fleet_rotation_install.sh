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
usage: $0 check | install | upgrade [--no-resume] | revert [BACKUP] | status | plan | dry-run [UTC-TIME] | rollback
  check     read-only: hub reachable, hub ssh to every node from a user unit, port $PORT free, config valid on the hub
  install   copy tool + config to $HUB:~/$DIR, write the units, pause the rotation, enable --now the timer and the schedule responder
  upgrade   installed rotation: pause (unless already paused), wait for a running tick, back up tool + config + state to ~/$DIR/backup-<UTC>,
            copy the new tool + config, check-config (restores the backup on failure), read-only plan, dry-run tick, restart the
            schedule responder, resume (only when upgrade itself paused; --no-resume leaves it paused)
  revert    put back the tool + config of the newest backup (or BACKUP): pause, wait, restore, check-config, plan, dry-run tick, resume as upgrade
  status    timer state and the rotation's 'now' view
  plan      the installed tool's read-only plan for this hour (observes, reads MemAvailable, runs prechecks; starts and stops nothing)
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

wait_tick() {
  local i=0 tick
  while :; do
    tick="$(on_hub "systemctl --user show -p ActiveState --value fleet-rotation.service")" || { echo "$1: cannot read the tick unit state on $HUB" >&2; exit 1; }
    case "$tick" in activating|active|deactivating|reloading) ;; *) break ;; esac
    [ "$i" -ge "$WAIT_POLLS" ] && { echo "a tick is still $tick after $((WAIT_POLLS / 2)) min; stop it by hand (systemctl --user stop fleet-rotation.service) and rerun $1" >&2; exit 1; }
    [ "$i" -eq 0 ] && echo "waiting for the running tick ($tick) to finish"
    sleep 30; i=$((i + 1))
  done
}

pause_for() {
  if [ "$(on_hub "test -e ~/$DIR/ROTATION_PAUSE && echo paused || echo running")" = paused ]; then
    echo "rotation already paused: $(on_hub "cat ~/$DIR/ROTATION_PAUSE") (left paused afterwards)"
    return 1
  fi
  on_hub "$TOOL pause $1"
  return 0
}

preview() {
  on_hub "$TOOL check-config && $TOOL plan; $TOOL --dry-run tick"
}

swap_in() {
  local tool="$1" config="$2" label="$3" we_paused="$4" backup="$5"
  scp -q "$tool" "$HUB:$DIR/bin/fleet_rotation.py.new"
  scp -q "$config" "$HUB:$DIR/rotation.json.new"
  on_hub "cd ~/$DIR && python3 bin/fleet_rotation.py.new --config rotation.json.new check-config" || { on_hub "rm -f ~/$DIR/bin/fleet_rotation.py.new ~/$DIR/rotation.json.new"; echo "$label: the new tool refuses the new config; nothing changed (rotation left paused)" >&2; exit 1; }
  on_hub "cd ~/$DIR && mv -f bin/fleet_rotation.py.new bin/fleet_rotation.py && mv -f rotation.json.new rotation.json && sha256sum bin/fleet_rotation.py rotation.json > INSTALLED.sha256 && echo \"$label \$(date -u +%FT%TZ) backup $backup\" >> INSTALL.log"
  on_hub "$TOOL check-config" || { on_hub "cd ~/$DIR && cp -p $backup/fleet_rotation.py bin/fleet_rotation.py && cp -p $backup/rotation.json rotation.json && cp -p $backup/INSTALLED.sha256 INSTALLED.sha256"; echo "$label: check-config failed on the hub; backup $backup restored (rotation left paused)" >&2; exit 1; }
  preview
  on_hub "systemctl --user restart fleet-rotation-schedule.service"
  if [ "$we_paused" = yes ] && [ "${NO_RESUME:-0}" != 1 ]; then
    on_hub "$TOOL resume"
  else
    echo "$label: rotation left paused; resume with: $SSH $HUB '$TOOL resume'"
  fi
  status
}

upgrade() {
  check
  local stamp we_paused=no backup
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  backup="backup-$stamp"
  pause_for "upgrade-$stamp" && we_paused=yes
  wait_tick "${FUNCNAME[0]}"
  on_hub "cd ~/$DIR && mkdir $backup && cp -p bin/fleet_rotation.py rotation.json INSTALLED.sha256 $backup/ && { cp -p state.json $backup/ 2>/dev/null || true; } && echo $(git -C "$HERE" rev-parse --short HEAD) > $backup/UPGRADED_TO"
  swap_in "$HERE/tools/fleet_rotation.py" "$CONFIG" "upgraded to $(git -C "$HERE" rev-parse --short HEAD)" "$we_paused" "$backup"
}

revert() {
  local backup="${1:-}" stamp we_paused=no scratch
  [ -n "$backup" ] || backup="$(on_hub "cd ~/$DIR && ls -1d backup-* 2>/dev/null | tail -n 1")"
  [ -n "$backup" ] || { echo "revert: no backup-* directory in ~/$DIR on $HUB" >&2; exit 1; }
  on_hub "test -s ~/$DIR/$backup/fleet_rotation.py && test -s ~/$DIR/$backup/rotation.json" || { echo "revert: ~/$DIR/$backup is incomplete" >&2; exit 1; }
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  pause_for "revert-$stamp" && we_paused=yes
  wait_tick "${FUNCNAME[0]}"
  on_hub "cd ~/$DIR && mkdir revert-$stamp && cp -p bin/fleet_rotation.py rotation.json INSTALLED.sha256 revert-$stamp/"
  scratch="$(mktemp -d)"
  scp -q "$HUB:$DIR/$backup/fleet_rotation.py" "$scratch/fleet_rotation.py"
  scp -q "$HUB:$DIR/$backup/rotation.json" "$scratch/rotation.json"
  swap_in "$scratch/fleet_rotation.py" "$scratch/rotation.json" "reverted to $backup" "$we_paused" "revert-$stamp"
  rm -rf "$scratch"
}

plan() {
  on_hub "$TOOL plan"
}

rollback() {
  on_hub "echo \"rollback \$(date -u +%FT%TZ)\" > ~/$DIR/ROTATION_PAUSE.tmp && mv -f ~/$DIR/ROTATION_PAUSE.tmp ~/$DIR/ROTATION_PAUSE; systemctl --user disable --now fleet-rotation.timer"
  wait_tick "${FUNCNAME[0]}"
  on_hub "systemctl --user disable --now fleet-rotation-schedule.service"
  on_hub "$TOOL converge --rollback"
  status
}

case "${1:-}" in
  check) check ;;
  install) install ;;
  upgrade) [ "${2:-}" = --no-resume ] && NO_RESUME=1; upgrade ;;
  revert) revert "${2:-}" ;;
  plan) plan ;;
  status) status ;;
  dry-run) dry_run "${2:-}" ;;
  rollback) rollback ;;
  *) usage ;;
esac
