#!/usr/bin/env bash
# k3_fleet_wave.sh — ONE-SIMULTANEOUS-WAVE launcher for the K3 TP4xPP4
# 16-rank fleet (the deployment contract demands all 16 ranks in one wave:
# the hidden-transport connect window refuses late joiners, and a rank
# without its fleet sits in transport-wait holding its full ~91 GiB weight
# in unified memory - the exact hazard the fleet-only rule forbids).
#
# Usage:
#   k3_fleet_wave.sh check          # verify every rank's prerequisites
#   k3_fleet_wave.sh launch         # TERM-sweep own pids, 45s, one wave
#   k3_fleet_wave.sh stop           # TERM own captured pids only, wait
#   k3_fleet_wave.sh status         # ready-line / pid liveness report
#
# Hard rules encoded here (docs/AGENT_LANE_BRIEFS/README.md):
#   - TERM never KILL; pids come ONLY from our own pidfiles (no fuzzy
#     pkill on a shared node).
#   - refuses to launch unless EVERY rank's pack is present and no
#     foreign sparkpipe_model residentd is running (exclusive window;
#     weights ~91.4 GiB + pools ~1.8 GiB per node vs the 110 GiB
#     operator ceiling - a second model's rank cannot coexist).
#   - 45s pre-launch sleep: EADDRINUSE TIME_WAIT at wave start.
set -u
RT_SUBDIR="sparkdata/k3.mxfp4.tp4pp4"
READY_LINE="model_residentd ready"
POLL_DEADLINE_S=900

hexes=(0 1 2 3 4 5 6 7 8 9 a b c d e f)

host_of() { printf 'spark%s' "$1"; }
rt_of()   { printf '/home/%s/%s' "$(host_of "$1")" "$RT_SUBDIR"; }

cmd_check() {
  local bad=0 i host rt pack rank
  for i in "${!hexes[@]}"; do
    host=$(host_of "${hexes[$i]}"); rt=$(rt_of "${hexes[$i]}")
    rank=$((i % 4))
    pack="k3.stage$((i / 4)).rank0${rank}.pack"
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "
      test -x $rt/bin/sparkpipe_model_residentd || echo 'no residentd'
      test -s $rt/lib/libk3_serving_adapter.so || echo 'no adapter so'
      test -s $rt/lib/hidden_transport.so || echo 'no transport so'
      test -s $rt/config/model_resident.json || echo 'no deployment json'
      test -s $rt/config/adapter.json || echo 'no adapter json'
      test -s $rt/packs/$pack || echo 'no pack'
      # exclusive window: ANY resident sparkpipe_model daemon counts, but
      # pgrep -f self-matches this ssh wrapper (\$\$) and -x cannot tell
      # whose daemon it is - enumerate, skip self, report
      for p in \$(pgrep -f 'bin/sparkpipe_model_residentd'); do
        [ \"\$p\" = \"\$\$\" ] && continue
        echo 'DAEMON RUNNING'
        break
      done" 2>&1)
    # memory envelope (operator ceiling 110 GiB of 119 unified): one rank
    # holds ~91.4 GiB weights + ~1.8 GiB pools + ~2-4 GiB context
    # ~= 96-97 GiB. Refuse any node reporting < 100 GiB available -
    # NVRM kills silently near 114 GiB, and co-resident ranks do not fit.
    avail=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "free -g | awk '/^Mem:/{print \$7}'" 2>/dev/null)
    case "$avail" in
      ''|*[!0-9]*) echo "MISSING $host rank=$i: no readable memory figure"; bad=1; continue ;;
    esac
    if [ "$avail" -lt 100 ]; then
      out="$out MEM ${avail}G available < 100G envelope"
    fi
    if [ -n "$out" ]; then echo "MISSING $host rank=$i: $out"; bad=1
    else echo "ok $host rank=$i stage=$((i / 4)) pack=$pack avail=${avail}G"; fi
  done
  [ "$bad" = 0 ] && echo "FLEET PREREQUISITES COMPLETE" || echo "FLEET INCOMPLETE (do not launch)"
  return "$bad"
}

cmd_stop() {
  local i host rt pid
  for i in "${!hexes[@]}"; do
    host=$(host_of "${hexes[$i]}"); rt=$(rt_of "${hexes[$i]}")
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "
      if [ -s $rt/residentd.pid ]; then
        pid=\$(cat $rt/residentd.pid)
        if kill -0 \$pid 2>/dev/null; then
          kill -TERM \$pid && echo \"$host TERM \$pid\"
        else echo \"$host pid \$pid already gone\"; fi
      else echo \"$host no pidfile\"; fi
      # pidfile gap (live 2026-08-30: captured pre-exec pids that later
      # vanished while the real daemon lived): also sweep THIS rank's own
      # runtime-root daemons by anchored cmdline + exact cwd match - the
      # same deployment-scoped rule the census and the launch capture use.
      # Never touches another family's daemon (cwd filter is OUR rt).
      for p in \$(pgrep -f 'bin/sparkpipe_model_residentd'); do
        [ \"\$p\" = \"\$\$\" ] && continue
        [ \"\$(readlink /proc/\$p/cwd 2>/dev/null)\" = \"$rt\" ] && {
          kill -TERM \$p 2>/dev/null && echo \"$host TERM \$p (cwd-scoped)\"
        }
      done" 2>&1
  done
  echo "waiting for TERM to settle (30s)"; sleep 30
  cmd_status
}

cmd_launch() {
  cmd_check || { echo "refusing to launch: prerequisites incomplete" >&2; exit 1; }
  echo "45s pre-launch sleep (EADDRINUSE TIME_WAIT)"; sleep 45
  local i host
  for i in "${!hexes[@]}"; do
    host=$(host_of "${hexes[$i]}")
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "
      cd $(rt_of "${hexes[$i]}") || exit 1
      rm -f residentd-r$i.log
      LD_LIBRARY_PATH=\$PWD/lib setsid -f \
        ./bin/sparkpipe_model_residentd --deployment config/model_resident.json \
        --rank-index $i >residentd-r$i.log 2>&1 < /dev/null
      # setsid -f double-forks; capture the real daemon pid: cmdline match
      # (comm -x is deployment-blind: glm5_next's daemon has the same comm),
      # cwd-filtered to THIS runtime root, self(\$\$)-excluded; retried up
      # to 10s because the forked child is only pgrep-visible after exec
      pid=""
      for t in 1 2 3 4 5 6 7 8 9 10; do
        pid=\$(for p in \$(pgrep -f 'bin/sparkpipe_model_residentd'); do
          [ \"\$p\" = \"\$\$\" ] && continue
          [ \"\$(readlink /proc/\$p/cwd 2>/dev/null)\" = \"\$PWD\" ] && echo \"\$p\"
        done | tail -1)
        [ -n \"\$pid\" ] && break
        sleep 1
      done
      echo \$pid > residentd.pid
      echo \"$host rank=$i pid=\$pid\"" 2>&1 &
  done
  wait
  echo "WAVE LAUNCHED (all 16 ranks); polling for ready lines up to ${POLL_DEADLINE_S}s"
  cmd_status --poll
}

cmd_status() {
  local poll="${1:-}" i host rt elapsed=0
  while :; do
    local ready=0 lines=""
    for i in "${!hexes[@]}"; do
      host=$(host_of "${hexes[$i]}"); rt=$(rt_of "${hexes[$i]}")
      line=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "
        grep -c '$READY_LINE' $rt/residentd-r$i.log 2>/dev/null || echo 0" 2>/dev/null)
      lines="$lines ${hexes[$i]}:$(echo "$line" | tail -1)"
      [ "$(echo "$line" | tail -1)" != "0" ] && ready=$((ready + 1))
    done
    echo "ready ranks: $ready/16 ($lines)"
    [ "$ready" = 16 ] && { echo "FLEET READY"; return 0; }
    [ "$poll" != "--poll" ] && return 1
    elapsed=$((elapsed + 30)); [ "$elapsed" -ge "$POLL_DEADLINE_S" ] && { echo "POLL DEADLINE"; return 1; }
    sleep 30
  done
}

case "${1:-}" in
  check)  cmd_check ;;
  launch) cmd_launch ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  *) echo "usage: k3_fleet_wave.sh check|launch|stop|status" >&2; exit 2 ;;
esac
