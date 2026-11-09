#!/bin/bash
set -u
E=/Users/cem/sparkpipe-agent-e2e
F=$E/fleet-20261109
H=/private/tmp/claude-501/-Users-cem-sparkpipe--claude-worktrees-nice-elion-edff10/9a30ff9a-6d9f-4914-b3e3-457fef440e67/scratchpad/hb/bin/harbor
LOG=$F/driver.log
STOP_HHMM=2100
MIN_FREE_GB=15
export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock
export COLUMNS=160

stamp() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S KST'; }

controller_state() {
  ssh -n -o BatchMode=yes spec@100.123.97.61 'printf "Authorization: Bearer %s\n" "$(cat /srv/workspace/sparkpipe/swap-controller/token)" | curl -s -m 10 -H @- 127.0.0.1:8440/status' \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['active'], d['state'], d.get('wanted'), d.get('target'))"
}

engine_live() {
  ssh -n -o BatchMode=yes spec@100.123.97.61 'curl -s -m 10 127.0.0.1:8463/health' | python3 -c "import json,sys; print(json.load(sys.stdin).get('live_requests'))"
}

colima_free_gb() {
  colima ssh -- df -BG --output=avail /var/lib/docker < /dev/null | tail -1 | tr -dc '0-9'
}

batch=0
while read -r line <&3; do
  [ -z "$line" ] && continue
  batch=$((batch + 1))
  name=$(printf 'tb21-cc-b%02d' "$batch")
  now=$(TZ=Asia/Seoul date +%H%M)
  if [ "$now" -ge "$STOP_HHMM" ]; then
    echo "$(stamp) STOP before $name: it is past 21:00 KST" >> "$LOG"
    break
  fi
  state=$(controller_state)
  if [ "$state" != "deepseek-v4.1-flash idle None None" ]; then
    echo "$(stamp) STOP before $name: controller reads '$state'" >> "$LOG"
    break
  fi
  live=""
  for attempt in 1 2 3 4; do
    live=$(engine_live)
    [ "$live" = "0" ] && break
    sleep 20
  done
  if [ "$live" != "0" ]; then
    echo "$(stamp) STOP before $name: the engine has $live live requests with no batch running" >> "$LOG"
    break
  fi
  free=$(colima_free_gb)
  if [ "$free" -lt "$MIN_FREE_GB" ]; then
    echo "$(stamp) STOP before $name: colima has $free GB free" >> "$LOG"
    break
  fi
  filters=()
  for task in $line; do
    filters+=(-i "terminal-bench/$task")
  done
  count=$(echo "$line" | wc -w | tr -d ' ')
  echo "$(stamp) START $name ($count tasks: $line) controller='$state' live=$live colima_free=${free}G" >> "$LOG"
  "$H" run -d terminal-bench/terminal-bench-2-1 "${filters[@]}" -a claude-code -m deepseek-v4.1-flash \
    --ak version=2.1.311 --ak config="$F/cc-deepseek.json" --ak reasoning_effort=max \
    --env-file "$F/cc-deepseek.env" -n "$count" -k 1 -o "$F/jobs" --job-name "$name" > "$F/$name.out" 2>&1 < /dev/null
  echo "$(stamp) END $name rc=$?" >> "$LOG"
  failed=$(python3 -c "
import json, pathlib, sys
trials = [json.loads(p.read_text()) for p in pathlib.Path(sys.argv[1]).glob('*/result.json')]
trials = [t for t in trials if 'task_name' in t]
print('all' if trials and all(t.get('exception_info') for t in trials) else 'no' if trials else 'none')" "$F/jobs/$name")
  if [ "$failed" != "no" ]; then
    echo "$(stamp) STOP after $name: every trial raised an exception or none ran ($failed)" >> "$LOG"
    break
  fi
done 3< "$F/task-batches.txt"
echo "$(stamp) DRIVER DONE after $batch batches" >> "$LOG"
