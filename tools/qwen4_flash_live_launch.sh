#!/usr/bin/env bash
# qwen4_flash v3 LIVE TP4 launcher: start residentd on every rank node, wait
# for the ready line, then the api on the coordinator (rank 0).
#
# Parameterized (no hardcoded nodes, per lane rules):
#   --hosts LIST    comma-separated rank hostnames in rank order (default
#                   spark4,spark5,spark6,spark7)
#   --deploy-dir D  the shared-layout deployment directory ON EACH NODE
#                   (default: /home/<host>/sparkdata/qwen4_flash.tp4/deploy_v3)
#   --rank R        launch ONLY this rank's residentd (single-node mode)
#   --api-only      launch only the coordinator api
#   --term          TERM any running residentd/api first (never KILL)
set -euo pipefail

hosts="spark4,spark5,spark6,spark7"
deploy_dir=""
rank=""
api_only=0
term=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --hosts) hosts="$2"; shift 2 ;;
    --deploy-dir) deploy_dir="$2"; shift 2 ;;
    --rank) rank="$2"; shift 2 ;;
    --api-only) api_only=1; shift ;;
    --term) term=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -ra host_array <<< "$hosts"

launch_node() {
  local host="$1" r="$2" dir="$3"
  ssh -o BatchMode=yes "$host" bash -s <<REMOTE
set -euo pipefail
dir="$dir"
if [[ $term -eq 1 ]]; then
  for proc in sparkpipe_model_residentd sparkpipe_model_api; do
    pid=\$(pgrep -f "\$proc.*--deployment" || true)
    if [[ -n "\$pid" ]]; then
      kill -TERM \$pid || true
      for _ in \$(seq 1 50); do pgrep -f "\$proc.*--deployment" >/dev/null || break; sleep 0.2; done
      pgrep -af "\$proc.*--deployment" && echo "$host: \$proc still alive after TERM (NO KILL per protocol)" >&2 || true
    fi
  done
fi
mkdir -p "\$dir/runtime-$r"
cd "\$dir"
nohup "\$dir/bin/sparkpipe_model_residentd" --deployment "\$dir" --rank-index "$r" > "\$dir/residentd-r$r.log" 2>&1 &
echo "$host: residentd rank $r launched pid \$!"
REMOTE
}

wait_ready() {
  local host="$1" r="$2" dir="$3" tries=0
  while ssh -o BatchMode=yes "$host" "grep -a 'model_residentd ready' '$dir/residentd-r$r.log' >/dev/null 2>&1" \
    || [[ $tries -ge 300 ]]; do
    break
  done
  # poll explicitly for a bound deadline (2 min)
  for _ in $(seq 1 240); do
    if ssh -o BatchMode=yes "$host" "grep -a 'ready' '$dir/residentd-r$r.log' >/dev/null 2>&1"; then
      echo "$host: rank $r READY: $(ssh -o BatchMode=yes "$host" "grep -a 'ready' '$dir/residentd-r$r.log' | tail -1")"
      return 0
    fi
    sleep 0.5
  done
  echo "$host: rank $r NOT READY after 120s; log tail:" >&2
  ssh -o BatchMode=yes "$host" "tail -5 '$dir/residentd-r$r.log'" >&2
  return 1
}

if [[ $api_only -eq 1 ]]; then
  host="${host_array[0]}"
  dir="${deploy_dir:-/home/$host/sparkdata/qwen4_flash.tp4/deploy_v3}"
  ssh -o BatchMode=yes "$host" bash -s <<REMOTE
set -euo pipefail
cd "$dir"
nohup "\$dir/bin/sparkpipe_model_api" --deployment "\$dir" --rank-index 0 > "\$dir/api.log" 2>&1 &
echo "$host: api launched pid \$!"
REMOTE
  exit 0
fi

status=0
if [[ -n "$rank" ]]; then
  host="${host_array[$rank]}"
  dir="${deploy_dir:-/home/$host/sparkdata/qwen4_flash.tp4/deploy_v3}"
  launch_node "$host" "$rank" "$dir"
  wait_ready "$host" "$rank" "$dir" || status=1
else
  for r in "${!host_array[@]}"; do
    host="${host_array[$r]}"
    dir="${deploy_dir:-/home/$host/sparkdata/qwen4_flash.tp4/deploy_v3}"
    launch_node "$host" "$r" "$dir"
  done
  for r in "${!host_array[@]}"; do
    host="${host_array[$r]}"
    dir="${deploy_dir:-/home/$host/sparkdata/qwen4_flash.tp4/deploy_v3}"
    wait_ready "$host" "$r" "$dir" || status=1
  done
fi
exit $status
