#!/usr/bin/env bash
# qwen4_flash v3 LIVE TP4 launcher (S5/S6 revision).
#
# Starts the residentd on every rank node as ONE SIMULTANEOUS WAVE (the
# 180s hidden-transport connect window refuses late joiners), waits for
# the per-rank ready line, then (optionally) the api on the coordinator.
#
# Safety model (lane rules 2026-08-28):
#   * Teardown TERMs ONLY the pids recorded in <deploy>/launch_pids.txt at
#     spawn time - NEVER a pgrep -f pattern (shared nodes host sibling
#     lanes' residentds; the fuzzy match is the documented killer).
#   * 45s pre-launch sleep so a previous wave's control sockets clear
#     TIME_WAIT (EADDRINUSE lesson from the glm53 bring-up).
#   * A memory watchdog polls each node during bring-up; past the
#     threshold it TERMs this deployment's own pids on that node (the
#     110 GiB operator ceiling; glm5_next co-resident ~32 GiB).
#
# Parameterized (no hardcoded nodes, per lane rules):
#   --hosts LIST      comma-separated rank hostnames in rank order
#                     (default spark4,spark5,spark6,spark7)
#   --rail-hosts LIST comma-separated rail IPs in rank order (TP collective)
#   --deploy-dir D    deployment directory ON EACH NODE (default
#                     /home/<host>/sparkdata/qwen4_flash.tp4/deploy_v3)
#   --port-base P     TP collective control port base (default 66640)
#   --identifier N    TP collective identifier (default: UTC seconds)
#   --pre-sleep S     seconds before the wave (default 45)
#   --ready-timeout S seconds to wait for the ready line (default 900)
#   --mem-limit-gib G per-node compute-memory watchdog limit (default 104)
#   --term            TERM the pids recorded in launch_pids.txt, then exit
#   --api-only        launch only the coordinator api (after the fleet)
#   --api-term        TERM the recorded api pid, then exit
set -euo pipefail

hosts="spark4,spark5,spark6,spark7"
rail_hosts="10.10.100.14,10.10.100.15,10.10.100.16,10.10.100.17"
deploy_dir=""
port_base="66640"
identifier="$(date -u +%s)"
pre_sleep="45"
ready_timeout="900"
mem_limit_gib="104"
api_only=0
api_term=0
term=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --hosts) hosts="$2"; shift 2 ;;
    --rail-hosts) rail_hosts="$2"; shift 2 ;;
    --deploy-dir) deploy_dir="$2"; shift 2 ;;
    --port-base) port_base="$2"; shift 2 ;;
    --identifier) identifier="$2"; shift 2 ;;
    --pre-sleep) pre_sleep="$2"; shift 2 ;;
    --ready-timeout) ready_timeout="$2"; shift 2 ;;
    --mem-limit-gib) mem_limit_gib="$2"; shift 2 ;;
    --api-only) api_only=1; shift ;;
    --api-term) api_term=1; shift ;;
    --term) term=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -ra host_array <<< "$hosts"
IFS=',' read -ra rail_array <<< "$rail_hosts"
[[ ${#host_array[@]} -eq ${#rail_array[@]} ]] || {
  echo "hosts (${#host_array[@]}) and rail-hosts (${#rail_array[@]}) must match" >&2; exit 2; }

dir_for() {  # host -> deploy dir (default is node-local)
  local host="$1"
  echo "${deploy_dir:-/home/$host/sparkdata/qwen4_flash.tp4/deploy_v3}"
}

term_recorded() {  # TERM exactly the pids this script recorded
  local host rank pid dir
  for entry in "${recorded[@]}"; do
    read -r host rank pid kind <<< "$entry"
    dir="$(dir_for "$host")"
    echo "TERM $kind pid $pid on $host (recorded)"
    ssh -o BatchMode=yes "$host" "kill -TERM $pid 2>/dev/null || echo '  (already gone)'" || true
  done
  for _ in $(seq 1 50); do
    local alive=0 entry host pid
    for entry in "${recorded[@]}"; do
      read -r host _rank pid _kind <<< "$entry"
      ssh -o BatchMode=yes "$host" "kill -0 $pid 2>/dev/null" && alive=1 || true
    done
    [[ $alive -eq 0 ]] && break
    sleep 0.5
  done
}

recorded=()
record_file=""

load_record() {  # host rank pid kind entries from the deploy dir's pid file
  local dir="$(dir_for "${host_array[0]}")"
  record_file="$dir/launch_pids.txt"
  [[ -z "$record_file" ]] && return 0
  local line host rank pid kind
  while read -r host rank pid kind; do
    [[ -n "$host" && -n "$pid" ]] && recorded+=("$host $rank $pid $kind")
  done < <(ssh -o BatchMode=yes "${host_array[0]}" "cat '$record_file' 2>/dev/null" || true)
}

if [[ $term -eq 1 || $api_term -eq 1 ]]; then
  load_record
  [[ ${#recorded[@]} -eq 0 ]] && { echo "no recorded pids (nothing to TERM)"; exit 0; }
  if [[ $term -eq 1 ]]; then
    term_recorded
    ssh -o BatchMode=yes "${host_array[0]}" "rm -f '$record_file'" || true
    echo "teardown complete (TERM only; no KILL per protocol)"
    exit 0
  fi
  # api-only teardown
  for entry in "${recorded[@]}"; do
    read -r host rank pid kind <<< "$entry"
    if [[ "$kind" == "api" ]]; then
      echo "TERM api pid $pid on $host (recorded)"
      ssh -o BatchMode=yes "$host" "kill -TERM $pid 2>/dev/null || echo '  (already gone)'"
      ssh -o BatchMode=yes "${host_array[0]}" "sed -i.bak '/$pid/d' '$record_file'" || true
    fi
  done
  exit 0
fi

if [[ $api_only -eq 1 ]]; then
  host="${host_array[0]}"
  dir="$(dir_for "$host")"
  pid=$(ssh -o BatchMode=yes "$host" bash -s <<REMOTE
set -euo pipefail
cd "$dir"
nohup "\$dir/bin/sparkpipe_model_api" --deployment "\$dir" --rank-index 0 > "\$dir/api.log" 2>&1 < /dev/null &
echo \$!
REMOTE
)
  echo "$host 0 $pid api" | ssh -o BatchMode=yes "$host" "cat >> '$(dir_for "$host")/launch_pids.txt'"
  echo "$host: api launched pid $pid (recorded)"
  exit 0
fi

# ---- fleet wave ----
[[ $EUID -ne 0 ]] || true
echo "qwen4_flash live launch: hosts=$hosts identifier=$identifier port_base=$port_base pre_sleep=${pre_sleep}s"

# TERM any previous wave recorded in the pid file (own pids only).
load_record
if [[ ${#recorded[@]} -gt 0 ]]; then
  echo "previous wave recorded (${#recorded[@]} procs); TERM sweep first"
  term_recorded
  ssh -o BatchMode=yes "${host_array[0]}" "rm -f '$record_file'" || true
  recorded=()
fi

echo "pre-launch sleep ${pre_sleep}s (TIME_WAIT clearance)..."
sleep "$pre_sleep"

launch_one() {  # host rank -> spawns residentd, echoes pid
  local host="$1" r="$2"
  local dir="$(dir_for "$host")"
  local rail_local="${rail_array[$r]}"
  local rail_csv="${rail_hosts}"
  ssh -o BatchMode=yes "$host" bash -s <<REMOTE
set -euo pipefail
dir="$dir"
mkdir -p "\$dir/runtime-$r"
cd "\$dir"
# Module TP collective env (the serving adapter sets only the stage env;
# the collective itself is env-wired, mirroring the P4 smoke pattern).
export SPARK_QWEN4_FLASH_TP_DEGREE=${#host_array[@]}
export SPARK_QWEN4_FLASH_TP_RANK=$r
export SPARK_QWEN4_FLASH_STAGE_TP_BACKEND_PATH="\$dir/lib/libhidden_transport_spark_host_rdma_verbs.so"
export SPARK_QWEN4_FLASH_STAGE_TP_IDENTIFIER=$identifier
export SPARK_QWEN4_FLASH_STAGE_TP_PORT_BASE=$port_base
export SPARK_QWEN4_FLASH_STAGE_TP_HOSTS="$rail_csv"
export SPARK_QWEN4_FLASH_STAGE_TP_LOCAL_HOST="$rail_local"
export SPARK_QWEN4_FLASH_STAGE_TP_TIMEOUT_MS=180000
export LD_LIBRARY_PATH="\$dir/lib:\${LD_LIBRARY_PATH:-}"
nohup "\$dir/bin/sparkpipe_model_residentd" --deployment "\$dir" --rank-index "$r" > "\$dir/residentd-r$r.log" 2>&1 < /dev/null &
echo \$!
REMOTE
}

# ONE WAVE: spawn every rank within the same instant (each pid recorded
# the moment it is captured, so a mid-wave failure still leaves a full
# teardown record).
record_host="${host_array[0]}"
record_dir="$(dir_for "$record_host")"
for r in "${!host_array[@]}"; do
  host="${host_array[$r]}"
  pid="$(launch_one "$host" "$r")"
  echo "${host_array[$r]} $r $pid residentd" | \
    ssh -o BatchMode=yes "$record_host" "cat >> '$record_dir/launch_pids.txt'"
  echo "$host: residentd rank $r launched pid $pid (recorded)"
done
echo "pids recorded at $record_host:$record_dir/launch_pids.txt"

# Memory watchdog + ready wait.
watchdog_trip=0
deadline=$(( $(date +%s) + ready_timeout ))
ready_count=0
declare -A ready_seen
while :; do
  now=$(date +%s)
  [[ $now -ge $deadline ]] && break
  for r in "${!host_array[@]}"; do
    host="${host_array[$r]}"; dir="$(dir_for "$host")"
    if [[ -z "${ready_seen[$r]:-}" ]]; then
      line=$(ssh -o BatchMode=yes "$host" "grep -a 'model_residentd ready' '$dir/residentd-r$r.log' 2>/dev/null | tail -1" || true)
      if [[ -n "$line" ]]; then
        ready_seen[$r]=1; ready_count=$((ready_count+1))
        echo "$host: rank $r READY: $line"
      fi
    fi
  done
  [[ $ready_count -eq ${#host_array[@]} ]] && { echo "ALL RANKS READY"; break; }
  # watchdog: per-node compute memory (sum of compute apps)
  for r in "${!host_array[@]}"; do
    host="${host_array[$r]}"
    mib=$(ssh -o BatchMode=yes "$host" \
      "nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1} END{print s+0}'" || echo 0)
    if (( mib > mem_limit_gib * 1024 )); then
      echo "WATCHDOG TRIP on $host: ${mib}MiB > ${mem_limit_gib}GiB limit - TERM own pids" >&2
      for r2 in "${!host_array[@]}"; do
        pid_r2=$(ssh -o BatchMode=yes "$record_host" "awk '\$1==\"${host_array[$r2]}\" && \$4==\"residentd\" {print \$3}' '$record_dir/launch_pids.txt' 2>/dev/null | tail -1" || true)
        [[ -n "$pid_r2" ]] && ssh -o BatchMode=yes "${host_array[$r2]}" "kill -TERM $pid_r2 2>/dev/null" || true
      done
      watchdog_trip=1
      break 2
    fi
  done
  sleep 5
done

if [[ $watchdog_trip -eq 1 ]]; then
  echo "ABORTED by memory watchdog (own pids TERMed; sibling lanes untouched)" >&2
  exit 3
fi
if [[ $ready_count -lt ${#host_array[@]} ]]; then
  echo "NOT all ranks ready after ${ready_timeout}s ($ready_count/${#host_array[@]})" >&2
  for r in "${!host_array[@]}"; do
    host="${host_array[$r]}"; dir="$(dir_for "$host")"
    echo "--- $host rank $r log tail ---" >&2
    ssh -o BatchMode=yes "$host" "tail -5 '$dir/residentd-r$r.log'" >&2 || true
  done
  exit 4
fi
echo "fleet ready; launch api with: $0 --hosts $hosts --api-only"
exit 0
