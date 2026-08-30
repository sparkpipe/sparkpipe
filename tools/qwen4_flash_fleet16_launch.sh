#!/usr/bin/env bash
# qwen4_flash v4 LIVE 16-rank TP4xPP4 fleet launcher (S5/S6).
#
# ONE SIMULTANEOUS WAVE over all 16 ranks (the 180s transport connect
# window refuses late joiners), driven by the launch table emitted by
# tools/qwen4_flash_deploy_v4.py (rotated stages: s0=spark4-7,
# s1=spark0-3, s2=spark8-b, s3=sparkc-f).
#
# Safety model (lane rules 2026-08-28):
#   * Teardown TERMs ONLY the pids recorded at spawn time in
#     <deploy>/launch_pids.txt on the coordinator - NEVER pgrep -f
#     (shared nodes host sibling lanes' residentds).
#   * 45s pre-launch sleep (TIME_WAIT clearance).
#   * Memory watchdog per node during bring-up; past the limit it TERMs
#     this fleet's own pids (110 GiB operator ceiling).
#   * The api (one client per residentd) launches AFTER the fleet is
#     ready, on the coordinator rank (--api-only / --api-term).
#
# Usage:
#   tools/qwen4_flash_fleet16_launch.sh --table <launch_table.json> [--deploy-dir D]
#       [--pre-sleep 45] [--ready-timeout 1200] [--mem-limit-gib 104]
#       [--term | --api-only | --api-term | --status]
set -euo pipefail

table_path=""
deploy_dir=""
pre_sleep="45"
ready_timeout="1200"
mem_limit_gib="104"
mode="wave"

while [[ $# -gt 0 ]]; do
  case $1 in
    --table) table_path="$2"; shift 2 ;;
    --deploy-dir) deploy_dir="$2"; shift 2 ;;
    --pre-sleep) pre_sleep="$2"; shift 2 ;;
    --ready-timeout) ready_timeout="$2"; shift 2 ;;
    --mem-limit-gib) mem_limit_gib="$2"; shift 2 ;;
    --term) mode="term"; shift ;;
    --api-only) mode="api"; shift ;;
    --api-term) mode="api_term"; shift ;;
    --status) mode="status"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$table_path" ]] || { echo "--table required" >&2; exit 2; }

coordinator_host=$(python3 -c "import json;print(json.load(open('$table_path'))[0]['host'])")
coordinator_rank=$(python3 -c "import json;print(json.load(open('$table_path'))[0]['rank'])")
[[ -n "$deploy_dir" ]] || deploy_dir="/home/$coordinator_host/sparkdata/qwen4_flash.tp4/deploy_v4"
pid_file="$deploy_dir/launch_pids.txt"

# Each rank's runtime tree lives under ITS OWN host's home (per
# qwen4_flash_deploy_v4.py runtime_root); only pid bookkeeping and the
# api stay on the coordinator's copy.
deploy_dir_for() { echo "/home/$1/sparkdata/qwen4_flash.tp4/deploy_v4"; }

rows() { python3 -c "
import json
for e in json.load(open('$table_path')):
    print(e['rank'],e['host'],e['chain_position'],e['pp_stage'],e['tp_rank'],e['rail_ip'],e['tp_hosts'])
"; }

record_pid() {  # host rank pid kind
  ssh -o BatchMode=yes "$coordinator_host" "echo '$1 $2 $3 $4' >> '$pid_file'"
}

term_recorded() {
  local kind_filter="${1:-}"
  while read -r host rank pid kind; do
    [[ -n "$host" && -n "$pid" ]] || continue
    if [[ -n "$kind_filter" && "$kind" != "$kind_filter" ]]; then continue; fi
    echo "TERM $kind pid $pid on $host (recorded)"
    ssh -o BatchMode=yes "$host" "kill -TERM $pid 2>/dev/null" || true
  done < <(ssh -o BatchMode=yes "$coordinator_host" "cat '$pid_file' 2>/dev/null" || true)
  local waited=0
  while (( waited < 30 )); do
    local alive=0
    while read -r host rank pid kind; do
      [[ -n "$kind_filter" && "$kind" != "$kind_filter" ]] && continue
      [[ -n "$pid" ]] && ssh -o BatchMode=yes "$host" "kill -0 $pid 2>/dev/null" && alive=1 || true
    done < <(ssh -o BatchMode=yes "$coordinator_host" "cat '$pid_file' 2>/dev/null" || true)
    (( alive == 0 )) && break
    sleep 1; waited=$((waited+1))
  done
}

case "$mode" in
  term)
    term_recorded
    ssh -o BatchMode=yes "$coordinator_host" "rm -f '$pid_file'"
    echo "teardown complete (TERM only; no KILL per protocol)"
    exit 0 ;;
  api_term)
    term_recorded api
    echo "api TERMed"
    exit 0 ;;
esac

echo "coordinator: $coordinator_host rank $coordinator_rank deploy $deploy_dir"

if [[ "$mode" == "api" ]]; then
  pid=$(ssh -o BatchMode=yes "$coordinator_host" bash -s <<REMOTE
set -euo pipefail
cd "$deploy_dir"
nohup "\$PWD/bin/sparkpipe_model_api" --deployment "\$PWD" --rank-index $coordinator_rank > "\$PWD/api.log" 2>&1 < /dev/null &
echo \$!
REMOTE
)
  record_pid "$coordinator_host" "$coordinator_rank" "$pid" api
  echo "$coordinator_host: api launched pid $pid (recorded)"
  exit 0
fi

if [[ "$mode" == "status" ]]; then
  while read -r host rank pid kind; do
    state=$(ssh -o BatchMode=yes "$host" "kill -0 $pid 2>/dev/null && echo alive || echo dead")
    ready=$(ssh -o BatchMode=yes "$host" "grep -ac 'model_residentd ready' '$(deploy_dir_for "$host")/residentd-r$rank.log' 2>/dev/null" || echo 0)
    echo "$host rank $rank $kind pid $pid $state ready_lines=$ready"
  done < <(ssh -o BatchMode=yes "$coordinator_host" "cat '$pid_file' 2>/dev/null" || true)
  exit 0
fi

# ---- fleet wave ----
# TERM any previous wave (own recorded pids only), then clear the file.
term_recorded
ssh -o BatchMode=yes "$coordinator_host" "rm -f '$pid_file'"

# Verify every rank's pack is present before spawning anything (a wave
# that dies mid-flight on a missing pack wastes the whole 180s window).
while read -r rank host pos pp tp rail tp_hosts; do
  pack=$(python3 -c "import json;print(json.load(open('$table_path'))[$(python3 -c "import json;t=json.load(open('$table_path'));print([i for i,e in enumerate(t) if e['rank']==$rank][0])")]['pack'])")
  ssh -o BatchMode=yes "$host" "[[ -s '$(deploy_dir_for "$host")/$pack' ]]" || { echo "PACK MISSING on $host: $pack" >&2; exit 5; }
done < <(rows)
echo "all 16 packs present"

echo "pre-launch sleep ${pre_sleep}s (TIME_WAIT clearance)..."
sleep "$pre_sleep"

# ONE WAVE: spawn every rank within the same instant; each pid is
# recorded the moment it is captured.
launch_one() {  # rank host pos pp tp rail tp_hosts
  local rank="$1" host="$2" pos="$3" pp="$4" tp="$5" rail="$6" tp_hosts="$7"
  local identifier="$8"
  local pid
  pid=$(ssh -o BatchMode=yes "$host" bash -s <<REMOTE
set -euo pipefail
dir="$(deploy_dir_for "$host")"
mkdir -p "\$dir/runtime-$rank"
cd "\$dir"
export SPARK_QWEN4_FLASH_TP_DEGREE=4
export SPARK_QWEN4_FLASH_TP_RANK=$tp
export SPARK_QWEN4_FLASH_STAGE_TP_BACKEND_PATH="\$dir/lib/libhidden_transport_spark_host_rdma_verbs.so"
export SPARK_QWEN4_FLASH_STAGE_TP_IDENTIFIER=$identifier
export SPARK_QWEN4_FLASH_STAGE_TP_PORT_BASE=66840
export SPARK_QWEN4_FLASH_STAGE_TP_HOSTS="$tp_hosts"
export SPARK_QWEN4_FLASH_STAGE_TP_LOCAL_HOST="$rail"
export SPARK_QWEN4_FLASH_STAGE_TP_TIMEOUT_MS=180000
export LD_LIBRARY_PATH="\$dir/lib:\${LD_LIBRARY_PATH:-}"
nohup "\$dir/bin/sparkpipe_model_residentd" --deployment "\$dir" --rank-index "$rank" > "\$dir/residentd-r$rank.log" 2>&1 < /dev/null &
echo \$!
REMOTE
)
  record_pid "$host" "$rank" "$pid" residentd
  echo "$host: rank $rank (chain $pos, stage $pp, tp $tp) residentd pid $pid (recorded)"
}

identifier="$(date -u +%s)"
while read -r rank host pos pp tp rail tp_hosts; do
  launch_one "$rank" "$host" "$pos" "$pp" "$tp" "$rail" "$tp_hosts" "$identifier"
done < <(rows)
echo "wave launched (identifier $identifier); pids at $coordinator_host:$pid_file"

# Ready-wait + memory watchdog.
declare -A ready_seen
deadline=$(( $(date +%s) + ready_timeout ))
ready_count=0
watchdog_trip=0
while :; do
  (( $(date +%s) >= deadline )) && break
  while read -r rank host pos pp tp rail tp_hosts; do
    line=$(ssh -o BatchMode=yes "$host" "grep -a 'model_residentd ready' '$(deploy_dir_for "$host")/residentd-r$rank.log' 2>/dev/null | tail -1" || true)
    if [[ -n "$line" ]]; then
      if [[ -z "${ready_seen[$rank]:-}" ]]; then ready_seen[$rank]=1; ready_count=$((ready_count+1)); echo "$host rank $rank READY: $line"; fi
    fi
  done < <(rows)
  (( ready_count >= 16 )) && { echo "ALL 16 RANKS READY"; break; }
  for host in $(python3 -c "import json;print(' '.join(sorted(set(e['host'] for e in json.load(open('$table_path'))))))"); do
    mib=$(ssh -o BatchMode=yes "$host" "nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1} END{print s+0}'" || echo 0)
    if (( mib > mem_limit_gib * 1024 )); then
      echo "WATCHDOG TRIP on $host: ${mib}MiB > ${mem_limit_gib}GiB - TERM own fleet" >&2
      term_recorded
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
if (( ready_count < 16 )); then
  echo "NOT all ranks ready after ${ready_timeout}s ($ready_count/16); tails:" >&2
  while read -r rank host pos pp tp rail tp_hosts; do
    echo "--- $host rank $rank ---" >&2
    ssh -o BatchMode=yes "$host" "tail -4 '$(deploy_dir_for "$host")/residentd-r$rank.log'" >&2 || true
  done < <(rows)
  exit 4
fi
echo "fleet ready; api: $0 --table $table_path --api-only"
