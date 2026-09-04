#!/usr/bin/env bash
# Deploy GLM52 TP8 rank packs + stage configs from the pack build host to
# each rank's local NVMe. Parameterized end to end (lane rule: no hardcoded
# nodes): hosts, build host, build dir, and runtime root are all arguments.
#
# Usage:
#   tools/glm52_deploy_packs.sh \
#     --build-host spark0 \
#     --build-dir /home/spark0/glm52_packbuild/rank_packs \
#     --hosts spark8,spark9,sparka,sparkb,sparkc,sparkd,sparke,sparkf \
#     --runtime-root '/home/{host}/sparkdata/glm52.tp8.fp8' \
#     --config-dir /tmp/glm52-deploy
#
# For rank r on host h this deploys:
#   <build-dir>/glm52_tp8_rank0r.fp8.glms52sp -> h:<runtime-root>/packs/
#   <config-dir>/<h>/config/*.json            -> h:<runtime-root>/config/
# Transfers run host-to-host (build host pushes over the fabric); at most
# --parallel transfers run at once (node I/O cap: 2).
set -euo pipefail

BUILD_HOST=""
BUILD_DIR=""
HOSTS=""
RUNTIME_ROOT='/home/{host}/sparkdata/glm52.tp8.fp8'
CONFIG_DIR=""
PARALLEL=2
PACK_TEMPLATE='glm52_tp8_rank{rank:02d}.fp8.glms52sp'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-host) BUILD_HOST="$2"; shift 2;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --hosts) HOSTS="$2"; shift 2;;
    --runtime-root) RUNTIME_ROOT="$2"; shift 2;;
    --config-dir) CONFIG_DIR="$2"; shift 2;;
    --parallel) PARALLEL="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

[[ -n "$BUILD_HOST" && -n "$BUILD_DIR" && -n "$HOSTS" ]] || {
  echo "required: --build-host, --build-dir, --hosts" >&2; exit 2; }

IFS=',' read -r -a HOST_ARRAY <<< "$HOSTS"
FAILED=0

transfer() {
  local rank="$1" host="$2"
  local pack
  pack=$(python3 -c "print('$PACK_TEMPLATE'.format(rank=$rank))")
  local root="${RUNTIME_ROOT//\{host\}/$host}"
  echo "[$(date +%H:%M:%S)] rank $rank -> $host:$root/packs/$pack"
  if ! ssh -o BatchMode=yes "$BUILD_HOST" \
      "rsync -a --inplace '$BUILD_DIR/$pack' '$host:$root/packs/'"; then
    echo "FAIL rank $rank -> $host" >&2
    return 1
  fi
  ssh -o BatchMode=yes "$host" "test -r '$root/packs/$pack'" || {
    echo "FAIL readable check $host:$root/packs/$pack" >&2; return 1; }
  if [[ -n "$CONFIG_DIR" && -d "$CONFIG_DIR/$host/config" ]]; then
    ssh -o BatchMode=yes "$host" "mkdir -p '$root/config'"
    scp -q "$CONFIG_DIR/$host/config/"*.json "$host:$root/config/" || {
      echo "FAIL config copy $host" >&2; return 1; }
  fi
  echo "[$(date +%H:%M:%S)] rank $rank -> $host OK"
}

RUNNING=0
for rank in "${!HOST_ARRAY[@]}"; do
  host="${HOST_ARRAY[$rank]}"
  transfer "$rank" "$host" &
  RUNNING=$((RUNNING + 1))
  # bash 3.2 has no `wait -n`: poll for any finished child instead
  while [[ "$RUNNING" -ge "$PARALLEL" ]]; do
    RUNNING=$PARALLEL
    for job in $(jobs -p); do
      kill -0 "$job" 2>/dev/null || RUNNING=$((RUNNING - 1))
    done
    [[ "$RUNNING" -ge "$PARALLEL" ]] && sleep 2
  done
done
wait || FAILED=1

if [[ "$FAILED" -ne 0 ]]; then
  echo "DEPLOY: FAIL (some transfers failed)" >&2
  exit 1
fi
echo "DEPLOY: OK ($(( ${#HOST_ARRAY[@]} )) ranks)"
