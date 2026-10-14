#!/usr/bin/env bash
HEX=0123456789abcdef
HUB_WORK=${HUB_WORK:-/srv/workspace/sparkpipe}
hub() { ssh -o BatchMode=yes -o ConnectTimeout=8 rtx5090 "$@" < /dev/null; }
node() { ssh -o BatchMode=yes -o ConnectTimeout=8 "spark${HEX:$1:1}" "$2" < /dev/null; }
fan() { local r; for r in $(seq 0 15); do node "$r" "$1" 2>&1 | awk -v host="spark${HEX:$r:1}: " '{ print host $0; fflush() }' & done; wait; }
fan_all() {
  local out good
  out=$(fan "{ $2 ; } && echo FAN-OK")
  good=$(printf '%s\n' "$out" | grep -c ': FAN-OK$')
  [ "$good" = 16 ] && return 0
  echo "$1 FAILED on $((16 - good)) nodes: $(printf '%s\n' "$out" | grep -v ': FAN-OK$' | cut -d: -f1 | sort -u | tr '\n' ' ')"
  printf '%s\n' "$out" | grep -v ': FAN-OK$' | head -8
  return 1
}
nodes_up() {
  local down="" r
  for r in $(seq 0 15); do node "$r" true > /dev/null 2>&1 || down="$down spark${HEX:$r:1}"; done
  [ -z "$down" ] || { echo "NODE-DOWN:$down"; return 1; }
}
stamp() { echo "== $(date -u +%FT%TZ) $*"; }
hub_space() {
  local free
  free=$(hub "mkdir -p $HUB_WORK && df -Pk $HUB_WORK \$HOME | awk 'NR > 1 {print int(\$4 / 1048576)}'" | tr '\n' ' ')
  set -- $free
  [ "${1:-0}" -ge 20 ] && [ "${2:-0}" -ge 2 ] && return 0
  echo "HUB-SPACE-LOW: $HUB_WORK has ${1:-?} GiB free (needs 20), the hub home has ${2:-?} GiB free (needs 2)"
  return 1
}
controller() { hub "curl -s localhost:8440/status" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state'], d['active'], d['error'])"; }
restore_max() {
  activate qwen3.8-max && return 0
  sleep 30; activate qwen3.8-max && return 0
  echo "RESTORE FAILED: qwen3.8-max is not active"; return 1
}
