#!/usr/bin/env bash
V=2c5d5b843
W=/Users/mac/wf/drv-rollover
FW=/Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-$V
APIB=/home/spec/api-build-$V/build
D=${ROLLOVER_LOGS:-/private/tmp/claude-502/-Users-mac-sparkpipe/df54bdc1-84bf-4584-9653-5211cf17258e/scratchpad/rollover}
HEX=0123456789abcdef
hub() { ssh -o BatchMode=yes -o ConnectTimeout=8 rtx5090 "$@" < /dev/null; }
node() { ssh -o BatchMode=yes -o ConnectTimeout=8 "spark${HEX:$1:1}" "$2" < /dev/null; }
fan() { local r; for r in $(seq 0 15); do node "$r" "$1" 2>&1 | sed "s/^/spark${HEX:$r:1}: /" & done; wait; }
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
controller() { hub "curl -s localhost:8440/status" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state'], d['active'], d['error'])"; }
activate() {
  local model="$1" i out
  hub "curl -s -X POST localhost:8440/activate -d '{\"model\":\"$model\"}'" > /dev/null
  for i in $(seq 1 120); do
    sleep 5
    out="$(hub "curl -s localhost:8440/status" | python3 -c "
import json, sys
d = json.load(sys.stdin)
h = (d.get('history') or [{}])[-1]
print(d['state'], d['active'], h.get('seconds'), json.dumps(d['error']))" 2>/dev/null)" || continue
    set -- $out
    [ "${1:-}" = idle ] || continue
    echo "activate $model: $out"
    [ "$2" = "$model" ] && [ "$4" = null ] && return 0
    return 3
  done
  echo "activate $model: timeout"; return 1
}
restore_max() {
  activate qwen3.8-max && return 0
  sleep 30; activate qwen3.8-max && return 0
  echo "RESTORE FAILED: qwen3.8-max is not active"; return 1
}
fw_sha() { shasum -a 256 "$FW/$1" | cut -c1-16; }
