#!/usr/bin/env bash
set -u
KIT=$(cd "$(dirname "$0")" && pwd)
. "$KIT/rollover_lib.sh"
GX_RELEASE=${GX_RELEASE:?GX_RELEASE names the release the fleet runs, e.g. c2b714343}
GX_OUT=${GX_OUT:?GX_OUT is the local directory the receipts are copied to}
GX_DIR=${GX_DIR:-$HUB_WORK/gx/$GX_RELEASE}
GX_PLAN=${GX_PLAN:-}
GX_END=${GX_END:-none}
GX_POLL=${GX_POLL:-20}
[ $# -ge 1 ] || { echo "usage: GX_RELEASE=<sha> GX_OUT=<dir> $0 MODEL..." >&2; exit 2; }
row() { python3 "$KIT/../tools/lane_config.py" release-row "$1" 2>/dev/null; }
field() { local r k="$2"; r=$(row "$1") || return 1; set -- $r; case "$k" in lane) echo "$1";; api) echo "$4";; port) echo "$5";; esac; }
lane_engines() { fan "pgrep -f '^[^ ]*bin/sparkpipe_model_[r]esidentd( |\$)' | wc -l" | awk '{s+=$2} END {print s+0}'; }

gx_change() {
  local id=$1 change=$2 lane api
  lane=$(field "$id" lane); api=$(field "$id" api)
  [ "$change" = none ] && return 0
  stamp "gx $id cache $change"
  deactivate || return 1
  [ "$(lane_engines)" = 0 ] || { echo "GX-CHANGE-FAILED $id: an engine still runs after the controller stopped every model"; return 1; }
  if [ "$change" = clear ]; then
    fan_all "clear $lane" "find \$HOME/$lane/root/kvsnapshot \$HOME/$lane/root/kvcache -mindepth 1 -delete" || return 1
    hub "rm -f ~/$api/runtime/prefix_index.spi" || return 1
  fi
  activate "$id"
}

gx_step() {
  local id=$1 step=$2 port=$3 api=$4 dir=$5 mark rc last
  mark=${step/:/-}
  hub "cd $dir && rm -f $mark.rc && (setsid nohup sh -c 'python3 ../gx_gate.py run $step --state state.json --api http://127.0.0.1:$port --api-log $HUB_WORK/logs/$api.log >> run.log 2>&1; echo \$? > $mark.rc' > /dev/null 2>&1 < /dev/null &)" || return 1
  last=""
  while true; do
    sleep "$GX_POLL"
    rc=$(hub "test -s $dir/$mark.rc && echo rc=\$(cat $dir/$mark.rc); tail -n 1 $dir/run.log 2>/dev/null") || continue
    case "$rc" in
      rc=0*) return 0 ;;
      rc=*) echo "$rc" | tail -n 1; return 1 ;;
    esac
    [ "$rc" = "$last" ] || echo "$rc"
    last=$rc
  done
}

gx_model() {
  local id=$1 lane api port dir limit steps step change state skip
  lane=$(field "$id" lane) && api=$(field "$id" api) && port=$(field "$id" port) || { echo "GX-FAILED $id: no lane record"; return 1; }
  case "$lane" in /*|*..*|*" "*) echo "GX-FAILED $id: lane '$lane' is not a path under the home"; return 1 ;; */lane[0-9]*) ;; *) echo "GX-FAILED $id: lane '$lane' is not <dir>/lane<N>"; return 1 ;; esac
  case "$api:$port" in *-kv-api:[0-9]*) ;; *) echo "GX-FAILED $id: api '$api' port '$port'"; return 1 ;; esac
  dir=$GX_DIR/$id
  stamp "gx $id (lane $lane, api $api, port $port)"
  hub "test ! -e $dir/state.json" || { echo "GX-FAILED $id: $dir holds a state; a state is planned once, so move it away or set GX_DIR"; return 1; }
  hub "mkdir -p $dir" && scp -q "$KIT/../tools/gx_gate.py" rtx5090:$GX_DIR/gx_gate.py || { echo "GX-FAILED $id: the hub did not take the tool"; return 1; }
  [ "$(lane_engines)" = 0 ] || [ "$(controller)" != "idle None None" ] || { echo "GX-FAILED $id: engines run that the controller did not start; the gate needs the fleet alone"; return 1; }
  activate "$id" || { echo "GX-ACTIVATE-FAILED $id"; return 1; }
  limit=$(hub "curl -s -m 10 http://127.0.0.1:$port/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])') || { echo "GX-FAILED $id: its API names no context length"; return 1; }
  steps=$(hub "cd $dir && python3 ../gx_gate.py plan --state state.json --model $id --release $GX_RELEASE --context-limit $limit $GX_PLAN") || { echo "GX-FAILED $id: no plan: $steps"; return 1; }
  skip=""
  while read -r step change state; do
    [ "$step" = "$skip" ] && { echo "GX-STEP-SKIPPED $id $step: its park step failed"; continue; }
    gx_change "$id" "$change" < /dev/null || { echo "GX-FAILED $id: the cache change before $step failed"; break; }
    stamp "gx $id $step"
    gx_step "$id" "$step" "$port" "$api" "$dir" < /dev/null && continue
    echo "GX-STEP-FAILED $id $step"
    case "$step" in chain) break ;; park:*) skip="restore:${step#park:}" ;; esac
  done <<< "$steps"
  hub "cd $dir && python3 ../gx_gate.py verdict --state state.json --output receipt.json > verdict.txt; rc=\$?; cat verdict.txt; exit \$rc"
  state=$?
  mkdir -p "$GX_OUT" && scp -q "rtx5090:$dir/receipt.json" "$GX_OUT/$id.json" && scp -q "rtx5090:$dir/verdict.txt" "$GX_OUT/$id.txt" && scp -q "rtx5090:$dir/run.log" "$GX_OUT/$id.log"
  case "$state" in 0) echo "GX-PASS $id" ;; 2) echo "GX-INCOMPLETE $id" ;; *) echo "GX-FAIL $id" ;; esac
  return "$state"
}

failed=0
for id in "$@"; do
  gx_model "$id" || failed=1
done
if [ "$GX_END" = none ]; then deactivate || failed=1; else activate "$GX_END" || failed=1; fi
exit "$failed"
