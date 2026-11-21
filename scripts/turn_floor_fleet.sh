#!/usr/bin/env bash
set -u
[ -d "$HOME/bin/sparkpipe-kit-ssh" ] && PATH=$HOME/bin/sparkpipe-kit-ssh:$PATH
KIT=$(cd "$(dirname "$0")" && pwd)
. "$KIT/rollover_lib.sh"
TF_RUN=${TF_RUN:?TF_RUN names this run, for example tf-20261122}
TF_OUT=${TF_OUT:?TF_OUT is the local directory the receipts are copied to}
TF_TOKENS=${TF_TOKENS:?TF_TOKENS is the hub directory that holds tokens-<family>.txt}
TF_DIR=${TF_DIR:-$HUB_WORK/turn-floor/$TF_RUN}
TF_TAILS=${TF_TAILS:-1,40,500,5000}
TF_REPEATS=${TF_REPEATS:-3}
TF_ONCE_FROM=${TF_ONCE_FROM:-2000}
TF_DECODE=${TF_DECODE:-96}
TF_SERVED=${TF_SERVED:-128}
TF_WAVE=${TF_WAVE:-20000}
TF_POLL=${TF_POLL:-15}
TF_DRY=${TF_DRY:-0}
[ $# -ge 1 ] || { echo "usage: TF_RUN=<name> TF_OUT=<dir> TF_TOKENS=<hub dir> $0 MODEL=FAMILY=LEVELS[=ARMS] ..." >&2; exit 2; }
row() { python3 "$KIT/../tools/lane_config.py" release-row "$1" 2>/dev/null; }
field() { local r k="$2"; r=$(row "$1") || return 1; set -- $r; case "$k" in lane) echo "$1";; api) echo "$4";; port) echo "$5";; esac; }
dry() { [ "$TF_DRY" = 1 ]; }

type activate > /dev/null 2>&1 || activate() {
  local model="$1" i out
  swap "-X POST localhost:8440/activate -d '{\"model\":\"$model\"}'" > /dev/null || { echo "activate $model: the swap controller refused"; return 2; }
  for i in $(seq 1 120); do
    sleep 5
    out=$(controller 2>/dev/null) || continue
    case "$out" in "idle $model None") echo "activate $model: $out"; return 0;; failed*) echo "activate $model: $out"; return 2;; idle*) echo "activate $model: $out"; return 3;; esac
  done
  echo "activate $model: timeout"; return 1
}

type deactivate > /dev/null 2>&1 || deactivate() {
  local i out
  swap "-X POST localhost:8440/deactivate" > /dev/null || { echo "deactivate: the swap controller refused"; return 2; }
  for i in $(seq 1 120); do
    sleep 5
    out=$(controller 2>/dev/null) || continue
    case "$out" in "idle None None") echo "deactivate: $out"; return 0;; failed*) echo "deactivate: $out"; return 2;; esac
  done
  echo "deactivate: timeout"; return 1
}

tf_on() {
  dry && { echo "  controller: activate $1 unless it is active"; return 0; }
  [ "$(controller 2>/dev/null)" = "idle $1 None" ] || activate "$1"
}

tf_remote() {
  local dir=$1 mark=$2 command=$3 rc last=""
  dry && { echo "  hub: cd $dir && $command"; return 0; }
  hub "cd $dir && rm -f $mark.rc && (setsid nohup sh -c '$command >> run.log 2>&1; echo \$? > $mark.rc' > /dev/null 2>&1 &)" || return 1
  while true; do
    sleep "$TF_POLL"
    rc=$(hub "test -s $dir/$mark.rc && echo rc=\$(cat $dir/$mark.rc); tail -n 1 $dir/run.log 2>/dev/null") || continue
    case "$rc" in rc=0*) return 0;; rc=*) echo "$rc" | tail -n 1 | cut -c1-240; return 1;; esac
    [ "$rc" = "$last" ] || echo "  $rc" | cut -c1-240
    last=$rc
  done
}

tf_done() { dry && return 1; hub "grep -qx $2 $1/steps.done 2>/dev/null"; }
tf_mark() { dry || hub "echo $2 >> $1/steps.done"; }

tf_step() {
  local id=$1 dir=$2 step=$3 command=$4
  tf_done "$dir" "$step" && { echo "tf $id $step: done before"; return 0; }
  stamp "tf $id $step"
  tf_remote "$dir" "$step" "$command" || { echo "TF-FAILED $id $step"; return 1; }
  tf_mark "$dir" "$step"
}

tf_model() {
  local spec=$1 id family levels arms lane api port dir run arm
  IFS== read -r id family levels arms <<< "$spec"
  arms=${arms:-turns,wave,restart}
  [ -n "$id" ] && [ -n "$family" ] && [ -n "$levels" ] || { echo "TF-FAILED $spec: a model is MODEL=FAMILY=LEVELS[=ARMS]"; return 1; }
  lane=$(field "$id" lane) && api=$(field "$id" api) && port=$(field "$id" port) || { echo "TF-FAILED $id: no lane record"; return 1; }
  case "$api:$port" in *-kv-api:[0-9]*) ;; *) echo "TF-FAILED $id: api '$api' port '$port'"; return 1 ;; esac
  for arm in ${arms//,/ }; do
    case "$arm" in turns|wave|openings|restart) ;; *) echo "TF-FAILED $id: unknown arm '$arm'; the arms are turns, wave, openings and restart"; return 1 ;; esac
  done
  dir=$TF_DIR/$id
  run="python3 ../turn_floor.py run --endpoint http://127.0.0.1:$port --tokens $TF_TOKENS/tokens-$family.txt --api-log $HUB_WORK/logs/$api.log --label $id --output . --contexts $levels --tails $TF_TAILS --repeats $TF_REPEATS --spec-repeats 0 --once-from $TF_ONCE_FROM --decode-tokens $TF_DECODE --served-tokens $TF_SERVED --wave-tokens $TF_WAVE"
  stamp "tf $id (lane $lane, api $api, port $port, levels $levels, arms $arms)"
  if ! dry; then
    hub "mkdir -p $dir && test -s $TF_TOKENS/tokens-$family.txt" || { echo "TF-FAILED $id: the hub has no $TF_TOKENS/tokens-$family.txt"; return 1; }
    sparkcp "$KIT/../tools/turn_floor.py" "hub:$TF_DIR/" > /dev/null || { echo "TF-FAILED $id: the hub did not take the tool"; return 1; }
  fi
  tf_on "$id" || { echo "TF-ACTIVATE-FAILED $id"; return 1; }
  for arm in ${arms//,/ }; do
    if [ "$arm" = restart ]; then
      if ! tf_done "$dir" restart-stop; then
        stamp "tf $id restart: stop and start"
        if dry; then echo "  controller: deactivate, then activate $id"; else deactivate && activate "$id" || { echo "TF-FAILED $id restart"; return 1; }; fi
        tf_mark "$dir" restart-stop
      fi
      tf_on "$id" || { echo "TF-ACTIVATE-FAILED $id"; return 1; }
      tf_step "$id" "$dir" restart "$run --arms resume --event restart" || return 1
    else
      tf_step "$id" "$dir" "$arm" "$run --arms $arm" || return 1
    fi
  done
  stamp "tf $id parts"
  tf_remote "$dir" parts "python3 ../turn_floor.py parts ." || { echo "TF-FAILED $id parts"; return 1; }
  tf_remote "$dir" excerpt "python3 ../turn_floor.py excerpt . --output engine.log" || { echo "TF-FAILED $id excerpt"; return 1; }
  dry && { echo "  copy: hub:$dir/{requests.jsonl,parts.jsonl,summary.json,engine.log,run.log} to $TF_OUT/$id/"; return 0; }
  mkdir -p "$TF_OUT/$id" && sparkcp "hub:$dir/requests.jsonl" "hub:$dir/parts.jsonl" "hub:$dir/summary.json" "hub:$dir/engine.log" "hub:$dir/run.log" "$TF_OUT/$id/" > /dev/null ||
    { echo "TF-FAILED $id: the receipts did not copy"; return 1; }
  echo "TF-DONE $id: receipts in $TF_OUT/$id"
}

failed=0
for spec in "$@"; do
  tf_model "$spec" || { failed=1; break; }
done
exit $failed
