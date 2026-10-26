#!/usr/bin/env bash
set -u
KIT=$(cd "$(dirname "$0")" && pwd)
. "$KIT/rollover_lib.sh"
RELEASE_FW=${FW:?FW is the local release firmware dir (release_build.sh output)}
RELEASE_APIB=${APIB:?APIB is the hub api build dir, e.g. $HUB_WORK/api-build-<sha>/build}
activate() {
  local model="$1" i out
  swap "-X POST localhost:8440/activate -d '{\"model\":\"$model\"}'" > /dev/null || { echo "activate $model: the swap controller refused"; return 2; }
  for i in $(seq 1 120); do
    sleep 5
    out="$(swap localhost:8440/status | python3 -c "
import json, sys
d = json.load(sys.stdin)
h = (d.get('history') or [{}])[-1]
print(d['state'], d['active'], h.get('seconds'), json.dumps(d['error']))" 2>/dev/null)" || continue
    set -- $out
    [ "${1:-}" = failed ] && { echo "activate $model: $out"; return 2; }
    [ "${1:-}" = idle ] || continue
    echo "activate $model: $out"
    [ "$2" = "$model" ] && [ "$4" = null ] && return 0
    return 3
  done
  echo "activate $model: timeout"; return 1
}
FW=$RELEASE_FW
APIB=$RELEASE_APIB
SHA=$(cat "$FW/SOURCE_COMMIT")
STATE=${STATE:-$KIT/state-$SHA}
mkdir -p "$STATE"
R="release-$SHA"
RD=$HUB_WORK/$R
BK=$HUB_WORK/backups
OLD_WD=${OLD_WD:-60b3ca5d0d3e444b}
NEW_WD=$(shasum -a 256 "$FW/sparkpipe_weightd" | cut -c1-16)
MODELS="qwen3.8-max glm-5.3 kimi-k3 glm-5.3-flash qwen3.8-27b qwen3.8-flash deepseek-v4.1-flash gemma-4-31b gemma-4-26b-a4b ling-3.0-flash ling-3.0-flash-fin hy4-preview muse-glimmer-30b laguna-s-2.1 mimo-v2.6-flash mimo-v2.6-pro k-exaone-2.0 a.x-k2"
row() { python3 "$KIT/../tools/lane_config.py" release-row "$1" 2>/dev/null; }
python3 -c "import json,sys; sys.exit(sorted(sys.argv[2].split()) != sorted(json.load(open(sys.argv[1]))['lanes']))" "$KIT/../deployment/lanes.json" "$MODELS" || { echo "MODELS differ from the records in deployment/lanes.json: refusing to run" >&2; exit 1; }
field() { local r k="$2"; r=$(row "$1") || return 1; set -- $r; case "$k" in lane) echo "$1";; fwlib) echo "$2";; lib) echo "$3";; api) echo "$4";; port) echo "$5";; label) echo "$6";; esac; }
for id in $MODELS; do for k in lane fwlib lib api port label; do v=$(field "$id" "$k"); case "$k:$v" in *:|*" "*|lane:/*|lane:*..*|[!l]*:*/*|l[!a]*:*/*) echo "field $k of $id is '$v': refusing to run" >&2; exit 1;; esac; done; case "$(field "$id" lane)" in */lane[0-9]*) ;; *) echo "lane of $id is not <dir>/lane<N>: refusing to run" >&2; exit 1;; esac; done
utc() { [ -f "$STATE/utc" ] || date -u +%Y%m%dT%H%M%SZ > "$STATE/utc"; cat "$STATE/utc"; }
backup_utc() { [ -s "$STATE/utc" ] || { echo "no backup stamp in $STATE/utc: run the backup step first" >&2; return 1; }; cat "$STATE/utc"; }
backups_present() {
  local U="$1" id L missing=0
  for id in $MODELS; do
    L=$(field $id lane)
    fan_all "backup $L/root.prev-$U" "test -d \$HOME/$L/root.prev-$U" || missing=1
    hub "test -d $BK/$(field $id api).prev-$U" || { echo "backup hub $BK/$(field $id api).prev-$U missing"; missing=1; }
  done
  fan_all "backup weightd_warm.prev-$U" "test -s \$HOME/sparkdata/swap/weightd_warm.prev-$U" || missing=1
  hub "test \"\$(sha256sum < ~/release-staging/weightd-rollback/sparkpipe_weightd.$OLD_WD | cut -c1-16)\" = $OLD_WD" || { echo "backup hub weightd $OLD_WD missing"; missing=1; }
  return $missing
}
one_variant() {
  local out
  out=$(cat)
  printf '%s\n' "$out" | sed "s/^/$1: /"
  [ "$(printf '%s\n' "$out" | grep -c .)" = 1 ] && printf '%s\n' "$out" | grep -q '^ *16 ' && return 0
  echo "$1: FAILED, expected one result on all 16 nodes"; return 1
}
restore_root() { echo "test -d root.prev-$1 && rm -rf root.failed-$1 && mv root root.failed-$1 && { cp -a root.prev-$1 root || { rm -rf root; mv root.failed-$1 root; false; }; }"; }
residentds() { fan "pgrep -f '^[^ ]*bin/sparkpipe_model_[r]esidentd( |\$)' | wc -l" | awk '{s+=$2} END {print s+0}'; }
running_wd() { fan "sha256sum < /proc/\$(pgrep -o -f 'sparkdata/weightd/sparkpipe_[w]eightd')/exe | cut -c1-16; test -e /tmp/weightd-mesh/.ready && echo ready" | awk '{c[$2]++} END {for (k in c) printf "%s=%d ", k, c[k]}'; }
wd_is() { local s; s=$(running_wd); echo "$s" | grep -q "$1=16" && echo "$s" | grep -q "ready=16"; }
swap_cmd() { hub "python3 -c \"import json,sys; d=json.load(open('/home/spec/sparkpipe-swap/model_swap.json')); print([m for m in d['models'] if m['id']=='$1'][0]['$2'])\""; }

capture() {
  local id=$1 tag=$2 port; port=$(field "$id" port)
  hub "mkdir -p $RD/$id && cd $RD && timeout 3600 python3 kit/release_capture.py $port $id/$tag.json kit/prompt_source.c > $id/$tag.log 2>&1; echo capture $id $tag rc=\$?"
  case "$id" in
    glm-5.3) hub "cd ~/rollover/tree && timeout 2400 python3 tools/glm53_full_golden.py capture --prompts qualification/glm53_full/golden/prompts.json --port $port --max-tokens 128 --logprobs 4 --label $R-$tag --out $RD/$id/golden-$tag.json > $RD/$id/golden-$tag.log 2>&1; echo golden rc=\$?" ;;
    kimi-k3) hub "cd ~/rollover/tree && timeout 1800 python3 $RD/kit/rollover_k3_golden.py $port $RD/$id/golden-$tag.json > $RD/$id/golden-$tag.log 2>&1; echo golden rc=\$?" ;;
    qwen3.8-27b|qwen3.8-flash|qwen3.8-max) hub "cd ~/rollover/tree && timeout 1800 python3 $RD/kit/rollover_lp_capture.py qualification/qwen38_27b/reference/reference3_fp8_numpy.json $port $RD/$id/golden-$tag.json > /dev/null 2>&1; echo golden rc=\$?" ;;
    deepseek-v4.1-flash) hub "cd ~/rollover/tree && timeout 1800 python3 $RD/kit/rollover_lp_capture.py qualification/dsv41_flash/reference_tokens.json $port $RD/$id/golden-$tag.json > /dev/null 2>&1; echo golden rc=\$?" ;;
  esac
}
compare() {
  local id=$1 a=$2 b=$3
  hub "cd $RD && python3 kit/release_compare.py $id/$a.json $id/$b.json | tail -7"
  hub "test -f $RD/$id/golden-$a.json && test -f $RD/$id/golden-$b.json && { python3 ~/perfoverlap/exact.py $RD/$id/golden-$a.json $RD/$id/golden-$b.json | head -3; cd ~/rollover/tree && python3 $RD/kit/rollover_golden_compare.py $RD/$id/golden-$a.json $RD/$id/golden-$b.json 0.5 $RD/$id/golden-compare-$a-$b.json 2>&1 | tail -2; } || true"
}

step=${1:-}
case "$step" in
  precheck)
    stamp "precheck release $SHA (weightd $OLD_WD -> $NEW_WD)"
    [ "$(controller)" = "idle qwen3.8-max None" ] || { echo "PRECHECK-FAILED controller $(controller)"; exit 1; }
    nodes_up || exit 9
    for f in sparkpipe_model_residentd weightd_warm libhidden_transport_spark_host_rdma_verbs.so sparkpipe_weightd sparkpipe_model_batch weightd_spine_budget.py; do [ -s "$FW/$f" ] || { echo "MISSING $FW/$f"; exit 1; }; done
    for id in $MODELS; do [ -s "$FW/$(field $id fwlib)" ] || { echo "MISSING $FW/$(field $id fwlib)"; exit 1; }; hub "test -s $APIB/$(field $id fwlib)" || { echo "MISSING hub $APIB/$(field $id fwlib)"; exit 1; }; done
    hub "test -x $APIB/sparkpipe_model_api && test -x $APIB/sparkpipe_tokenize_prompt && cat ~/release/core/WEIGHTSD_BIN" | grep -q "$OLD_WD" || { echo "hub core weightd is not $OLD_WD"; exit 1; }
    echo "weightd running: $(running_wd)"
    fan "ls -d \$HOME/sparkdata/glm53full.fp8lin.tp16/packs/*.glm52sp | wc -l" | awk '{print $2}' | sort | uniq -c | sed 's/^/fp8lin packs per node: /'
    hub_space || exit 1
    hub "mkdir -p $RD/kit $BK" && scp -q "$KIT/../tools/release_capture.py" "$KIT/../tools/release_compare.py" "$KIT/../tools/rollover_golden_compare.py" "$KIT/../tools/rollover_k3_golden.py" "$KIT/../tools/rollover_lp_capture.py" rtx5090:$RD/kit/ && scp -q "$KIT/../runtime/json.c" rtx5090:$RD/kit/prompt_source.c
    echo PRECHECK-OK ;;
  old)
    for id in $MODELS; do stamp "old $id"; activate "$id" && capture "$id" old || echo "OLD-FAILED $id"; done
    restore_max ;;
  stage)
    stamp "stage $FW on 16 nodes (~/rp-release-$SHA) and check hashes"
    (cd "$FW" && COPYFILE_DISABLE=1 tar cf "$STATE/fw.tar" .)
    for r in $(seq 0 15); do (scp -q -o BatchMode=yes "$STATE/fw.tar" spark${HEX:$r:1}:rp-release-$SHA.tar && node $r "rm -rf ~/rp-release-$SHA && mkdir ~/rp-release-$SHA && tar -C ~/rp-release-$SHA -xf ~/rp-release-$SHA.tar 2>/dev/null && rm ~/rp-release-$SHA.tar && cd ~/rp-release-$SHA && sha256sum --quiet -c SHA256SUMS && echo verified" | sed "s/^/spark${HEX:$r:1}: /") & done | awk '{print $NF}' | sort | uniq -c | one_variant stage || { echo "STAGE-FAILED"; exit 1; } ;;
  backup)
    hub_space || exit 1
    U=$(utc); stamp "backup root.prev-$U"
    for id in $MODELS; do
      L=$(field $id lane)
      fan_all "backup $L" "cd \$HOME/$L && [ ! -e root.prev-$U ] && mkdir root.prev-$U && rsync -a '--exclude=/kvsnapshot/*' '--exclude=/kvcache/*' root/ root.prev-$U/" || exit 1
      hub "[ -e $BK/$(field $id api).prev-$U ] || cp -a ~/$(field $id api) $BK/$(field $id api).prev-$U" || { echo "BACKUP-FAILED hub $(field $id api)"; exit 1; }
    done
    fan_all "backup weightd" "cp -p \$HOME/sparkdata/swap/weightd_warm \$HOME/sparkdata/swap/weightd_warm.prev-$U && cp -p \$HOME/sparkdata/weightd/sparkpipe_weightd \$HOME/sparkdata/weightd/sparkpipe_weightd.prev-$U" || exit 1
    hub "mkdir -p ~/release-staging/weightd-rollback && r=~/release-staging/weightd-rollback/sparkpipe_weightd.$OLD_WD && { [ -f \$r ] || cp ~/release/core/bin/sparkpipe_weightd \$r; } && test \"\$(sha256sum < \$r | cut -c1-16)\" = $OLD_WD && cp -p ~/sparkpipe-swap/model_swap.json ~/sparkpipe-swap/model_swap.json.prev-$U && echo hub backups ok" || { echo "BACKUP-FAILED hub"; exit 1; } ;;
  down)
    stamp "down: stop the active model, no residentd anywhere"
    for id in $MODELS; do
      if hub "curl -s -m 3 http://127.0.0.1:$(field $id port)/health | grep -q connected_ranks"; then
        stamp "stopping $id"; hub "$(swap_cmd $id api_stop)"; fan "$(swap_cmd $id node_stop) > /dev/null 2>&1; echo \$?" | awk '{print $2}' | sort | uniq -c
      fi
    done
    n=$(residentds); echo "residentds left: $n"; [ "$n" = 0 ] || { echo "DOWN-FAILED"; exit 1; } ;;
  weightd)
    n=$(residentds); [ "$n" = 0 ] || { echo "residentds still running: $n"; exit 1; }
    stamp "publish + announce weightd $NEW_WD"
    scp -q "$FW/sparkpipe_weightd" rtx5090:release/core/bin/sparkpipe_weightd.new || exit 1
    hub "cd ~/release/core && exec 9>\$HOME/release/.core.publish.lock && flock 9 && test \"\$(sha256sum < bin/sparkpipe_weightd.new | cut -c1-16)\" = $NEW_WD && chmod 755 bin/sparkpipe_weightd.new && mv bin/sparkpipe_weightd.new bin/sparkpipe_weightd && find bin -type f | sort | xargs sha256sum > MANIFEST.tmp && mv MANIFEST.tmp MANIFEST && printf '%s\n' $NEW_WD > WEIGHTSD_BIN.tmp && mv WEIGHTSD_BIN.tmp WEIGHTSD_BIN && echo announced \$(cat WEIGHTSD_BIN)" || exit 1
    for i in $(seq 1 60); do sleep 10; wd_is $NEW_WD && break; done
    echo "weightd: $(running_wd)"; wd_is $NEW_WD || { echo "WEIGHTD-NOT-READY: rollback with 'release_rollout.sh rollback'"; exit 1; } ;;
  install)
    n=$(residentds); [ "$n" = 0 ] || { echo "residentds running: $n"; exit 1; }
    stamp "install release $SHA on every lane and hub API dir"
    for id in $MODELS; do
      L=$(field $id lane)
      fan "cd \$HOME/$L/root && cp ~/rp-release-$SHA/sparkpipe_model_residentd ~/rp-release-$SHA/weightd_warm ~/rp-release-$SHA/weightd_spine_budget.py bin/ && { [ ! -e bin/sparkpipe_model_batch ] || cp ~/rp-release-$SHA/sparkpipe_model_batch bin/; } && cp ~/rp-release-$SHA/libhidden_transport_spark_host_rdma_verbs.so lib/hidden_transport.so && cp ~/rp-release-$SHA/$(field $id fwlib) lib/$(field $id lib) && sha256sum bin/sparkpipe_model_residentd bin/weightd_warm lib/hidden_transport.so lib/$(field $id lib) | cut -c1-16 | tr '\n' ' '" | awk '{$1=""; print}' | sort | uniq -c | one_variant "$id" || { echo "INSTALL-FAILED $id"; exit 1; }
      hub "cd ~/$(field $id api) && install -m 755 $APIB/sparkpipe_model_api $APIB/sparkpipe_tokenize_prompt bin/ && install -m 755 $APIB/$(field $id fwlib) runtime/lib/$(field $id lib) && rm -f runtime/prefix_index.spi && echo hub $id \$(sha256sum bin/sparkpipe_model_api runtime/lib/$(field $id lib) | cut -c1-16 | tr '\n' ' ')" || { echo "INSTALL-FAILED hub $id"; exit 1; }
    done
    stamp "glm-5.3 lane switches to the official FP8 linear packs (glm53full.fp8lin.tp16)"
    fan "cd \$HOME/stage-lanes/glm53_full/lane6/root && n=glm53full.fp8lin.tp16-rank\$(python3 -c 'import json;print(json.load(open(\"config/adapter.json\"))[\"tp_rank\"])').glm52sp && d=\$HOME/sparkdata/glm53full.fp8lin.tp16/packs && test -s \$d/\$n.experts && rm -f packs/glm53full.fp8.tp16-rank*.glm52sp packs/glm53full.fp8.tp16-rank*.glm52sp.experts && ln -sfn \$d/\$n packs/\$n && ln -sfn \$d/\$n.experts packs/\$n.experts && printf '%s' \$(cut -c1-64 \$d/\$n.sha256) > packs/pack.sha256 && sed -i 's#glm53full\\.fp8\\.tp16#glm53full.fp8lin.tp16#g' config/adapter.json && grep -c fp8lin config/adapter.json" | awk '{print $2}' | sort | uniq -c | sed 's/^/glm fp8lin adapter.json lines: /'
    fan "cp ~/rp-release-$SHA/weightd_warm \$HOME/sparkdata/swap/weightd_warm && sha256sum < \$HOME/sparkdata/swap/weightd_warm | cut -c1-16" | awk '{print $2}' | sort | uniq -c | one_variant "shared weightd_warm" || { echo "INSTALL-FAILED shared weightd_warm"; exit 1; } ;;
  qualify)
    for id in ${ONLY:-$MODELS}; do
      stamp "qualify $id"
      if ! activate "$id"; then echo "NEW-ACTIVATE-FAILED $id"; fan "grep -a -i -E 'error|fail|refus|abi' \$HOME/$(field $id lane)/root/residentd.log | tail -2" | head -8; continue; fi
      capture "$id" new
      compare "$id" old new
      hub "cd ~/rollover/tree && timeout 3000 python3 tools/ui_smoke.py --endpoint http://127.0.0.1:4000 --model $id --reasoning both --out $RD/$id/ui-smoke.json > $RD/$id/smoke.log 2>&1; tail -2 $RD/$id/smoke.log | cut -c1-200"
      if [ -n "${COMPSEC:-}" ] || hub "cd $RD && python3 kit/release_compare.py $id/old.json $id/new.json | grep -q 'first-diff@'"; then
        eff="--reasoning-effort minimal"; [ "$id" = qwen3.8-max ] && eff="--max-tokens 2048"
        hub "cd ~/rollover/tree && timeout 7200 python3 tools/compsec17_openai.py --endpoint http://127.0.0.1:4000 --model $id $eff --out $RD/$id/compsec17 > $RD/$id/compsec.log 2>&1; grep -o 'RESULT: [0-9]*/[0-9]*' $RD/$id/compsec.log"
      fi
      hub "cd $RD/$id && python3 ~/evalkit/tools/api_serving_perf.py --endpoint http://127.0.0.1:$(field $id port) --cases o256,ttft6000 --repeats 1 --label $R --output perf.jsonl > /dev/null 2>&1; tail -2 perf.jsonl | python3 -c \"import json,sys; [print('$id', d.get('case'), 'decode', d.get('median_decode_tok_s'), 'prefill', d.get('prefill_tok_s')) for d in map(json.loads, sys.stdin)]\""
      fan "grep -a -c -i -E 'error|fail|fault|abort' \$HOME/$(field $id lane)/root/residentd.log" | awk '{s+=$2} END {print "rank log error lines:", s}'
    done
    restore_max ;;
  rollback)
    U=$(backup_utc) || exit 1
    stamp "ROLLBACK to root.prev-$U and weightd $OLD_WD"
    backups_present "$U" || { echo "ROLLBACK-REFUSED: the backups for $U are incomplete; nothing was changed"; exit 1; }
    "$0" down || exit 1
    restored=1
    for id in $MODELS; do
      L=$(field $id lane)
      fan_all "restore $L" "cd \$HOME/$L && $(restore_root $U) && { rmdir root/kvsnapshot root/kvcache 2>/dev/null; mv root.failed-$U/kvsnapshot root.failed-$U/kvcache root/ 2>/dev/null; true; }" || restored=0
      hub "test -d $BK/$(field $id api).prev-$U && rm -rf $BK/$(field $id api).failed-$U && mv ~/$(field $id api) $BK/$(field $id api).failed-$U && { cp -a $BK/$(field $id api).prev-$U ~/$(field $id api) || { rm -rf ~/$(field $id api); mv $BK/$(field $id api).failed-$U ~/$(field $id api); false; }; } && rm -f ~/$(field $id api)/runtime/prefix_index.spi" || { echo "restore hub $(field $id api) FAILED"; restored=0; }
    done
    fan_all "restore weightd_warm" "cp -p \$HOME/sparkdata/swap/weightd_warm.prev-$U \$HOME/sparkdata/swap/weightd_warm" || restored=0
    [ "$restored" = 1 ] || { echo "ROLLBACK-INCOMPLETE: a restore failed; weightd and the models were left down for a manual repair"; exit 1; }
    hub "cd ~/release/core && exec 9>\$HOME/release/.core.publish.lock && flock 9 && cp ~/release-staging/weightd-rollback/sparkpipe_weightd.$OLD_WD bin/sparkpipe_weightd.new && chmod 755 bin/sparkpipe_weightd.new && mv bin/sparkpipe_weightd.new bin/sparkpipe_weightd && test \"\$(sha256sum < bin/sparkpipe_weightd | cut -c1-16)\" = $OLD_WD && find bin -type f | sort | xargs sha256sum > MANIFEST.tmp && mv MANIFEST.tmp MANIFEST && printf '%s\n' $OLD_WD > WEIGHTSD_BIN.tmp && mv WEIGHTSD_BIN.tmp WEIGHTSD_BIN && echo announced \$(cat WEIGHTSD_BIN)"
    for i in $(seq 1 60); do sleep 10; wd_is $OLD_WD && break; done
    echo "weightd: $(running_wd)"
    wd_is $OLD_WD || { echo "ROLLBACK-WEIGHTD-NOT-READY: weightd $OLD_WD is not ready on 16 nodes; the models were left down"; exit 1; }
    for id in ${ONLY:-$MODELS}; do activate "$id" && capture "$id" rolledback && compare "$id" old rolledback; done
    restore_max ;;
  *) echo "usage: FW=<fw dir> APIB=<hub build dir> release_rollout.sh precheck|old|stage|backup|down|weightd|install|qualify|rollback"; exit 2 ;;
esac
stamp "$step done ($(controller))"
