#!/usr/bin/env bash
set -u
KIT=$(cd "$(dirname "$0")" && pwd)
RELEASE_FW=${FW:?FW is the local release firmware dir (release_build.sh output)}
RELEASE_APIB=${APIB:?APIB is the hub api build dir, e.g. /home/spec/api-build-<sha>/build}
. "$KIT/rollover_lib.sh"
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
OLD_WD=${OLD_WD:-60b3ca5d0d3e444b}
NEW_WD=$(shasum -a 256 "$FW/sparkpipe_weightd" | cut -c1-16)
MODELS="qwen3.8-max glm-5.3 kimi-k3 glm-5.3-flash qwen3.8-27b qwen3.8-flash deepseek-v4.1-flash gemma-4-31b gemma-4-26b-a4b ling-3.0-flash ling-3.0-flash-fin hy4-preview muse-glimmer-30b laguna-s-2.1 mimo-v2.6-flash mimo-v2.6-pro"
EXTRA_LANES="stage-lanes/qwen38_27b_bf16/lane11"
row() {
  case "$1" in
    glm-5.3) echo "stage-lanes/glm53_full/lane6 libglm53_full_tp16_serving_adapter.so libglm53_full_tp16_serving_adapter.so glm53_full-kv-api 8475 glm53_full_tp16_stage" ;;
    glm-5.3-flash) echo "stage-lanes/glm53_flash/lane0 libglm53_flash_tp16_serving_adapter.so libglm53_flash_tp16_serving_adapter.so glm53_flash-kv-api 8476 glm53_flash_stage" ;;
    kimi-k3) echo "k3-lanes/lane8 libk3_tp16_serving_adapter.so libk3_serving_adapter.so k3-kv-api 8459 k3_stage" ;;
    qwen3.8-27b) echo "stage-lanes/qwen38_27b/lane9 libqwen38_27b_tp16_serving_adapter.so libqwen38_27b_tp16_serving_adapter.so qwen38_27b-kv-api 8461 qwen38_27b_stage" ;;
    deepseek-v4.1-flash) echo "stage-lanes/dsv41_flash/lane10 libdsv41_flash_tp16_serving_adapter.so libdsv41_flash_tp16_serving_adapter.so dsv41_flash-kv-api 8463 dsv41_flash_stage" ;;
    qwen3.8-max) echo "stage-lanes/qwen38_max/lane12 libqwen38_max_tp16_serving_adapter.so libqwen38_max_tp16_serving_adapter.so qwen38_max-kv-api 8464 qwen38_max_stage" ;;
    qwen3.8-flash) echo "stage-lanes/qwen38_flash/lane13 libqwen38_flash_tp16_serving_adapter.so libqwen38_flash_tp16_serving_adapter.so qwen38_flash-kv-api 8465 qwen38_flash_stage" ;;
    gemma-4-31b) echo "stage-lanes/gemma4_31b/lane14 libgemma4_31b_tp16_serving_adapter.so libgemma4_31b_tp16_serving_adapter.so gemma4_31b-kv-api 8466 gemma4_31b_stage" ;;
    gemma-4-26b-a4b) echo "stage-lanes/gemma4_26b/lane15 libgemma4_26b_tp16_serving_adapter.so libgemma4_26b_tp16_serving_adapter.so gemma4_26b-kv-api 8467 gemma4_26b_stage" ;;
    ling-3.0-flash) echo "stage-lanes/ling3_flash/lane2 libling3_flash_tp16_serving_adapter.so libling3_flash_tp16_serving_adapter.so ling3_flash-kv-api 8468 ling3_stage" ;;
    ling-3.0-flash-fin) echo "stage-lanes/ling3_fin/lane2 libling3_fin_tp16_serving_adapter.so libling3_fin_tp16_serving_adapter.so ling3_fin-kv-api 8469 ling3_stage" ;;
    hy4-preview) echo "stage-lanes/hy4/lane3 libhy4_tp16_serving_adapter.so libhy4_tp16_serving_adapter.so hy4-kv-api 8470 hy4_tp16_stage" ;;
    muse-glimmer-30b) echo "stage-lanes/muse_glimmer/lane4 libmuse_glimmer_tp16_serving_adapter.so libmuse_glimmer_tp16_serving_adapter.so muse_glimmer-kv-api 8471 muse_glimmer_stage" ;;
    laguna-s-2.1) echo "stage-lanes/laguna/lane5 liblaguna_tp16_serving_adapter.so liblaguna_tp16_serving_adapter.so laguna-kv-api 8472 laguna_stage" ;;
    mimo-v2.6-flash) echo "stage-lanes/mimo26_flash/lane7 libmimo26_flash_tp16_serving_adapter.so libmimo26_flash_tp16_serving_adapter.so mimo26_flash-kv-api 8473 mimo26_flash_tp16_stage" ;;
    mimo-v2.6-pro) echo "stage-lanes/mimo26_pro/lane1 libmimo26_pro_tp16_serving_adapter.so libmimo26_pro_tp16_serving_adapter.so mimo26_pro-kv-api 8474 mimo26_pro_tp16_stage" ;;
    *) return 1 ;;
  esac
}
field() { local r k="$2"; r=$(row "$1") || return 1; set -- $r; case "$k" in lane) echo "$1";; fwlib) echo "$2";; lib) echo "$3";; api) echo "$4";; port) echo "$5";; label) echo "$6";; esac; }
for id in $MODELS; do for k in lane fwlib lib api port label; do v=$(field "$id" "$k"); case "$k:$v" in *:|*" "*|lane:/*|lane:*..*|[!l]*:*/*|l[!a]*:*/*) echo "field $k of $id is '$v': refusing to run" >&2; exit 1;; esac; done; case "$(field "$id" lane)" in */lane[0-9]*) ;; *) echo "lane of $id is not <dir>/lane<N>: refusing to run" >&2; exit 1;; esac; done
utc() { [ -f "$STATE/utc" ] || date -u +%Y%m%dT%H%M%SZ > "$STATE/utc"; cat "$STATE/utc"; }
backup_utc() { [ -s "$STATE/utc" ] || { echo "no backup stamp in $STATE/utc: run the backup step first" >&2; return 1; }; cat "$STATE/utc"; }
backups_present() {
  local U="$1" id L missing=0
  for id in $MODELS; do
    L=$(field $id lane)
    fan_all "backup $L/root.prev-$U" "test -d \$HOME/$L/root.prev-$U" || missing=1
    hub "test -d ~/$(field $id api).prev-$U" || { echo "backup hub ~/$(field $id api).prev-$U missing"; missing=1; }
  done
  for L in $EXTRA_LANES; do fan_all "backup $L/root.prev-$U" "test -d \$HOME/$L/root.prev-$U" || missing=1; done
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
residentds() { fan "pgrep -c -f 'bin/sparkpipe_model_[r]esidentd' || true" | awk '{s+=$2} END {print s+0}'; }
running_wd() { fan "sha256sum < /proc/\$(pgrep -o -f 'sparkdata/weightd/sparkpipe_[w]eightd')/exe | cut -c1-16; test -e /tmp/weightd-mesh/.ready && echo ready" | awk '{c[$2]++} END {for (k in c) printf "%s=%d ", k, c[k]}'; }
wd_is() { local s; s=$(running_wd); echo "$s" | grep -q "$1=16" && echo "$s" | grep -q "ready=16"; }
swap_cmd() { hub "python3 -c \"import json,sys; d=json.load(open('/home/spec/sparkpipe-swap/model_swap.json')); print([m for m in d['models'] if m['id']=='$1'][0]['$2'])\""; }

capture() {
  local id=$1 tag=$2 port; port=$(field "$id" port)
  hub "mkdir -p ~/$R/$id && cd ~/$R && timeout 3600 python3 kit/release_capture.py $port $id/$tag.json kit/prompt_source.c > $id/$tag.log 2>&1; echo capture $id $tag rc=\$?"
  case "$id" in
    glm-5.3) hub "cd ~/rollover/tree && timeout 2400 python3 tools/glm53_full_golden.py capture --prompts qualification/glm53_full/golden/prompts.json --port $port --max-tokens 128 --logprobs 4 --label $R-$tag --out ~/$R/$id/golden-$tag.json > ~/$R/$id/golden-$tag.log 2>&1; echo golden rc=\$?" ;;
    kimi-k3) hub "cd ~/rollover/tree && timeout 1800 python3 ~/$R/kit/rollover_k3_golden.py $port ~/$R/$id/golden-$tag.json > ~/$R/$id/golden-$tag.log 2>&1; echo golden rc=\$?" ;;
    qwen3.8-27b|qwen3.8-flash|qwen3.8-max) hub "cd ~/rollover/tree && timeout 1800 python3 ~/$R/kit/rollover_lp_capture.py qualification/qwen38_27b/reference/reference3_fp8_numpy.json $port ~/$R/$id/golden-$tag.json > /dev/null 2>&1; echo golden rc=\$?" ;;
    deepseek-v4.1-flash) hub "cd ~/rollover/tree && timeout 1800 python3 ~/$R/kit/rollover_lp_capture.py qualification/dsv41_flash/reference_tokens.json $port ~/$R/$id/golden-$tag.json > /dev/null 2>&1; echo golden rc=\$?" ;;
  esac
}
compare() {
  local id=$1 a=$2 b=$3
  hub "cd ~/$R && python3 kit/release_compare.py $id/$a.json $id/$b.json | tail -7"
  hub "test -f ~/$R/$id/golden-$a.json && test -f ~/$R/$id/golden-$b.json && { python3 ~/perfoverlap/exact.py ~/$R/$id/golden-$a.json ~/$R/$id/golden-$b.json | head -3; cd ~/rollover/tree && python3 ~/$R/kit/rollover_golden_compare.py ~/$R/$id/golden-$a.json ~/$R/$id/golden-$b.json 0.5 ~/$R/$id/golden-compare-$a-$b.json 2>&1 | tail -2; } || true"
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
    hub "mkdir -p ~/$R/kit" && scp -q "$KIT/../tools/release_capture.py" "$KIT/../tools/release_compare.py" "$KIT/../tools/rollover_golden_compare.py" "$KIT/../tools/rollover_k3_golden.py" "$KIT/../tools/rollover_lp_capture.py" rtx5090:$R/kit/ && scp -q "$KIT/../runtime/json.c" rtx5090:$R/kit/prompt_source.c
    echo PRECHECK-OK ;;
  old)
    for id in $MODELS; do stamp "old $id"; activate "$id" && capture "$id" old || echo "OLD-FAILED $id"; done
    restore_max ;;
  stage)
    stamp "stage $FW on 16 nodes (~/rp-release-$SHA) and check hashes"
    (cd "$FW" && COPYFILE_DISABLE=1 tar cf "$STATE/fw.tar" .)
    for r in $(seq 0 15); do (scp -q -o BatchMode=yes "$STATE/fw.tar" spark${HEX:$r:1}:rp-release-$SHA.tar && node $r "rm -rf ~/rp-release-$SHA && mkdir ~/rp-release-$SHA && tar -C ~/rp-release-$SHA -xf ~/rp-release-$SHA.tar 2>/dev/null && rm ~/rp-release-$SHA.tar && cd ~/rp-release-$SHA && sha256sum --quiet -c SHA256SUMS && echo verified" | sed "s/^/spark${HEX:$r:1}: /") & done | awk '{print $NF}' | sort | uniq -c | one_variant stage || { echo "STAGE-FAILED"; exit 1; } ;;
  backup)
    U=$(utc); stamp "backup root.prev-$U"
    for id in $MODELS; do
      L=$(field $id lane)
      fan_all "backup $L" "cd \$HOME/$L && [ ! -e root.prev-$U ] && mkdir root.prev-$U && rsync -a '--exclude=/kvsnapshot/*' '--exclude=/kvcache/*' root/ root.prev-$U/" || exit 1
      hub "[ -e ~/$(field $id api).prev-$U ] || cp -a ~/$(field $id api) ~/$(field $id api).prev-$U" || { echo "BACKUP-FAILED hub $(field $id api)"; exit 1; }
    done
    for L in $EXTRA_LANES; do fan_all "backup $L" "cd \$HOME/$L && [ ! -e root.prev-$U ] && mkdir root.prev-$U && rsync -a '--exclude=/kvsnapshot/*' '--exclude=/kvcache/*' root/ root.prev-$U/" || exit 1; done
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
    for L in $EXTRA_LANES; do fan_all "install $L" "cd \$HOME/$L/root && cp ~/rp-release-$SHA/sparkpipe_model_residentd ~/rp-release-$SHA/weightd_warm bin/ && cp ~/rp-release-$SHA/libhidden_transport_spark_host_rdma_verbs.so lib/hidden_transport.so && cp ~/rp-release-$SHA/libqwen38_27b_bf16_tp16_serving_adapter.so lib/" || exit 1; done
    stamp "glm-5.3 lane switches to the official FP8 linear packs (glm53full.fp8lin.tp16)"
    fan "cd \$HOME/stage-lanes/glm53_full/lane6/root && n=glm53full.fp8lin.tp16-rank\$(python3 -c 'import json;print(json.load(open(\"config/adapter.json\"))[\"tp_rank\"])').glm52sp && d=\$HOME/sparkdata/glm53full.fp8lin.tp16/packs && test -s \$d/\$n.experts && rm -f packs/glm53full.fp8.tp16-rank*.glm52sp packs/glm53full.fp8.tp16-rank*.glm52sp.experts && ln -sfn \$d/\$n packs/\$n && ln -sfn \$d/\$n.experts packs/\$n.experts && printf '%s' \$(cut -c1-64 \$d/\$n.sha256) > packs/pack.sha256 && sed -i 's#glm53full\\.fp8\\.tp16#glm53full.fp8lin.tp16#g' config/adapter.json && grep -c fp8lin config/adapter.json" | awk '{print $2}' | sort | uniq -c | sed 's/^/glm fp8lin adapter.json lines: /'
    fan "cp ~/rp-release-$SHA/weightd_warm \$HOME/sparkdata/swap/weightd_warm && sha256sum < \$HOME/sparkdata/swap/weightd_warm | cut -c1-16" | awk '{print $2}' | sort | uniq -c | one_variant "shared weightd_warm" || { echo "INSTALL-FAILED shared weightd_warm"; exit 1; } ;;
  qualify)
    for id in ${ONLY:-$MODELS}; do
      stamp "qualify $id"
      if ! activate "$id"; then echo "NEW-ACTIVATE-FAILED $id"; fan "grep -a -i -E 'error|fail|refus|abi' \$HOME/$(field $id lane)/root/residentd.log | tail -2" | head -8; continue; fi
      capture "$id" new
      compare "$id" old new
      hub "cd ~/rollover/tree && timeout 3000 python3 tools/ui_smoke.py --endpoint http://127.0.0.1:4000 --model $id --reasoning both --out ~/$R/$id/ui-smoke.json > ~/$R/$id/smoke.log 2>&1; tail -2 ~/$R/$id/smoke.log | cut -c1-200"
      if [ -n "${COMPSEC:-}" ] || ! hub "cd ~/$R && python3 kit/release_compare.py $id/old.json $id/new.json | grep -q 'EXACT 6/6'"; then
        eff="--reasoning-effort minimal"; [ "$id" = qwen3.8-max ] && eff="--max-tokens 2048"
        hub "cd ~/rollover/tree && timeout 7200 python3 tools/compsec17_openai.py --endpoint http://127.0.0.1:4000 --model $id $eff --out ~/$R/$id/compsec17 > ~/$R/$id/compsec.log 2>&1; grep -o 'RESULT: [0-9]*/[0-9]*' ~/$R/$id/compsec.log"
      fi
      hub "cd ~/$R/$id && python3 ~/evalkit/tools/api_serving_perf.py --endpoint http://127.0.0.1:$(field $id port) --cases o256,ttft6000 --repeats 1 --label $R --output perf.jsonl > /dev/null 2>&1; tail -2 perf.jsonl | python3 -c \"import json,sys; [print('$id', d.get('case'), 'decode', d.get('median_decode_tok_s'), 'prefill', d.get('prefill_tok_s')) for d in map(json.loads, sys.stdin)]\""
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
      hub "test -d ~/$(field $id api).prev-$U && rm -rf ~/$(field $id api).failed-$U && mv ~/$(field $id api) ~/$(field $id api).failed-$U && { cp -a ~/$(field $id api).prev-$U ~/$(field $id api) || { rm -rf ~/$(field $id api); mv ~/$(field $id api).failed-$U ~/$(field $id api); false; }; } && rm -f ~/$(field $id api)/runtime/prefix_index.spi" || { echo "restore hub $(field $id api) FAILED"; restored=0; }
    done
    for L in $EXTRA_LANES; do fan_all "restore $L" "cd \$HOME/$L && $(restore_root $U)" || restored=0; done
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
