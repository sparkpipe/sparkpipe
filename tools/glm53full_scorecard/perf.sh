#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SSH="${PERF_SSH:-ssh -o BatchMode=yes -o ConnectTimeout=8}"
SCP="${PERF_SCP:-scp -q}"
HUB="${PERF_HUB:-rtx5090}"
RUNNER_HOST="${PERF_RUNNER_HOST:-sparkf}"
SESSIONS_DIR="${PERF_SESSIONS:-/Users/mac/wf/glmfull-scorecard-kit/sessions}"
OUT_BASE="${PERF_OUT:-/Users/mac/wf/glmfull-scorecard-kit/perf-out}"
HEX=0123456789abcdef
LIVE_LANE=6
LIVE_ROOT="glmfull-lane6/root"
LIVE_UNIT=sp-glmfull-rd6
SPINE_GIB=8
FLOOR_GIB=10
READY_TIMEOUT_S=900
DEFAULT_SESSIONS="t1,b1,b1c,b8,b16,b32,b64,b128,b256,ttft1k,ttft4k,ttft16k,pl8,compsec17"
SPEC_SESSIONS="t1,b1,b1c,b8"
LONG_SESSIONS="ttft4k,ttft16k"

ROOT="" LANE="" NAME="" SESSIONS="$DEFAULT_SESSIONS" SEQ="" POS="" LONG="" SPEC_ENV="" EXTRA_ENV=""
usage() {
  cat >&2 <<EOF
usage: $0 run|stage|up|down|status --root NODE_RELATIVE_ROOT --lane N --name NAME
          [--sessions a,b,..] [--kv SEQUENCES:POSITIONS] [--long SEQUENCES:POSITIONS]
          [--env SPARK_X=v]... [--spec-env "SPARK_X=v SPARK_Y=w"]
  run    = stage + up + sessions (+ long-context and spec phases) + down + one table
  root   = a GLM Full root under each node's home (glmfull-lane6/root = live release, or a lane's dev root)
EOF
  exit 2
}
CMD="${1:-}"; [ -n "$CMD" ] || usage; shift
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --lane) LANE="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --sessions) SESSIONS="$2"; shift 2 ;;
    --kv) SEQ="${2%%:*}"; POS="${2##*:}"; shift 2 ;;
    --long) LONG="$2"; shift 2 ;;
    --env) EXTRA_ENV="$EXTRA_ENV $2"; shift 2 ;;
    --spec-env) SPEC_ENV="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$ROOT" =~ ^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$ ]] || usage
[[ "$LANE" =~ ^[0-9]+$ ]] && [ "$LANE" -lt 16 ] && [ "$LANE" != "$LIVE_LANE" ] || { echo "--lane must be 0..15 and not the live lane $LIVE_LANE" >&2; exit 2; }
[[ "$NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "--name must be lowercase [a-z0-9-]" >&2; exit 2; }
for kv in $EXTRA_ENV $SPEC_ENV; do [[ "$kv" =~ ^SPARK_[A-Z0-9_]+=[A-Za-z0-9_.,:/-]+$ ]] || { echo "bad env '$kv'" >&2; exit 2; }; done
DEST="glmfull-dev/$NAME"
UNIT="sp-gfd-$NAME-rd"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUT_BASE/$STAMP-$NAME"
RUN_DIR="/home/$RUNNER_HOST/build-glmfull-scorecard/perf"
mkdir -p "$OUT"
host_of() { echo "spark${HEX:$1:1}"; }
log() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$OUT/perf.log"; }
die() { log "FAIL: $*"; exit 1; }

each_node() {
  local r pids=() failed=""
  for r in $(seq 0 15); do
    $SSH "$(host_of "$r")" "${1//@rank@/$r}" > "$OUT/node-$r.out" 2>&1 &
    pids[$r]=$!
  done
  for r in $(seq 0 15); do wait "${pids[$r]}" || failed="$failed $(host_of "$r")"; done
  [ -z "$failed" ] || { log "failed on:$failed"; return 1; }
}

live_field() {
  $SSH "$HUB" "python3 -c 'import json,os; print(json.load(open(os.path.expanduser(\"~/fleet-rotation/rotation.json\")))[\"models\"][\"glmfull\"][\"engine\"][\"$1\"])'"
}

start_cmd() {
  local envs="$1" start extra="-E SPARK_WEIGHTD_SHARE=readonly"
  start="$(live_field start)"
  for kv in $envs; do extra="$extra -E $kv"; done
  grep -q "SPARK_WEIGHTD_LANE=$LIVE_LANE " <<<"$start" && grep -q "$LIVE_UNIT" <<<"$start" && grep -q "$LIVE_ROOT" <<<"$start" || die "the live start command changed shape: $start"
  start="${start//$LIVE_ROOT/$DEST}"
  start="${start//$LIVE_UNIT/$UNIT}"
  start="${start//SPARK_WEIGHTD_LANE=$LIVE_LANE /SPARK_WEIGHTD_LANE=$LANE }"
  echo "${start/ bash -c / $extra bash -c }"
}

ready_check() {
  local ready
  ready="$(live_field ready)"
  grep -q "$LIVE_UNIT" <<<"$ready" && grep -q "$LIVE_ROOT" <<<"$ready" || die "the live ready command changed shape: $ready"
  ready="${ready//$LIVE_ROOT/$DEST}"
  echo "${ready//$LIVE_UNIT/$UNIT} && grep -q 'WD-MAP-POOL-BULK .*access=read-only' residentd.log && grep -q \"\$HOME/$DEST/lib/model_driver.so\" /proc/\$(systemctl --user show -p MainPID --value $UNIT)/maps"
}

mem_need_gib() {
  python3 -c "import math; print($SPINE_GIB + math.ceil($1 * $2 * 95232 / 2**30))"
}

preflight() {
  local need="$1" r m bad=""
  each_node "test -S /tmp/spark_weightd.sock && echo SOCK; cd ~/$ROOT && test -x bin/sparkpipe_model_residentd -a -f lib/model_driver.so -a -f model_resident.json -a -f config/stage_\$(printf %02d @rank@).json -a -s packs/pack.sha256 && echo ROOT; systemctl --user is-active -q $UNIT && echo UNIT-ACTIVE; awk '/MemAvailable/{print \"MEM\", int(\$2/1048576)}' /proc/meminfo" || die "a node is unreachable"
  for r in $(seq 0 15); do
    grep -qx SOCK "$OUT/node-$r.out" || bad="$bad $(host_of "$r"):no-weightd-socket"
    grep -qx ROOT "$OUT/node-$r.out" || bad="$bad $(host_of "$r"):root-incomplete"
    grep -qx UNIT-ACTIVE "$OUT/node-$r.out" && bad="$bad $(host_of "$r"):$UNIT-already-active"
    m="$(awk '/^MEM /{print $2}' "$OUT/node-$r.out")"
    [ "${m:-0}" -ge $((FLOOR_GIB + need)) ] || bad="$bad $(host_of "$r"):MemAvailable-${m}GiB<$((FLOOR_GIB + need))"
  done
  [ -z "$bad" ] || die "preflight:$bad"
  log "preflight OK: root ~/$ROOT 16/16, weightd socket 16/16, MemAvailable >= $((FLOOR_GIB + need)) GiB 16/16; other GLM units now: $(others)"
}

others() {
  each_node "systemctl --user list-units --state=active --no-legend 'sp-gfd-*-rd*' 'sp-glmfull-rd*' | awk '\$1 != \"$UNIT.service\" {print \"OTHER\", \$1}'" > /dev/null 2>&1 || true
  cat "$OUT"/node-*.out | awk '/^OTHER /{print $2}' | sort | uniq -c | awk '{printf "%s(%s) ", $2, $1}' | sed 's/ $//' | grep . || echo none
}

stage() {
  local seq="$1" pos="$2" src="$OUT/source" gen="$OUT/staged" r
  mkdir -p "$src/config"
  $SCP "spark0:$ROOT/model_resident.json" "spark0:$ROOT/RELEASE" "$src/" || die "fetch the source deployment"
  for r in $(seq 0 15); do $SCP "$(host_of "$r"):$ROOT/config/stage_$(printf %02d "$r").json" "$src/config/" & done
  wait
  python3 "$HERE/runner.py" relane "$src" "$gen" "$LANE" "$DEST" $seq $pos | tee -a "$OUT/perf.log"
  each_node "rm -rf ~/$DEST.tmp && mkdir -p ~/$DEST.tmp/config ~/$DEST.tmp/kvcache && cd ~/$ROOT && cp -a bin lib packs ~/$DEST.tmp/ && cp -a SOURCE_COMMIT MODULE_IDENTITY ~/$DEST.tmp/ 2>/dev/null; (grep -q \"^\$HOME/$DEST	\" ~/KEEP || printf \"\$HOME/$DEST\tglmfull-scorecard perf.sh\tKEEP\tGLM Full perf-test root (copy of ~/$ROOT on lane $LANE)\n\" >> ~/KEEP)" || die "copy the root"
  for r in $(seq 0 15); do
    ( h="$(host_of "$r")"
      $SCP "$gen/model_resident.json" "$gen/RELEASE" "$h:$DEST.tmp/" && $SCP "$gen/config/stage_$(printf %02d "$r").json" "$h:$DEST.tmp/config/" ) &
  done
  wait
  each_node "cd ~/$DEST.tmp && sha256sum bin/* lib/* config/* model_resident.json RELEASE packs/pack.sha256 > MANIFEST && rm -rf ~/$DEST && mv ~/$DEST.tmp ~/$DEST && echo STAGED" || die "stage"
  grep -h '^source_commit=\|^release=' "$src/RELEASE" | tr '\n' ' ' | sed "s/^/source root: /" | tee -a "$OUT/perf.log"; echo
  log "staged ~/$DEST on 16/16 (lane $LANE, unit $UNIT)"
}

down() {
  each_node "systemctl --user stop $UNIT 2>/dev/null; systemctl --user reset-failed $UNIT 2>/dev/null; systemctl --user is-active -q $UNIT && echo STILL-ACTIVE; true" || true
  grep -qx STILL-ACTIVE "$OUT"/node-*.out && die "$UNIT still active somewhere"
  log "$UNIT stopped on 16/16"
}

up() {
  local envs="$1" start rcmd t0=$SECONDS n r
  start="$(start_cmd "$envs")"
  rcmd="$(ready_check)"
  echo "$start" > "$OUT/start.cmd"
  log "start $UNIT from ~/$DEST (live start command; lane $LANE; SPARK_WEIGHTD_SHARE=readonly; extra env '$envs')"
  each_node "cd ~/$DEST && sha256sum -c --quiet MANIFEST && $start" || { down; die "start"; }
  while [ $((SECONDS - t0)) -lt $READY_TIMEOUT_S ]; do
    sleep 10
    each_node "if $rcmd; then echo READY; elif systemctl --user is-active -q $UNIT; then echo STARTING; else echo DEAD; tail -5 ~/$DEST/residentd.log; fi" || true
    n="$(grep -lx READY "$OUT"/node-*.out | wc -l | tr -d ' ')"
    if grep -qx DEAD "$OUT"/node-*.out; then
      for r in $(seq 0 15); do grep -qx DEAD "$OUT/node-$r.out" && { log "rank $r died:"; cat "$OUT/node-$r.out" >> "$OUT/perf.log"; }; done
      rank_logs up-dead; down; die "a rank died during start (ERRSITE: read include/sparkpipe/spark_status.h; 15 = BUSY)"
    fi
    [ "$n" = 16 ] && { log "16/16 ready in $((SECONDS - t0)) s (read-only arena map, pinned experts, graph chain, own driver mapped)"; return 0; }
  done
  rank_logs up-timeout; down; die "not ready after $READY_TIMEOUT_S s"
}

rank_logs() {
  local tag="$1" r
  each_node "grep -E 'model_residentd ready|WD-MAP-POOL-BULK|EXPERT-RESIDENCY|GLM52-CHAIN-MODE|GLM-KV|refused|ERRSITE' ~/$DEST/residentd.log | head -8; echo ERRSITE-COUNT \$(grep -c ERRSITE ~/$DEST/residentd.log)" || true
  for r in $(seq 0 15); do printf '%s: %s\n' "$(host_of "$r")" "$(cut -c1-220 "$OUT/node-$r.out" | tr '\n' ';')"; done > "$OUT/ranks-$tag.txt"
  $SCP "spark0:$DEST/residentd.log" "$OUT/rank0-$tag.log" 2>/dev/null || true
}

stage_runner() {
  $SSH "$RUNNER_HOST" "mkdir -p $RUN_DIR/sessions $RUN_DIR/tools/glm53full_scorecard $RUN_DIR/out-$STAMP && (grep -q '^/home/$RUNNER_HOST/build-glmfull-scorecard	' ~/KEEP || printf '/home/$RUNNER_HOST/build-glmfull-scorecard\tglmfull-scorecard\tKEEP\tGLM Full perf runner + receipts\n' >> ~/KEEP)"
  $SCP "$HERE/runner.py" "$HERE/roofline.py" "$RUNNER_HOST:$RUN_DIR/tools/glm53full_scorecard/"
  $SCP "$REPO/tools/glm53full_lane.py" "$RUNNER_HOST:$RUN_DIR/tools/"
  $SCP "$SESSIONS_DIR"/*.json "$RUNNER_HOST:$RUN_DIR/sessions/"
}

session() {
  local name="$1" tag="$2" code=0
  [ -f "$SESSIONS_DIR/$name.json" ] || { log "no session $name in $SESSIONS_DIR"; return 0; }
  log "session $name ($tag)"
  $SSH "$RUNNER_HOST" "cd $RUN_DIR && python3 tools/glm53full_scorecard/runner.py run sessions/$name.json $RUN_DIR/out-$STAMP \$HOME/$DEST/bin \$HOME/$DEST $tag" >> "$OUT/sessions.log" 2>&1 || code=$?
  $SCP -r "$RUNNER_HOST:$RUN_DIR/out-$STAMP/." "$OUT/" || true
  case "$code" in
    0) log "  $name PASS"; return 0 ;;
    3) log "  $name INVALID (a fixed-length case stopped early)"; return 0 ;;
    4) log "  $name SKIPPED ($(tail -1 "$OUT/results.jsonl" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("skipped"))'))"; return 0 ;;
    *) log "  $name FAIL exit $code (sessions.log)"; return 1 ;;
  esac
}

compsec() {
  local tag="$1"
  [ -f "$OUT/$tag-compsec17.ndjson" ] || return 0
  python3 "$REPO/tools/glm53full_compsec17.py" grade --prompts "$REPO/qualification/glm53full/compsec17_prompts_off.json" \
    --tokenizer "$REPO/qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json" --max-tokens 512 \
    --out "$OUT/compsec17-$tag-graded.json" "$OUT/$tag-compsec17.ndjson" > "$OUT/compsec17-$tag.txt" 2>&1 || true
  log "COMPSEC-17 $tag: $(python3 -c "import json; d=json.load(open('$OUT/compsec17-$tag-graded.json')); print(d.get('passed', d.get('pass_count')), '/', d.get('total', 17), 'verdict', d.get('verdict', d.get('status')))" 2>/dev/null || tail -2 "$OUT/compsec17-$tag.txt")"
}

phase() {
  local tag="$1" list="$2" s rc
  for s in ${list//,/ }; do
    rc=0; session "$s" "$tag" || rc=$?
    if [ "$rc" != 0 ] && [ "$s" = t1 ]; then log "T1-FAIL $tag: tokens differ from the reference, remaining $tag sessions skipped"; return 1; fi
    if [ "$rc" != 0 ]; then
      each_node "systemctl --user is-active -q $UNIT && echo UP || echo DOWN" || true
      for r in $(seq 0 15); do cp "$OUT/node-$r.out" "$OUT/rank-$r.state"; done
      if grep -qx DOWN "$OUT"/node-*.out; then
        rank_logs "$tag-$s"
        log "rank(s) down after $s:$(for r in $(seq 0 15); do grep -qx DOWN "$OUT/rank-$r.state" 2>/dev/null && printf ' %s' "$(host_of "$r")"; done) (their last log lines: ranks-$tag-$s.txt; units are started with identical command lines on every lane, so a pkill -f by command line from any lane kills this one too)"
        return 1
      fi
    fi
  done
  compsec "$tag"
}

run_all() {
  local seq pos need lseq lpos
  seq="${SEQ:-0}"; pos="${POS:-0}"
  stage_runner
  if [ -z "$SEQ" ]; then
    $SCP "spark0:$ROOT/RELEASE" "$OUT/source-RELEASE" || die "read ~/$ROOT/RELEASE"
    seq="$(sed -n 's/^sequences=//p' "$OUT/source-RELEASE")"; pos="$(sed -n 's/^positions=//p' "$OUT/source-RELEASE")"
    need="$(mem_need_gib "$seq" "$pos")"; preflight "$need"; stage "" ""
  else
    need="$(mem_need_gib "$seq" "$pos")"; preflight "$need"; stage "$seq" "$pos"
  fi
  trap 'log "interrupted"; down; exit 1' INT TERM
  others > "$OUT/others-start.txt"
  up "$EXTRA_ENV"
  rank_logs base
  phase base "$SESSIONS" || true
  down
  if [ -n "$LONG" ]; then
    lseq="${LONG%%:*}"; lpos="${LONG##*:}"
    preflight "$(mem_need_gib "$lseq" "$lpos")"; stage "$lseq" "$lpos"
    up "$EXTRA_ENV" && phase long "$LONG_SESSIONS" || true
    down
  fi
  if [ -n "$SPEC_ENV" ]; then
    stage "${SEQ:-}" "${POS:-}"
    up "$EXTRA_ENV $SPEC_ENV" && phase spec "$SPEC_SESSIONS" || true
    python3 "$HERE/runner.py" compare "$OUT/results.jsonl" base spec t1,b1 2>&1 | tee -a "$OUT/perf.log" || log "spec tokens differ from base (see above)"
    down
  fi
  trap - INT TERM
  python3 "$HERE/runner.py" summary "$OUT/results.jsonl" "$OUT" "GLM-5.3 Full perf ~/$ROOT on lane $LANE ($STAMP; other GLM units at start: $(cat "$OUT/others-start.txt"); at end: $(others))" | tee -a "$OUT/perf.log"
  log "DONE: $OUT/SUMMARY.md (receipts results.jsonl, *.ndjson, perf.log); root ~/$DEST kept (KEEP)"
}

case "$CMD" in
  run) run_all ;;
  stage) stage "${SEQ:-}" "${POS:-}" ;;
  up) preflight "$(mem_need_gib "${SEQ:-16}" "${POS:-2048}")"; up "$EXTRA_ENV" ;;
  down) down ;;
  status) each_node "systemctl --user is-active $UNIT; grep -c 'model_residentd ready' ~/$DEST/residentd.log 2>/dev/null; awk '/MemAvailable/{print int(\$2/1048576) \" GiB\"}' /proc/meminfo" || true; for r in $(seq 0 15); do echo "$(host_of "$r") $(tr '\n' ' ' < "$OUT/node-$r.out")"; done ;;
  *) usage ;;
esac
