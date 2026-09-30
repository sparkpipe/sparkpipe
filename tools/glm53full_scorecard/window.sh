#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
KIT="${SCORECARD_KIT:-/Users/mac/wf/glmfull-scorecard-kit}"
SSH="${WINDOW_SSH:-ssh -o BatchMode=yes -o ConnectTimeout=8}"
SCP="${WINDOW_SCP:-scp -q}"
HUB="${WINDOW_HUB:-rtx5090}"
HEX=0123456789abcdef
LANE=6
NODE_ROOT="glmfull-lane$LANE"
UNIT=sp-glmfull-rd6
API_UNIT=glmfull-api6
API_PORT=8446
ROT="cd \$HOME/fleet-rotation && python3 bin/fleet_rotation.py --config rotation.json"
RUNNER_HOST="${WINDOW_RUNNER_HOST:-sparkf}"
WINDOW_DIR="/home/$RUNNER_HOST/build-glmfull-scorecard/window"
LABEL="${WINDOW_LABEL:-glmfull-scorecard}"
PERF="${WINDOW_PERF-python3 /Users/mac/sparkpipe-coord/tools/perf_window.py}"
PERF_HOLDER_FILE="${WINDOW_PERF_HOLDER:-/Users/mac/sparkpipe-coord/lanes/PERF_HOLDER}"
FLOOR_GIB="${FLOOR_GIB:-20}"
MAX_MINUTES="${MAX_MINUTES:-55}"
RESTORE_RESERVE_MIN=7
KV_BYTES_PER_TOKEN=95232
LIVE_TOKENS=32768
DEFAULT_SESSIONS="t1,b1,b1c,b8,b16,b32,b64,b128,ttft,ttft1k,ttft4k,ttft16k,ttft32k,pl8,pl16,compsec17"
STAMP="${WINDOW_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="$KIT/window-out/$STAMP"
T_START=$SECONDS
TOUCHED=0
API_STOPPED=0
LIVE_RUNNING_AS_C=0
declare -a ARM_TAGS=() ARM_ROOTS=() ARM_ENVS=() ARM_READY=() ARM_SESSIONS=()
mkdir -p "$OUT"
est() {
  case "$1" in
    t1|b8|ttft|ttft1k|pl8) echo 1 ;;
    b16) echo 1.5 ;;
    b1|b1c|b32|ttft4k|pl16|seq6) echo 2 ;;
    b64) echo 2.5 ;;
    compsec17) echo 3 ;;
    b128|ttft16k) echo 4 ;;
    ttft32k) echo 9 ;;
    *) echo 2 ;;
  esac
}
host_of() { echo "spark${HEX:$1:1}"; }
log() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$OUT/window.log"; }
elapsed_min() { echo $(( (SECONDS - T_START) / 60 )); }
left_min() { echo $(( MAX_MINUTES - (SECONDS - T_START) / 60 )); }

each_node() {
  local r pids=() failed=""
  for r in $(seq 0 15); do
    $SSH "$(host_of "$r")" "${1//@rank@/$r}" > "$OUT/node-$r.out" 2>&1 &
    pids[$r]=$!
  done
  for r in $(seq 0 15); do wait "${pids[$r]}" || failed="$failed $(host_of "$r")"; done
  if [ -n "$failed" ]; then log "FAILED on:$failed"; return 1; fi
}

rot_field() {
  $SSH "$HUB" "python3 -c 'import json,os; m=json.load(open(os.path.expanduser(\"~/fleet-rotation/rotation.json\")))[\"models\"][\"glmfull\"]; print(m$1)'"
}

min_avail() {
  each_node "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" || return 1
  for r in $(seq 0 15); do head -1 "$OUT/node-$r.out"; done | sort -n | head -1
}

health() {
  local m
  m="$(min_avail)" || stop "a node is unreachable"
  [ "$m" -ge "$FLOOR_GIB" ] || stop "min MemAvailable $m GiB < floor $FLOOR_GIB GiB"
  log "health: min MemAvailable $m GiB"
}

load_arms() {
  local file="$1" tag root envs ready sessions
  [ -f "$file" ] || { echo "arms file $file not found" >&2; exit 2; }
  while IFS='|' read -r tag root envs ready sessions; do
    case "$tag" in ''|'#'*) continue ;; esac
    [[ "$tag" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "bad arm tag '$tag'" >&2; exit 2; }
    [[ "$root" =~ ^(root|arm\.[A-Za-z0-9_.-]+)$ ]] || { echo "arm $tag: root must be 'root' or arm.<tag> under ~/$NODE_ROOT, got '$root'" >&2; exit 2; }
    for kv in $envs; do [[ "$kv" =~ ^SPARK_[A-Z0-9_]+=[A-Za-z0-9_.,:/-]+$ ]] || { echo "arm $tag: bad env '$kv'" >&2; exit 2; }; done
    ARM_TAGS+=("$tag"); ARM_ROOTS+=("$root"); ARM_ENVS+=("$envs"); ARM_READY+=("$ready"); ARM_SESSIONS+=("${sessions:-$DEFAULT_SESSIONS}")
  done < "$file"
  [ "${#ARM_TAGS[@]}" -gt 0 ] || { echo "arms file $file has no arms" >&2; exit 2; }
  cp "$file" "$OUT/arms.txt"
}

holder_guard() {
  local holder
  [ -n "$PERF" ] || { log "WINDOW_PERF is empty: running under the lead's own lock"; return 0; }
  holder="$(cat "$PERF_HOLDER_FILE" 2>/dev/null || true)"
  case "$holder" in
    lead-"$LABEL"*) ;;
    lead-*|glm-*) log "REFUSED: PERF_HOLDER is '${holder:0:100}'; wait, or run with WINDOW_PERF= inside the lead's own window"; return 1 ;;
  esac
  log "PERF_HOLDER='${holder:0:80}' perf label $LABEL (LEAD_RESERVED 'lead-$LABEL <until>' lets these sessions through)"
}

fleet_guard() {
  local status timer
  status="$($SSH "$HUB" "$ROT status")"
  log "rotation: $status"
  timer="$($SSH "$HUB" "systemctl --user is-active fleet-rotation.timer 2>/dev/null; test -e \$HOME/fleet-rotation/ROTATION_PAUSE && echo MANUAL-PAUSED || echo NOT-PAUSED; true" | tr '\n' ' ')"
  log "rotation timer/pause: $timer"
  case "$timer" in "inactive MANUAL-PAUSED "*|"failed MANUAL-PAUSED "*) ;; *) log "REFUSED: the fleet is not manual (timer must be inactive and ROTATION_PAUSE present)"; return 1 ;; esac
  case "$status" in *" active=glmfull "*|*" active=glmfull,"*|*",glmfull "*|*",glmfull,"*) ;; *) log "REFUSED: glmfull is not active"; return 1 ;; esac
}

arm_check_cmd() {
  local root="$1"
  echo "cd ~/$NODE_ROOT/$root && sha256sum -c --quiet MANIFEST && echo MANIFEST-OK-$root && sed -n 's/^\\(release\\|source_commit\\|sequences\\|rows\\|positions\\)=/CAP-$root-\\1=/p' RELEASE && { [ '$root' = root ] || grep -q '\"runtime_root\": \"/home/'\$(hostname)'/$NODE_ROOT/$root\"' model_resident.json && echo SELF-OK-$root; }"
}

passive() {
  local r bad="" cmd="" i root need m
  fleet_guard || return 1
  cmd="$(rot_field '["engine"]["ready"]') && echo LIVE-READY"
  for i in "${!ARM_TAGS[@]}"; do cmd="$cmd; $(arm_check_cmd "${ARM_ROOTS[$i]}")"; done
  each_node "$cmd" || true
  for r in $(seq 0 15); do
    grep -qx LIVE-READY "$OUT/node-$r.out" || bad="$bad $(host_of "$r"):live-not-ready"
    for i in "${!ARM_TAGS[@]}"; do
      root="${ARM_ROOTS[$i]}"
      grep -qx "MANIFEST-OK-$root" "$OUT/node-$r.out" || bad="$bad $(host_of "$r"):$root-manifest"
      [ "$root" = root ] || grep -qx "SELF-OK-$root" "$OUT/node-$r.out" || bad="$bad $(host_of "$r"):$root-runtime_root"
    done
  done
  cp "$OUT/node-0.out" "$OUT/passive-node0.txt"
  [ -z "$bad" ] || { log "PASSIVE-FAIL:$bad"; return 1; }
  m="$(min_avail)" || { log "PASSIVE-FAIL: a node is unreachable"; return 1; }
  for i in "${!ARM_TAGS[@]}"; do
    root="${ARM_ROOTS[$i]}"
    need="$(arm_extra_gib "$root")"
    log "arm ${ARM_TAGS[$i]}: ~/$NODE_ROOT/$root $(grep "^CAP-$root-" "$OUT/passive-node0.txt" | sed "s/^CAP-$root-//" | tr '\n' ' ')KV +${need} GiB/node over live; env '${ARM_ENVS[$i]}'; sessions ${ARM_SESSIONS[$i]}"
    [ "$m" -ge $((FLOOR_GIB + need)) ] || { log "PASSIVE-FAIL: arm ${ARM_TAGS[$i]} needs $((FLOOR_GIB + need)) GiB free, min MemAvailable is $m GiB"; return 1; }
  done
  log "PASSIVE-OK: fleet manual, live GLM Full ready 16/16, ${#ARM_TAGS[@]} arm roots verified 16/16, min MemAvailable $m GiB, estimate $(plan_minutes) min of $MAX_MINUTES"
}

arm_extra_gib() {
  local root="$1" seq pos
  seq="$(sed -n "s/^CAP-$root-sequences=//p" "$OUT/passive-node0.txt")"
  pos="$(sed -n "s/^CAP-$root-positions=//p" "$OUT/passive-node0.txt")"
  python3 -c "import math; print(max(0, math.ceil(($seq * $pos - $LIVE_TOKENS) * $KV_BYTES_PER_TOKEN / 2**30)))"
}

plan_minutes() {
  local i s total=6
  for i in "${!ARM_TAGS[@]}"; do
    [ "$i" = 0 ] && [ "${ARM_ROOTS[$i]}" = root ] && [ -z "${ARM_ENVS[$i]}" ] || total="$(python3 -c "print($total + 5)")"
    for s in ${ARM_SESSIONS[$i]//,/ }; do total="$(python3 -c "print($total + $(est "$s"))")"; done
  done
  python3 -c "print(int($total + 0.999))"
}

api_stop() {
  $SSH "$HUB" "systemctl --user stop $API_UNIT 2>/dev/null; systemctl --user reset-failed $API_UNIT 2>/dev/null; true"
  API_STOPPED=1
  log "API $API_UNIT stopped on $HUB (sessions use sparkpipe_model_batch)"
}

start_cmd() {
  local root="$1" envs="$2" start extra=""
  start="$(rot_field '["engine"]["start"]')"
  start="${start//glmfull-lane$LANE\/root/glmfull-lane$LANE/$root}"
  for kv in $envs; do extra="$extra -E $kv"; done
  echo "${start/ bash -c / $extra bash -c }"
}

ready_cmd() {
  local root="$1" ready="$2" base checks="" pattern
  base="$(rot_field '["engine"]["ready"]')"
  base="${base//glmfull-lane$LANE\/root/glmfull-lane$LANE/$root}"
  if [ -n "$ready" ]; then
    IFS=';' read -ra pats <<<"$ready"
    for pattern in "${pats[@]}"; do [ -z "$pattern" ] || checks="$checks && grep -q '$pattern' residentd.log"; done
  fi
  checks="$checks && grep -q \"\$HOME/$NODE_ROOT/$root/lib/model_driver.so\" /proc/\$(systemctl --user show -p MainPID --value $UNIT)/maps"
  [ "$root" = root ] || checks="$checks && ! grep -q \"\$HOME/$NODE_ROOT/root/lib/\" /proc/\$(systemctl --user show -p MainPID --value $UNIT)/maps"
  echo "${base/ && systemctl --user is-active -q $UNIT/$checks && systemctl --user is-active -q $UNIT}"
}

unit_stop() {
  each_node "systemctl --user stop $UNIT 2>/dev/null; systemctl --user reset-failed $UNIT 2>/dev/null; systemctl --user is-active -q $UNIT && echo STILL-ACTIVE; true" || true
  for r in $(seq 0 15); do grep -qx STILL-ACTIVE "$OUT/node-$r.out" && { log "unit still active on $(host_of "$r")"; return 1; }; done
  log "$UNIT stopped on 16"
}

arm_up() {
  local tag="$1" root="$2" envs="$3" ready="$4" start rcmd deadline=$((SECONDS + 900)) r n
  start="$(start_cmd "$root" "$envs")"
  rcmd="$(ready_cmd "$root" "$ready")"
  echo "$start" > "$OUT/start-$tag.cmd"
  echo "$rcmd" > "$OUT/ready-$tag.cmd"
  log "arm $tag: start $UNIT from ~/$NODE_ROOT/$root (rotation start command, root replaced, env '$envs')"
  each_node "cd ~/$NODE_ROOT/$root && sha256sum -c --quiet MANIFEST && $start" || { log "arm $tag: start failed"; return 1; }
  while [ $SECONDS -lt $deadline ]; do
    each_node "if $rcmd; then echo READY; elif systemctl --user is-active -q $UNIT; then echo STARTING; else echo DEAD; tail -4 ~/$NODE_ROOT/$root/residentd.log; fi" || true
    n=0
    for r in $(seq 0 15); do
      grep -qx READY "$OUT/node-$r.out" && n=$((n + 1))
      if grep -qx DEAD "$OUT/node-$r.out"; then cat "$OUT/node-$r.out" >> "$OUT/window.log"; log "arm $tag: rank $r unit is not active"; rank_logs "$root" "$tag-dead"; return 1; fi
    done
    [ "$n" -eq 16 ] && { log "arm $tag: 16/16 ready (own driver mapped)"; return 0; }
    sleep 10
  done
  log "arm $tag: ranks not ready after 900 s"
  rank_logs "$root" "$tag-notready"
  return 1
}

rank_logs() {
  local root="$1" tag="$2" r
  mkdir -p "$OUT/ranklogs"
  each_node "cp ~/$NODE_ROOT/$root/residentd.log /tmp/glmfull-scorecard-$tag.log 2>/dev/null; grep -E 'model_residentd ready|EXPERT-RESIDENCY|GLM52-CHAIN-MODE|GLM52-PROJECTION-SPLIT|GLM52-PREFILL-WAVE-ROWS|GLM52-L2-PREFETCH|GLM52-GRAPH-CAPTURE' ~/$NODE_ROOT/$root/residentd.log | head -12; echo ERRSITE \$(grep -c ERRSITE ~/$NODE_ROOT/$root/residentd.log); awk '/GLM52-CHAIN-TIME mode=graph .* rows=1 .* status=ok/{for(i=1;i<=NF;i++) if(\$i ~ /^total_us=/){split(\$i,a,\"=\"); print a[2]}}' ~/$NODE_ROOT/$root/residentd.log | sort -n | awk '{v[NR]=\$1} END{if(NR) print \"B1-CHAIN-TOTAL-MEDIAN-US\", v[int((NR+1)/2)], \"n=\" NR}'" || true
  for r in $(seq 0 15); do printf '%s: %s\n' "$(host_of "$r")" "$(tr '\n' ';' < "$OUT/node-$r.out")"; done > "$OUT/ranks-$tag.txt"
  for r in 0 15; do $SCP "$(host_of "$r"):/tmp/glmfull-scorecard-$tag.log" "$OUT/ranklogs/$tag-rank$r.log" 2>/dev/null || true; done
  each_node "rm -f /tmp/glmfull-scorecard-$tag.log" || true
  log "rank log lines for $tag in ranks-$tag.txt (ERRSITE: read include/sparkpipe/spark_status.h before judging; 15 = BUSY)"
}

stage_runner() {
  $SSH "$RUNNER_HOST" "mkdir -p $WINDOW_DIR/sessions $WINDOW_DIR/out-$STAMP && (grep -q '^/home/$RUNNER_HOST/build-glmfull-scorecard	' ~/KEEP || printf '/home/$RUNNER_HOST/build-glmfull-scorecard\tglmfull-scorecard\tKEEP\tGLM Full scorecard window runner + receipts\n' >> ~/KEEP)"
  $SCP "$HERE/runner.py" "$HERE/roofline.py" "$RUNNER_HOST:$WINDOW_DIR/"
  $SCP "$KIT"/sessions/*.json "$RUNNER_HOST:$WINDOW_DIR/sessions/"
}

session() {
  local name="$1" root="$2" tag="$3" code=0 need
  need="$(python3 -c "print(int($(est "$name") + $RESTORE_RESERVE_MIN + 0.999))")"
  if [ "$(left_min)" -lt "$need" ]; then log "SKIP $tag $name: $(left_min) min left, needs $need incl. restore"; return 2; fi
  health
  log "session $name on $root as $tag"
  local run="cd $WINDOW_DIR && python3 runner.py run sessions/$name.json $WINDOW_DIR/out-$STAMP /home/$RUNNER_HOST/$NODE_ROOT/$root/bin /home/$RUNNER_HOST/$NODE_ROOT/$root $tag"
  if [ -n "$PERF" ]; then
    $PERF "$LABEL" 20 -- $SSH "$RUNNER_HOST" "$run" >> "$OUT/sessions.log" 2>&1 || code=$?
  else
    $SSH "$RUNNER_HOST" "$run" >> "$OUT/sessions.log" 2>&1 || code=$?
  fi
  $SCP -r "$RUNNER_HOST:$WINDOW_DIR/out-$STAMP/." "$OUT/" || true
  case "$code" in
    0) log "session $name $tag PASS"; return 0 ;;
    3) log "session $name $tag INVALID (a fixed-length case stopped at EOS)"; return 0 ;;
    4) log "session $name $tag SKIPPED (arm capacity; see results.jsonl)"; return 0 ;;
    *) log "session $name $tag FAIL exit $code (see sessions.log)"; return 1 ;;
  esac
}

grade_compsec() {
  local tag="$1" wt="${SCORECARD_REPO:-$HERE/../..}"
  [ -f "$OUT/$tag-compsec17.ndjson" ] || return 0
  python3 "$wt/tools/glm53full_compsec17.py" grade --prompts "$wt/qualification/glm53full/compsec17_prompts_off.json" \
    --tokenizer "$wt/qualification/ds4_eval/tokenizer/glm-5.3-flash-tokenizer.json" --max-tokens 512 \
    --out "$OUT/compsec17-$tag-graded.json" "$OUT/$tag-compsec17.ndjson" 2>&1 | tee -a "$OUT/window.log" || log "compsec17 grading failed for $tag"
}

api_smoke_paris() {
  local out
  out="$($SSH "$HUB" "curl -s --max-time 600 http://127.0.0.1:$API_PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":16,\"temperature\":0}'")" || return 1
  echo "$out" >> "$OUT/smoke.txt"
  grep -q Paris <<<"$out"
}

restore_live() {
  local i=0 status
  log "restore: stop $UNIT, then the rotation tool's own converge glmfull (production dfa12a5 root + API)"
  unit_stop || return 1
  $SSH "$HUB" "$ROT converge glmfull" >> "$OUT/converge.out" 2>&1 || { log "restore: converge glmfull failed (see converge.out)"; return 1; }
  status="$($SSH "$HUB" "$ROT status")"
  log "restore: $status"
  while [ $i -lt 40 ]; do $SSH "$HUB" "$(rot_field '["api"]["health"]')" && break; sleep 3; i=$((i + 1)); done
  [ $i -lt 40 ] || { log "restore: API health failed"; return 1; }
  api_smoke_paris || { log "restore: Paris smoke failed"; return 1; }
  log "RESTORED: live GLM Full serving again (converge OK, API health OK, Paris PASS)"
}

stop() {
  trap - ERR
  log "STOP: $*"
  if [ "$TOUCHED" = 1 ] || [ "$API_STOPPED" = 1 ]; then
    restore_live || log "RESTORE-FAILED: run by hand: ssh $HUB '$ROT converge glmfull' (stop $UNIT on all 16 first if an arm is still up); nothing was reclaimed"
  fi
  finish_summary || true
  exit 1
}

finish_summary() {
  [ -f "$OUT/results.jsonl" ] || return 0
  python3 "$HERE/runner.py" summary "$OUT/results.jsonl" "$OUT" "GLM-5.3 Full standard window $STAMP" > /dev/null || true
  log "SUMMARY: $OUT/SUMMARY.md, $OUT/summary.json, receipts results.jsonl + *.ndjson"
}

run_arm() {
  local i="$1" tag="${ARM_TAGS[$1]}" root="${ARM_ROOTS[$1]}" envs="${ARM_ENVS[$1]}" ready="${ARM_READY[$1]}" s rc
  if [ "$i" = 0 ] && [ "$root" = root ] && [ -z "$envs" ] && [ -z "$ready" ]; then
    LIVE_RUNNING_AS_C=1
    log "arm $tag: the live release as it runs now (no restart)"
  else
    TOUCHED=1
    unit_stop || stop "could not stop $UNIT before arm $tag"
    if ! arm_up "$tag" "$root" "$envs" "$ready"; then
      log "ARM-NOT-READY $tag (rank logs saved); next arm"
      unit_stop || stop "could not stop arm $tag"
      return 0
    fi
  fi
  for s in ${ARM_SESSIONS[$i]//,/ }; do
    [ -f "$KIT/sessions/$s.json" ] || { log "arm $tag: no session $s in $KIT/sessions"; continue; }
    rc=0; session "$s" "$root" "$tag" || rc=$?
    [ "$rc" = 2 ] && continue
    if [ "$rc" != 0 ] && [ "$s" = t1 ]; then log "T1-FAIL $tag: skipping its remaining sessions"; break; fi
    if [ "$rc" != 0 ]; then
      each_node "systemctl --user is-active -q $UNIT && echo UP || echo DOWN" || true
      if grep -qx DOWN "$OUT"/node-*.out; then log "arm $tag: a rank is down after $s; next arm"; break; fi
    fi
  done
  rank_logs "$root" "$tag"
  grade_compsec "$tag"
  [ "$i" = 0 ] || python3 "$HERE/runner.py" compare "$OUT/results.jsonl" "${ARM_TAGS[0]}" "$tag" t1,b1 2>&1 | tee -a "$OUT/window.log" || log "$tag t1/b1 tokens differ from ${ARM_TAGS[0]} (see above)"
}

run_window() {
  local i
  passive || exit 1
  holder_guard || exit 1
  [ "$(plan_minutes)" -le "$MAX_MINUTES" ] || log "PLAN: estimate $(plan_minutes) min > MAX_MINUTES=$MAX_MINUTES; later sessions are skipped when time runs out"
  trap 'stop "unexpected error at line $LINENO"' ERR
  stage_runner
  api_stop
  health
  for i in "${!ARM_TAGS[@]}"; do run_arm "$i"; done
  if [ "$TOUCHED" = 1 ] || [ "$API_STOPPED" = 1 ]; then
    TOUCHED=1
    restore_live || stop "restore failed"
  fi
  trap - ERR
  finish_summary
  log "WINDOW-DONE: arms ${ARM_TAGS[*]}; $(elapsed_min) min"
}

usage() { echo "usage: $0 passive ARMS_FILE | run ARMS_FILE | restore | summary OUT_DIR" >&2; exit 2; }
case "${1:-}" in
  passive) [ -n "${2:-}" ] || usage; load_arms "$2"; passive ;;
  run) [ -n "${2:-}" ] || usage; load_arms "$2"; run_window ;;
  restore) TOUCHED=1; restore_live ;;
  summary) [ -n "${2:-}" ] || usage; python3 "$HERE/runner.py" summary "$2/results.jsonl" "$2" "GLM-5.3 Full standard window $(basename "$2")" ;;
  *) usage ;;
esac
