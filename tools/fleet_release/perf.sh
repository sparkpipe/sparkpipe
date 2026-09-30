#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
need PERF_WINDOW RELEASE_PERF_EXPECT
[ -f "$RELEASE_PERF_EXPECT" ] || { echo "RELEASE_PERF_EXPECT $RELEASE_PERF_EXPECT is not a file"; exit 2; }
R="release-perf-$S7"
ts=$(date -u +%Y%m%dT%H%M%SZ)
L="$LOGS/perf-$ts"
mkdir -p "$L"
STEP=perf
log="$L/perf.log"
exec > >(tee "$log") 2>&1
fail=0
bad() { echo "FAIL $*"; fail=$((fail + 1)); }
echo "perf $S7 $(date -u +%FT%TZ): windows lead-$S7-* from $RELEASE_CHECKS (perf); results $RELEASE_HUB:~/$R, local $L"

echo "== gate: engines, API"
nodes --expect exe="$NEW_RESIDENTD" --expect drv="$NEW_DRIVER" --expect wd="$NEW_WEIGHTD" --expect eng_n=1 --expect "ready>=1" --expect hold=no --log "$L/nodes-before.log" > /dev/null || die "fleet is not on $S7 $NODE_COUNT/$NODE_COUNT (see $L/nodes-before.log)"
others=$(grep -h "others=" "$L/nodes-before.log" | grep -v "others=none" | sed 's/^[A-Z]* *\([^ ]*\) .*others=\([^ ]*\).*/\1:\2/' | tr '\n' ' ')
if [ -n "$others" ]; then
    [ "${ALLOW_OTHERS:-0}" = 1 ] || die "other residentds run ($others); perf windows are exclusive (stop them: rotation converge to production alone, or ALLOW_OTHERS=1 and accept noisier numbers)"
    echo "WARN ALLOW_OTHERS=1: other residentds run during the perf windows ($others); numbers are noisier"
fi
api=$(hub "echo \$(sha256sum < ~/$RELEASE_API_CHANNEL/bin/sparkpipe_model_api | cut -c1-16) \$(curl -s --max-time 10 http://127.0.0.1:$RELEASE_API_PORT/health)")
case "$api" in "$NEW_API "*'"tokenizer":true'*|"$NEW_API "*'"tokenizer": true'*) echo "OK   api $api";; *) die "api/health unexpected: $api";; esac
hub "test ! -e ~/$R && mkdir -p ~/$R/results" || die "~/$R exists on the hub (move it aside for a rerun)"

expand() {
    local s=$1
    s=${s//@results@/\~/$R}
    s=${s//@build@/\~/$API_BUILD}
    s=${s//@channel@/\~/$RELEASE_API_CHANNEL}
    s=${s//@port@/$RELEASE_API_PORT}
    s=${s//@s7@/$S7}
    s=${s//@prev7@/$PREV7}
    printf '%s' "$s"
}

python3 - "$RELEASE_CHECKS" > "$L/perf-plan.tsv" <<'PY' || die "no valid perf section in $RELEASE_CHECKS"
import json, sys
perf = json.load(open(sys.argv[1]))["perf"]
for w in perf["windows"]:
    assert w["label"] and 0 < w["minutes"] <= 30 and w["command"], w
    print("W\t" + w["label"] + "\t" + str(w["minutes"]) + "\t" + w["command"])
for g in perf.get("gates", []):
    assert g["name"] and g["command"], g
    print("G\t" + g["name"] + "\t0\t" + g["command"])
for f in perf["fetch"]:
    print("F\t" + f + "\t0\t-")
PY

while IFS="$(printf '\t')" read -r kind label minutes cmd; do
    [ "$kind" = W ] || continue
    echo "== window lead-$S7-$label ($minutes min) $(date -u +%H:%M:%SZ)"
    python3 "$PERF_WINDOW" "lead-$S7-$label" "$minutes" -- $RELEASE_SSH "$RELEASE_HUB" "$(expand "$cmd")" > "$L/$label.txt" 2>&1 < /dev/null
    rc=$?
    tail -n 8 "$L/$label.txt"
    [ $rc -eq 0 ] || bad "window $label rc=$rc"
done < "$L/perf-plan.tsv"

while IFS="$(printf '\t')" read -r kind label minutes cmd; do
    [ "$kind" = G ] || continue
    echo "== gate $label"
    hub "$(expand "$cmd")" > "$L/gate-$label.txt" 2>&1 < /dev/null
    rc=$?
    cat "$L/gate-$label.txt"
    [ $rc -eq 0 ] || bad "gate $label rc=$rc"
done < "$L/perf-plan.tsv"

echo "== engine logs (read-only copies) and chain budget"
i=0
args=""
for h in $RELEASE_NODES; do
    scp -q "$h:sparkdata/$RELEASE_ROOT_NAME/residentd.log" "$L/r$i.$h.log" < /dev/null || bad "copy $h residentd.log"
    args="$args $i=$L/r$i.$h.log"
    i=$((i + 1))
done
python3 "$KIT/engine_logs.py" --checks "$RELEASE_CHECKS" --section perf_logs || bad "engine log gate (perf_logs)"
python3 "$REPO/tools/tp_chain_budget.py" $args > "$L/budget.txt" 2>&1 || echo "NOTE tp_chain_budget failed, see $L/budget.txt"
tail -n 20 "$L/budget.txt"

echo "== numbers vs expectations"
while IFS="$(printf '\t')" read -r kind label minutes cmd; do
    [ "$kind" = F ] || continue
    scp -q "$RELEASE_HUB:$R/results/$label" "$L/" < /dev/null || bad "fetch $label"
done < "$L/perf-plan.tsv"
python3 "$KIT/perf_summary.py" --expect "$RELEASE_PERF_EXPECT" --results "$L" || bad "perf numbers regress or are missing"

if [ "$fail" -eq 0 ]; then echo "PERF PASS $(date -u +%H:%M:%SZ) $L"; exit 0; fi
echo "PERF FAIL ($fail) $L"
exit 1
