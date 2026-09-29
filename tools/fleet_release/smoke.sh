#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
start_log smoke
fail=0
echo "smoke $S7 $(date -u +%FT%TZ): API requests from $RELEASE_CHECKS (smoke), api.log ERRSITE since install, engine log gates (smoke_logs), node identities"
spec=$(python3 -c 'import base64, json, sys; print(base64.b64encode(json.dumps(json.load(open(sys.argv[1]))["smoke"]).encode()).decode())' "$RELEASE_CHECKS") || die "no smoke section in $RELEASE_CHECKS"
allow=$(python3 -c 'import json, sys; print("|".join(json.load(open(sys.argv[1]))["api_errsite_allow"]))' "$RELEASE_CHECKS") || die "no api_errsite_allow list in $RELEASE_CHECKS"
$RELEASE_SSH "$RELEASE_HUB" "python3 - --port $RELEASE_API_PORT --spec-b64 $spec" < "$KIT/hub_smoke.py" || fail=$((fail + 1))
echo "== api.log ERRSITE since the API install"
hub "off=\$(cat ~/$ASSEMBLE/API_LOG_OFFSET 2>/dev/null || echo 0); tail -n +\$((off + 1)) ~/$RELEASE_API_CHANNEL/api.log | grep -o 'ERRSITE [^ ]* status=[0-9]*' | sort | uniq -c | sort -rn" > "$log.api-errsite" 2>&1
cat "$log.api-errsite"
unknown=$(grep -v -E "${allow:-^$}" "$log.api-errsite" || true)
if [ -n "$unknown" ]; then echo "ERRSITE-UNKNOWN (api; read include/sparkpipe/spark_status.h before judging):"; echo "$unknown"; fail=$((fail + 1)); fi
echo "== engine logs"
python3 "$KIT/engine_logs.py" --checks "$RELEASE_CHECKS" --section smoke_logs || fail=$((fail + 1))
rules=()
for e in $RELEASE_CONVERGE_EXPECT; do [ "$e" = none ] || rules+=(--expect "$e"); done
nodes --expect eng_n=1 --expect "ready>=1" --expect exe="$NEW_RESIDENTD" --expect drv="$NEW_DRIVER" --expect wd="$NEW_WEIGHTD" ${rules[@]+"${rules[@]}"} --log "$log.nodes" | tail -n 1
[ "${PIPESTATUS[0]}" -eq 0 ] || fail=$((fail + 1))
d=$(( $(date -u +%s) - $(drain_start) ))
if [ "$fail" -eq 0 ]; then echo "SMOKE PASS $(date -u +%H:%M:%SZ) (outage so far: hold to smoke $d s). Next: rotation resume (or perf.sh first, see the runbook). log $log"; exit 0; fi
echo "SMOKE FAIL ($fail). Roll back (rollback.sh all) if the API smoke fails. log $log"
exit 1
