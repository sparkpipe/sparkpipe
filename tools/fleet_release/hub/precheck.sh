set -uo pipefail
W="$HOME/release-assemble-$S7"
S="$HOME/release/$RELEASE_ROOT_NAME"
A="$W/root"
D="$HOME/api-build-$S7"
C="$HOME/$RELEASE_API_CHANNEL"
fail=0
ok() { echo "OK   $*"; }
bad() { echo "FAIL $*"; fail=$((fail + 1)); }
want() { if [ "$2" = "$3" ]; then ok "$1=$2"; else bad "$1=$2 want $3"; fi; }
want core/WEIGHTSD_BIN "$(tr -d '[:space:]' < "$HOME/release/core/WEIGHTSD_BIN")" "$OLD_WEIGHTD"
want core/bin/sparkpipe_weightd "$(sha256sum < "$HOME/release/core/bin/sparkpipe_weightd" | cut -c1-16)" "$OLD_WEIGHTD"
want served/MANIFEST "$(sha256sum < "$S/MANIFEST" | cut -d' ' -f1)" "$SERVED_MANIFEST_SHA"
if (cd "$S" && sha256sum -c --quiet MANIFEST > /dev/null 2>&1); then ok "served root matches its MANIFEST"; else bad "served root does not match its MANIFEST"; fi
if [ -e "$S/.publish.lock" ] && command -v fuser > /dev/null && fuser "$S/.publish.lock" > /dev/null 2>&1; then bad "served .publish.lock is held by a process"; else ok "served .publish.lock not held"; fi
if cmp -s "$W/MANIFEST.served" "$S/MANIFEST"; then ok "served MANIFEST unchanged since assembly (cmp MANIFEST.served)"; else bad "served MANIFEST changed since assembly"; fi
want assembled/MANIFEST "$(sha256sum < "$A/MANIFEST" | cut -d' ' -f1)" "$NEW_MANIFEST_SHA"
if (cd "$A" && sha256sum -c --quiet MANIFEST > /dev/null 2>&1); then ok "assembled root matches its MANIFEST"; else bad "assembled root does not match its MANIFEST"; fi
want assembled/SOURCE_COMMIT "$(cat "$W/root-SOURCE_COMMIT")" "$SHA"
want assembled/residentd "$(sha256sum < "$A/bin/sparkpipe_model_residentd" | cut -c1-16)" "$NEW_RESIDENTD"
want assembled/driver "$(sha256sum < "$A/$RELEASE_DRIVER" | cut -c1-16)" "$NEW_DRIVER"
changed=$(awk 'NR==FNR{old[$2]=$1; next} old[$2]!=$1{print $2}' "$W/MANIFEST.served" "$A/MANIFEST" | tr '\n' ' ')
removed=$(awk 'NR==FNR{new[$2]=$1; next} !($2 in new){print $2}' "$A/MANIFEST" "$W/MANIFEST.served" | tr '\n' ' ')
echo "INFO changed: $changed"
echo "INFO removed-from-manifest: $removed"
want changed-count "$(echo $changed | wc -w | tr -d ' ')" "$EXPECT_CHANGED"
case " $changed " in *" model_resident.json "*|*" bin/sparkpipe_weightd "*|*" bin/sparkpipe_registrar "*) bad "assembled root changes model_resident.json or the inert weightd/registrar";; *) ok "no model_resident.json, weightd or registrar change";; esac
ncfg=$(echo "$changed" | tr ' ' '\n' | grep -c '^config/stage_[0-9][0-9]\.json$')
if [ "$STAGE_CONFIG_ADD" != none ]; then
    want changed-stage-configs "$ncfg" "$RELEASE_STAGE_WORLD"
    adds=""
    for a in $STAGE_CONFIG_ADD; do adds="$adds --add $a"; done
    if python3 "$W/stage_config_patch.py" check --current "$S" --new "$A" --world "$RELEASE_STAGE_WORLD" $adds; then ok "assembled stage configs = served + $STAGE_CONFIG_ADD"; else bad "assembled stage configs differ from served beyond $STAGE_CONFIG_ADD"; fi
else
    want changed-stage-configs "$ncfg" 0
fi
want api-build/SOURCE_COMMIT "$(cat "$D/SOURCE_COMMIT.build" 2>/dev/null)" "$SHA"
want api-build/api "$(sha256sum < "$D/build/sparkpipe_model_api" | cut -c1-16)" "$NEW_API"
want api-build/adapter "$(sha256sum < "$D/$RELEASE_ADAPTER_BUILD" | cut -c1-16)" "$NEW_ADAPTER"
if ldd -r "$D/build/sparkpipe_model_api" "$D/$RELEASE_ADAPTER_BUILD" 2>&1 | grep -Eq 'not found|undefined symbol'; then bad "api/adapter ldd -r unresolved"; else ok "api/adapter ldd -r clean"; fi
if [ "$RELEASE_API_REQUIRED_STRING" != none ]; then
    if grep -a -q -- "$RELEASE_API_REQUIRED_STRING" "$D/build/sparkpipe_model_api"; then ok "new api carries '$RELEASE_API_REQUIRED_STRING'"; else bad "new api lacks '$RELEASE_API_REQUIRED_STRING'"; fi
fi
want staged-channel/model_resident.json "$(sha256sum < "$W/channel/model_resident.json" | cut -c1-16)" "$NEW_CHANNEL_DEPLOYMENT"
want channel/api "$(sha256sum < "$C/bin/sparkpipe_model_api" | cut -c1-16)" "$OLD_API"
want channel/adapter "$(sha256sum < "$C/runtime/lib/model_serving_adapter.so" | cut -c1-16)" "$OLD_ADAPTER"
want channel/model_resident.json "$(sha256sum < "$C/model_resident.json" | cut -c1-16)" "$CHANNEL_DEPLOYMENT"
want channel/tokenizer.json "$(sha256sum < "$C/runtime/tokenizer/tokenizer.json" | cut -c1-16)" "$TOKENIZER_SHA"
if (cd "$C" && sha256sum -c --quiet SHA256SUMS > /dev/null 2>&1); then ok "channel SHA256SUMS"; else bad "channel SHA256SUMS mismatch"; fi
if python3 - "$C" "$W/channel/model_resident.json" "$CHANNEL_ADD" <<'PY'
import json, sys
live = json.load(open(sys.argv[1] + "/model_resident.json"))
new = json.load(open(sys.argv[2]))
for item in ([] if sys.argv[3] == "none" else sys.argv[3].split()):
    new.pop(item.partition("=")[0])
assert new == live
PY
then ok "staged channel deployment = live + ${CHANNEL_ADD}"; else bad "staged channel deployment differs from live beyond ${CHANNEL_ADD}"; fi
want "$RELEASE_API_UNIT" "$(systemctl --user is-active "$RELEASE_API_UNIT")" active
health=$(curl -s --max-time 10 "http://127.0.0.1:$RELEASE_API_PORT/health")
case "$health" in *'"tokenizer":true'*|*'"tokenizer": true'*) ok "health $health";; *) bad "health $health";; esac
echo "INFO hub disk: $(df -h "$HOME" | awk 'NR==2{print $4 " free of " $2}')"
echo "HUB-FAILS=$fail"
exit $((fail > 0))
