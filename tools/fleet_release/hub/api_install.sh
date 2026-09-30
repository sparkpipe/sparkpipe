set -euo pipefail
W="$HOME/release-assemble-$S7"
B="$HOME/api-build-$S7"
C="$HOME/$RELEASE_API_CHANNEL"
api="$B/build/sparkpipe_model_api"
adapter="$B/$RELEASE_ADAPTER_BUILD"
dep="$W/channel/model_resident.json"
case "$(sha256sum "$api" | cut -c1-16)" in "$NEW_API") ;; *) echo api-sha-unexpected; exit 1;; esac
case "$(sha256sum "$adapter" | cut -c1-16)" in "$NEW_ADAPTER") ;; *) echo adapter-sha-unexpected; exit 1;; esac
case "$(sha256sum "$dep" | cut -c1-16)" in "$NEW_CHANNEL_DEPLOYMENT") ;; *) echo staged-channel-deployment-unexpected; exit 1;; esac
if [ "$RELEASE_API_REQUIRED_STRING" != none ]; then
    grep -a -q -- "$RELEASE_API_REQUIRED_STRING" "$api" || { echo "api-lacks-required-string $RELEASE_API_REQUIRED_STRING"; exit 1; }
fi
case "$(sha256sum "$C/bin/sparkpipe_model_api" | cut -c1-16)" in "$OLD_API") ;; *) echo channel-api-not-production; exit 1;; esac
case "$(sha256sum "$C/runtime/lib/model_serving_adapter.so" | cut -c1-16)" in "$OLD_ADAPTER") ;; *) echo channel-adapter-not-production; exit 1;; esac
case "$(sha256sum "$C/model_resident.json" | cut -c1-16)" in "$CHANNEL_DEPLOYMENT") ;; *) echo channel-deployment-unexpected; exit 1;; esac
(cd "$C" && sha256sum -c --quiet SHA256SUMS) || { echo channel-SHA256SUMS-bad; exit 1; }
python3 - "$C" "$dep" "$CHANNEL_ADD" <<'PY'
import hashlib, json, sys
c, staged, spec = sys.argv[1:4]
live = json.load(open(c + "/model_resident.json"))
new = json.load(open(staged))
t = new.get("tokenizer")
assert t and t == live.get("tokenizer"), "tokenizer block"
data = open(c + "/runtime/" + t["path"], "rb").read()
assert hashlib.sha256(data).hexdigest() == t["sha256"], "tokenizer sha"
doc = json.loads(data)
ids = list(doc["model"]["vocab"].values()) + [x["id"] for x in doc.get("added_tokens", [])]
assert max(ids) + 1 == t["vocabulary_size"], "vocabulary_size"
keys = [] if spec == "none" else [item.partition("=")[0] for item in spec.split()]
for k in keys:
    new.pop(k)
assert new == live, "staged deployment differs from the live one beyond " + (" ".join(keys) or "nothing")
print("tokenizer block OK", t["path"], t["sha256"][:16], t["vocabulary_size"], "; added", " ".join(keys) or "nothing")
PY
bak="$HOME/$RELEASE_API_CHANNEL.bak-$PREV7-$(date -u +%Y%m%dT%H%M%S)"
if [ -e "$bak" ]; then echo backup-exists; exit 1; fi
cp -a "$C" "$bak"
printf '%s\n' "$bak" > "$W/API_BACKUP_DIR.new"
mv -f "$W/API_BACKUP_DIR.new" "$W/API_BACKUP_DIR"
cp -p "$api" "$C/bin/sparkpipe_model_api.new"
cp -p "$adapter" "$C/runtime/lib/model_serving_adapter.so.new"
cp -p "$dep" "$C/model_resident.json.new"
printf '%s\n' "$SHA" > "$C/SOURCE_COMMIT.new"
mv -f "$C/bin/sparkpipe_model_api.new" "$C/bin/sparkpipe_model_api"
mv -f "$C/runtime/lib/model_serving_adapter.so.new" "$C/runtime/lib/model_serving_adapter.so"
mv -f "$C/model_resident.json.new" "$C/model_resident.json"
mv -f "$C/SOURCE_COMMIT.new" "$C/SOURCE_COMMIT"
(cd "$C" && sha256sum bin/sparkpipe_model_api runtime/lib/model_serving_adapter.so runtime/tokenizer/tokenizer.json model_resident.json SOURCE_COMMIT > SHA256SUMS && sha256sum -c --quiet SHA256SUMS)
cut -c1-16,65- "$C/SHA256SUMS"
wc -l < "$C/api.log" > "$W/API_LOG_OFFSET"
systemctl --user restart "$RELEASE_API_UNIT"
h=""
for i in $(seq 1 60); do
    h=$(curl -s --max-time 5 "http://127.0.0.1:$RELEASE_API_PORT/health" || true)
    case "$h" in *'"tokenizer":true'*|*'"tokenizer": true'*) break;; esac
    sleep 2
done
echo "$RELEASE_API_UNIT $(systemctl --user is-active "$RELEASE_API_UNIT") health $h"
off=$(cat "$W/API_LOG_OFFSET")
tail -n +"$((off + 1))" "$C/api.log" | grep -E "tokenizer sidecar|chat_template|REFUSED|FAILED|refusing|no tokenizer" | cut -c1-200 || true
tail -n 3 "$C/api.log" | cut -c1-160
echo backup "$bak"
case "$h" in *'"tokenizer":true'*|*'"tokenizer": true'*) ;; *) echo API-NOT-HEALTHY; exit 1;; esac
if [ "$RELEASE_API_REQUIRED_LOG" != none ]; then
    IFS='|'
    for line in $RELEASE_API_REQUIRED_LOG; do
        n=$(tail -n +"$((off + 1))" "$C/api.log" | grep -c -F -- "$line" || true)
        [ "${n:-0}" -ge 1 ] || { echo "API-LOG-MISSING $line"; exit 1; }
    done
    unset IFS
fi
case "$(sha256sum < "$C/bin/sparkpipe_model_api" | cut -c1-16) $(sha256sum < "$C/runtime/lib/model_serving_adapter.so" | cut -c1-16) $(sha256sum < "$C/model_resident.json" | cut -c1-16)" in
    "$NEW_API $NEW_ADAPTER $NEW_CHANNEL_DEPLOYMENT") echo API-INSTALLED;;
    *) echo API-IDENTITY-UNEXPECTED; exit 1;;
esac
