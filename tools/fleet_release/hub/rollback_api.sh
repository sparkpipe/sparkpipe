set -euo pipefail
W="$HOME/release-assemble-$S7"
C="$HOME/$RELEASE_API_CHANNEL"
sha16() { sha256sum < "$1" | cut -c1-16; }
production() {
    [ "$(sha16 "$1/bin/sparkpipe_model_api")" = "$OLD_API" ] &&
    [ "$(sha16 "$1/runtime/lib/model_serving_adapter.so")" = "$OLD_ADAPTER" ] &&
    [ "$(sha16 "$1/model_resident.json")" = "$CHANNEL_DEPLOYMENT" ] &&
    (cd "$1" && sha256sum -c --quiet SHA256SUMS > /dev/null 2>&1)
}
health() {
    h=""
    for i in $(seq 1 60); do
        h=$(curl -s --max-time 5 "http://127.0.0.1:$RELEASE_API_PORT/health" || true)
        case "$h" in *'"tokenizer":true'*|*'"tokenizer": true'*) break;; esac
        sleep 2
    done
    echo "$RELEASE_API_UNIT $(systemctl --user is-active "$RELEASE_API_UNIT") health $h"
    case "$h" in *'"tokenizer":true'*|*'"tokenizer": true'*) return 0;; *) return 1;; esac
}
if production "$C"; then
    echo "api already production: api $OLD_API adapter $OLD_ADAPTER deployment $CHANNEL_DEPLOYMENT, SHA256SUMS OK"
    systemctl --user is-active --quiet "$RELEASE_API_UNIT" || systemctl --user start "$RELEASE_API_UNIT"
    health && { echo API-ALREADY-PRODUCTION; exit 0; }
    echo API-NOT-HEALTHY; exit 1
fi
[ -f "$W/API_BACKUP_DIR" ] || { echo "channel is not production and no api backup is recorded; stop"; exit 1; }
bak=$(cat "$W/API_BACKUP_DIR")
[ -d "$bak" ] || { echo "backup missing $bak"; exit 1; }
production "$bak" || { echo "backup $bak is not api $OLD_API + adapter $OLD_ADAPTER + deployment $CHANNEL_DEPLOYMENT with valid SHA256SUMS"; exit 1; }
failed="$HOME/$RELEASE_API_CHANNEL.failed-$S7-$(date -u +%Y%m%dT%H%M%S)"
[ ! -e "$failed" ] || { echo failed-dir-exists; exit 1; }
systemctl --user stop "$RELEASE_API_UNIT"
mv "$C" "$failed"
cp -a "$bak" "$C"
production "$C" || { echo "restored channel does not verify; failed channel kept at $failed"; exit 1; }
wc -l < "$C/api.log" > "$W/API_LOG_OFFSET"
systemctl --user start "$RELEASE_API_UNIT"
ok=0; health || ok=1
cut -c1-16,65- "$C/SHA256SUMS"
echo "failed channel kept at $failed"
[ $ok -eq 0 ] && echo API-ROLLED-BACK || { echo API-NOT-HEALTHY; exit 1; }
