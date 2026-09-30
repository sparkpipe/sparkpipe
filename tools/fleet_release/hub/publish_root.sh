set -euo pipefail
W="$HOME/release-assemble-$S7"
A="$W/root"
S="$HOME/release/$RELEASE_ROOT_NAME"
B="$HOME/release-rollback-$(date -u +%Y%m%dT%H%M%S)-$PREV7"
exec 9>"$S/.publish.lock"
flock -n 9 || { echo lock-busy; exit 1; }
case "$(cat "$HOME/release/core/WEIGHTSD_BIN")" in "$NEW_WEIGHTD"*) ;; *) echo weightd-bin-unexpected "$(cat "$HOME/release/core/WEIGHTSD_BIN")"; exit 1;; esac
if ! (cd "$A" && sha256sum -c --quiet MANIFEST); then echo assembled-bad; exit 1; fi
[ "$(sha256sum < "$A/MANIFEST" | cut -d' ' -f1)" = "$NEW_MANIFEST_SHA" ] || { echo assembled-manifest-unexpected; exit 1; }
if ! cmp -s "$W/MANIFEST.served" "$S/MANIFEST"; then echo served-changed; exit 1; fi
[ "$(sha256sum < "$S/MANIFEST" | cut -d' ' -f1)" = "$SERVED_MANIFEST_SHA" ] || { echo served-manifest-unexpected; exit 1; }
if [ -e "$B" ]; then echo rollback-exists; exit 1; fi
changed=$(awk 'NR==FNR{old[$2]=$1; next} old[$2]!=$1{print $2}' "$S/MANIFEST" "$A/MANIFEST")
removed=$(awk 'NR==FNR{new[$2]=$1; next} !($2 in new){print $2}' "$A/MANIFEST" "$S/MANIFEST")
echo "changed:" $changed
echo "removed-from-manifest:" $removed
n=$(echo $changed | wc -w | tr -d " ")
[ "$n" -eq "$EXPECT_CHANGED" ] || { echo "changed-count-unexpected $n want $EXPECT_CHANGED"; exit 1; }
for f in $changed $removed MANIFEST; do
    if [ -e "$S/$f" ]; then mkdir -p "$(dirname "$B/$f")"; cp -p "$S/$f" "$B/$f"; fi
done
for f in $changed $removed; do
    if [ -e "$B/$f" ]; then
        want=$(awk -v p="$f" '$2==p{print $1}' "$S/MANIFEST")
        got=$(sha256sum "$B/$f" | cut -d' ' -f1)
        if [ "$want" != "$got" ]; then echo "rollback-bad $f"; exit 1; fi
    fi
done
cmp -s "$B/MANIFEST" "$S/MANIFEST" || { echo rollback-manifest-bad; exit 1; }
printf '%s\n' "$B" > "$W/ROLLBACK_DIR.new"
mv -f "$W/ROLLBACK_DIR.new" "$W/ROLLBACK_DIR"
echo rollback-saved "$B"
for f in $changed; do
    mkdir -p "$(dirname "$S/$f")"
    cp -p "$A/$f" "$S/$f.new"
    mv -f "$S/$f.new" "$S/$f"
done
cp -p "$A/MANIFEST" "$S/MANIFEST.new"
if ! (cd "$S" && sha256sum -c --quiet MANIFEST.new); then echo manifest-check-failed; rm -f "$S/MANIFEST.new"; exit 1; fi
mv -f "$S/MANIFEST.new" "$S/MANIFEST"
got=$(sha256sum < "$S/MANIFEST" | cut -d' ' -f1)
[ "$got" = "$NEW_MANIFEST_SHA" ] || { echo "published-manifest-unexpected $got"; exit 1; }
echo PUBLISHED "$(date -u +%H:%M:%SZ)" "$(printf %s "$got" | cut -c1-16)" rollback "$B"
