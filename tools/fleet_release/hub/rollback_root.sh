set -euo pipefail
W="$HOME/release-assemble-$S7"
S="$HOME/release/$RELEASE_ROOT_NAME"
exec 9>"$S/.publish.lock"
flock -n 9 || { echo lock-busy; exit 1; }
cur=$(sha256sum < "$S/MANIFEST" | cut -d' ' -f1)
if [ "$cur" = "$SERVED_MANIFEST_SHA" ] && (cd "$S" && sha256sum -c --quiet MANIFEST > /dev/null 2>&1); then
    echo "root already production ${SERVED_MANIFEST_SHA:0:16} (MANIFEST and every file match)"
    exit 0
fi
case "$cur" in
    "$SERVED_MANIFEST_SHA") echo "served MANIFEST is production but some files are not (partial publish): restoring files";;
    "$NEW_MANIFEST_SHA") ;;
    *) echo "served MANIFEST is ${cur:0:16}, neither the release nor production; stop"; exit 1;;
esac
[ -f "$W/ROLLBACK_DIR" ] || { echo no-rollback-dir-recorded; exit 1; }
B=$(cat "$W/ROLLBACK_DIR")
[ -d "$B" ] || { echo "rollback dir missing $B"; exit 1; }
[ "$(sha256sum < "$B/MANIFEST" | cut -d' ' -f1)" = "$SERVED_MANIFEST_SHA" ] || { echo rollback-manifest-unexpected; exit 1; }
files=$(cd "$B" && find . -type f ! -name MANIFEST | sed 's|^\./||' | sort)
[ -n "$files" ] || { echo rollback-dir-empty; exit 1; }
echo "restoring from $B:" $files
for f in $files; do
    want=$(awk -v p="$f" '$2==p{print $1}' "$B/MANIFEST")
    got=$(sha256sum "$B/$f" | cut -d' ' -f1)
    [ -n "$want" ] && [ "$want" = "$got" ] || { echo "rollback-copy-bad $f"; exit 1; }
done
for f in $files; do
    cp -p "$B/$f" "$S/$f.new"
    mv -f "$S/$f.new" "$S/$f"
done
if [ "$cur" != "$SERVED_MANIFEST_SHA" ]; then
    cp -p "$B/MANIFEST" "$S/MANIFEST.new"
    if ! (cd "$S" && sha256sum -c --quiet MANIFEST.new); then echo manifest-check-failed; rm -f "$S/MANIFEST.new"; exit 1; fi
    mv -f "$S/MANIFEST.new" "$S/MANIFEST"
fi
(cd "$S" && sha256sum -c --quiet MANIFEST) || { echo served-root-does-not-match-MANIFEST-after-rollback; exit 1; }
got=$(sha256sum < "$S/MANIFEST" | cut -d' ' -f1)
[ "$got" = "$SERVED_MANIFEST_SHA" ] || { echo "rolled-back-manifest-unexpected $got"; exit 1; }
echo ROOT-ROLLED-BACK "$(date -u +%H:%M:%SZ)" "${got:0:16}"
