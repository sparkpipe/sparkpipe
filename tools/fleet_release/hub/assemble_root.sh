set -euo pipefail
W="$HOME/release-assemble-$S7"
S="$HOME/release/$RELEASE_ROOT_NAME"
A="$W/root"
X="$W/build"
N="$W/newcfg"
T="$RELEASE_ARTIFACT.tar.gz"
cd "$W"
[ "$(sha256sum < "$T" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$T.sha256")" ] || { echo tarball-bad; exit 1; }
echo tarball OK "$(cut -c1-16 "$T.sha256")"
[ ! -e "$X" ] && [ ! -e "$A" ] || { echo assemble-dirs-exist; exit 1; }
mkdir "$X"
tar -xzf "$T" -C "$X"
(cd "$X" && sha256sum -c --quiet SHA256SUMS) || { echo build-SHA256SUMS-bad; exit 1; }
[ "$(cat "$X/SOURCE_COMMIT")" = "$SHA" ] || { echo source-commit-bad; exit 1; }
cp "$X/SOURCE_COMMIT" "$W/root-SOURCE_COMMIT"
cp -p "$S/MANIFEST" "$W/MANIFEST.served"
[ "$(sha256sum < "$W/MANIFEST.served" | cut -d' ' -f1)" = "$SERVED_MANIFEST_SHA" ] || { echo served-manifest-is-not-production; exit 1; }
(cd "$S" && sha256sum -c --quiet "$W/MANIFEST.served") || { echo served-root-does-not-match-its-MANIFEST; exit 1; }
mkdir "$A"
awk '{print $2}' "$W/MANIFEST.served" | while read -r f; do
    mkdir -p "$(dirname "$A/$f")"
    cp -p "$S/$f" "$A/$f"
done
(cd "$A" && sha256sum -c --quiet "$W/MANIFEST.served") || { echo copy-of-served-bad; exit 1; }
for g in $RELEASE_BUILD_FILES; do
    case "$g" in
        *'*'*)
            dropped=$(cd "$A" && ls $g 2>/dev/null | tr '\n' ' ' || true)
            echo "replace $g: dropping ${dropped:-nothing}"
            (cd "$A" && rm -f $g)
            files=$(cd "$X" && ls $g) || { echo "build has no $g"; exit 1; }
            ;;
        *) files=$g;;
    esac
    for f in $files; do
        [ -f "$X/$f" ] || { echo "build has no $f"; exit 1; }
        mkdir -p "$(dirname "$A/$f")"
        cp -p "$X/$f" "$A/$f"
    done
done
if [ "$STAGE_CONFIG_ADD" != none ]; then
    (cd "$N" && sha256sum -c --quiet SHA256SUMS) || { echo newcfg-SHA256SUMS-bad; exit 1; }
    adds=""
    for a in $STAGE_CONFIG_ADD; do adds="$adds --add $a"; done
    python3 "$W/stage_config_patch.py" check --current "$A" --new "$N" --world "$RELEASE_STAGE_WORLD" $adds
    for f in $(cd "$N" && ls config/stage_*.json); do
        cp -p "$N/$f" "$A/$f"
    done
fi
(cd "$A" && find lib bin stages config model_resident.json -type f ! -name stage.json ! -name MANIFEST | sort | xargs sha256sum > MANIFEST.tmp && mv MANIFEST.tmp MANIFEST)
(cd "$A" && sha256sum -c --quiet MANIFEST) && echo assembled-OK
echo "== changed vs served (sha16 served -> assembled)"
awk 'NR==FNR{old[$2]=$1; next} ($2 in old) && old[$2]!=$1{printf "  %s %s -> %s\n", $2, substr(old[$2],1,16), substr($1,1,16)} !($2 in old){printf "  %s (new) %s\n", $2, substr($1,1,16)}' "$W/MANIFEST.served" "$A/MANIFEST"
awk 'NR==FNR{new[$2]=$1; next} !($2 in new){printf "  %s removed (was %s)\n", $2, substr($1,1,16)}' "$A/MANIFEST" "$W/MANIFEST.served"
echo "changed-count $(awk 'NR==FNR{old[$2]=$1; next} old[$2]!=$1{n++} END{print n+0}' "$W/MANIFEST.served" "$A/MANIFEST")"
echo "residentd $(sha256sum < "$A/bin/sparkpipe_model_residentd" | cut -c1-16) driver $(sha256sum < "$A/$RELEASE_DRIVER" | cut -c1-16)"
echo MANIFEST "$(sha256sum "$A/MANIFEST" | cut -d' ' -f1)"
echo served-MANIFEST "$(sha256sum "$W/MANIFEST.served" | cut -d' ' -f1)"
