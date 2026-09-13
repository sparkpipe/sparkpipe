#!/usr/bin/env bash
# strip_k3_verify.sh — strip comments from K3 sources and PROVE token-stream
# equivalence via preprocessed-output comparison (pre vs post, whitespace-
# normalized, line markers dropped).
set -uo pipefail
REPO=/Users/mac/dsh.sparkpipe
SRC="$REPO/modules/k3_resident_decode_stage/source"
BACKUP=/tmp/k3-strip-backup
STUB=/tmp/k3-cuda-stub
OUT=/tmp/k3-strip-check
mkdir -p "$BACKUP" "$STUB" "$OUT"
[ -f "$STUB/cuda_runtime.h" ] || : > "$STUB/cuda_runtime.h"

norm() {  # $1 = pp output file -> normalized token stream on stdout
    grep -v '^#' "$1" | tr -s '[:space:]' ' ' | sed 's/^ //; s/ $//'
}

cd "$REPO"
for f in "$SRC"/*.c; do
    base=$(basename "$f")
    cp "$f" "$BACKUP/$base.orig"
done

snapshot() {  # phase = pre|post
    local phase="$1"
    for f in "$SRC"/*.c; do
        base=$(basename "$f")
        cc -E -P -I"$STUB" -I"$REPO/include" -I"$REPO" -I"$REPO/.agents/k3/src" -Imodules/k3_resident_decode_stage/include "$f" > "$OUT/$base.$phase.pp" 2>"$OUT/$base.$phase.err"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "PPFAIL $base ($phase)"
            return 1
        fi
        norm "$OUT/$base.$phase.pp" > "$OUT/$base.$phase.norm"
    done
    return 0
}

snapshot pre || exit 1
echo "-- pre snapshots ok --"

python3 "$REPO/tools/devcycle/matrix/strip_c_comments.py" "$SRC"/*.c || exit 1

snapshot post || exit 1
echo "-- post snapshots ok --"

fails=0
for f in "$SRC"/*.c; do
    base=$(basename "$f")
    if cmp -s "$OUT/$base.pre.norm" "$OUT/$base.post.norm"; then
        echo "EQUIVALENT  $base"
    else
        echo "MISMATCH    $base"
        diff <(tr ' ' '\n' < "$OUT/$base.pre.norm")              <(tr ' ' '\n' < "$OUT/$base.post.norm") | head -5
        fails=$((fails+1))
    fi
done
echo "== equivalence fails=$fails =="
exit $fails
