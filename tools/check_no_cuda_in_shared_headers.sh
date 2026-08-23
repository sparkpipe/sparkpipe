#!/bin/sh
# check_no_cuda_in_shared_headers.sh
#
# ENFORCEMENT ARM of the hwiface_v1_freeze §F4 "landmine rule" extended per the
# r2c2 review (.agents/coord/hwiface_pccore_review_r2.md R2-C2): no header that
# a ROCm (gfx950) translation unit can consume may evaluate CUDA-only
# constructs — cudaStream_t, cudaEvent_t, __global__, or <<< >>> launch syntax
# — outside an explicit __CUDACC__/__CUDA_ARCH__ conditional guard. A ROCm
# build that silently took CUDA's stream/event/tile machinery would be a
# correctness + performance hazard the linker cannot catch; this gate catches
# it instead, so the gfx950 port cannot silently regress the freeze.
#
# Scope decision (deliberate superset): the set of headers reachable from ROCm
# TUs changes every port (2-4) and cannot be trusted to stay inventoried, so we
# scan EVERY tracked .h/.cuh and exempt only paths listed in
# tools/check_no_cuda_shared_headers_baseline.txt:
#   [allow]         CUDA-by-design files (kernel .cuh chains, host-side stubs)
#                   that no ROCm TU may include unguarded anyway;
#   [pending-r2c2]  known leaks that must vanish when the r2c2 CUDA-guard split
#                   of spark_stage_module_common.h lands; such entries go
#                   STALE-fail once fixed, forcing the baseline to shrink.
# Any leak in any other header fails loudly as NEW LEAK with path:line:token.
#
# Headers whose CUDA tokens live entirely inside #if(def) __CUDACC__ /
# __CUDA_ARCH__ regions are recognized automatically: the tracker below walks
# #if/#ifdef/#ifndef/#elif/#else/#endif with a per-level state stack
# (1 = inside the CUDA-taken arm -> skip, 2 = complementary arm -> scan,
# 0 = unrelated condition -> conservatively scan both arms). C comments are
# stripped before matching, so prose like "// the <<< >>> syntax" does not trip.
#
# Usage:
#   tools/check_no_cuda_in_shared_headers.sh          # gate: exit 0 = clean
#   tools/check_no_cuda_in_shared_headers.sh --list   # raw hits, exit 0 always
#   SCAN_UNIVERSE=1 tools/...                         # include untracked files
#
# Known limitations (honest): string literals are not parsed, so a "//" or
# "/*" inside one could truncate that line's scan (false-negative risk only);
# the token set is exactly the four freeze-named constructs.

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "FATAL: not inside a git repository (run from repo root)" >&2
    exit 2
}
BASELINE="$SCRIPT_DIR/check_no_cuda_shared_headers_baseline.txt"
TOKEN_RE='cudaStream_t|cudaEvent_t|__global__|<<<'

LIST_ONLY=0
[ "${1:-}" = "--list" ] && LIST_ONLY=1
[ "$LIST_ONLY" = 1 ] || [ -f "$BASELINE" ] || {
    echo "FATAL: baseline missing: $BASELINE" >&2
    exit 2
}

# ---------------------------------------------------------------------------
# 1. File universe: tracked headers (build/, tmp/ scratch trees are untracked,
#    hence excluded automatically).
# ---------------------------------------------------------------------------
if [ "${SCAN_UNIVERSE:-0}" = "1" ]; then
    UNIVERSE=$(find "$ROOT" \( -name '*.h' -o -name '*.cuh' \) \
               -not -path '*/.git/*' | sed "s|^$ROOT/||")
else
    UNIVERSE=$(git -C "$ROOT" ls-files '*.h' '*.cuh')
fi

# ---------------------------------------------------------------------------
# 2. Scan: strip C comments, track CUDA conditional-compilation state, report
#    unguarded token hits as <path>:<line>:<token>.
# ---------------------------------------------------------------------------
HITS=$(printf '%s\n' "$UNIVERSE" | awk -v root="$ROOT" -v tre="$TOKEN_RE" '
{
    file = $0; path = root "/" file
    incom = 0; d = 0; ln = 0
    while ((getline line < path) > 0) {
        ln++
        out = ""; i = 1; n = length(line)
        while (i <= n) {
            two = substr(line, i, 2)
            if (incom)            { if (two == "*/") { incom = 0; i += 2 } else i++ }
            else if (two == "/*") { incom = 1; i += 2 }
            else if (two == "//") break
            else { out = out substr(line, i, 1); i++ }
        }
        t = out; sub(/^[ \t]+/, "", t)
        if (t ~ /^#/) {
            directive = t; sub(/^#[ \t]*/, "", directive); sub(/[ \t].*$/, "", directive)
            cond = t;      sub(/^#[ \t]*[A-Za-z]+[ \t]*/, "", cond)
            hascuda = (cond ~ /__CUDACC__|__CUDA_ARCH__/)
            negated = (cond ~ /^![ \t]*defined[ \t]*\([ \t]*(__CUDACC__|__CUDA_ARCH__)/) || \
                      (cond ~ /^![ \t]*(__CUDACC__|__CUDA_ARCH__)($|[) \t])/)
            if (directive == "if" || directive == "ifdef" || directive == "ifndef") {
                d++
                if (directive == "ifndef")   st[d] = hascuda ? 2 : 0
                else if (hascuda && negated) st[d] = 2
                else if (hascuda)            st[d] = 1
                else                         st[d] = 0
            } else if (directive == "elif") {
                if (d > 0) {
                    cur = st[d]
                    if (cur == 0)                 st[d] = 0     # unknown: stay conservative
                    else if (hascuda && !negated) st[d] = 1     # elif arm takes CUDA
                    else                          st[d] = (cur == 1) ? 2 : cur
                }
            } else if (directive == "else") {
                if (d > 0 && st[d] != 0) st[d] = (st[d] == 1) ? 2 : 1
            } else if (directive == "endif") {
                if (d > 0) d--
            }
        }
        incuda = (d > 0 && st[d] == 1)
        if (!incuda && match(out, tre))
            printf "%s:%d:%s\n", file, ln, substr(out, RSTART, RLENGTH)
    }
    close(path)
}')

if [ "$LIST_ONLY" = "1" ]; then
    printf '%s\n' "$HITS"
    exit 0
fi

LEAKS=$(printf '%s\n' "$HITS" | cut -d: -f1 | sort -u | grep -v '^$')

# ---------------------------------------------------------------------------
# 3. Baseline parse: "[allow]|[pending-r2c2]" <path> [@MISSING-OK] [# reason]
# ---------------------------------------------------------------------------
ALLOWED=""; PENDING=""; MISSING_OK=""; seen=""
while IFS= read -r bl; do
    case "$bl" in ''|\#*) continue ;; esac
    key=${bl%% *}
    rest=${bl#* }
    path=${rest%%[#@]*}
    while [ -n "$path" ] && [ "${path%"${path%?}"}" = " " ]; do path=${path%?}; done
    [ -n "$path" ] || { echo "BASELINE ERROR: empty path in line: $bl" >&2; exit 2; }
    case "$seen" in *"|$key $path|"*)
        echo "BASELINE ERROR: duplicate entry [$key] $path" >&2; exit 2 ;; esac
    seen="$seen|$key $path|"
    mo=0; case "$bl" in *@MISSING-OK*) mo=1 ;; esac
    case "$key" in
        '[allow]')
            ALLOWED="$ALLOWED $path "
            [ "$mo" = 1 ] && MISSING_OK="$MISSING_OK $path " ;;
        '[pending-r2c2]')
            PENDING="$PENDING $path "
            [ "$mo" = 1 ] && MISSING_OK="$MISSING_OK $path " ;;
        *) echo "BASELINE ERROR: unknown section '$key' in line: $bl" >&2; exit 2 ;;
    esac
done < "$BASELINE"

fail=0

# --- dangling baseline entries ---------------------------------------------
for p in $ALLOWED $PENDING; do
    case " $MISSING_OK " in *" $p "*) continue ;; esac
    git -C "$ROOT" ls-files --error-unmatch "$p" >/dev/null 2>&1 || {
        echo "STALE BASELINE: '$p' listed but absent from tree — prune it." >&2
        fail=1
    }
done

# --- NEW LEAK: hit outside the baseline --------------------------------------
for p in $LEAKS; do
    case " $ALLOWED $PENDING " in *" $p "*) ;; *)
        echo "NEW LEAK (freeze F4 landmine rule): $p" >&2
        printf '%s\n' "$HITS" | grep "^$p:" | sed 's/^/    /' >&2
        fail=1 ;;
    esac
done

# --- STALE pending entry: leak gone, baseline must shrink --------------------
for p in $PENDING; do
    printf '%s\n' "$LEAKS" | grep -qxF "$p" || {
        echo "STALE BASELINE: '[pending-r2c2] $p' no longer leaks (r2c2 landed?) — remove the entry." >&2
        fail=1
    }
done

nfiles=$(printf '%s\n' "$UNIVERSE" | grep -c .)
nhits=$(printf '%s\n' "$HITS" | grep -c .)
nleaks=$(printf '%s\n' "$LEAKS" | grep -c .)
if [ "$fail" != 0 ]; then
    echo "check_no_cuda_in_shared_headers: $nfiles headers scanned, $nhits unguarded hits in $nleaks files — gate FAILED above." >&2
    echo "FAIL: freeze F4 landmine rule violated (new leak or stale baseline above)." >&2
    exit 1
fi
echo "check_no_cuda_in_shared_headers: $nfiles headers scanned, $nhits unguarded hits in $nleaks files, all within baseline."
exit 0
