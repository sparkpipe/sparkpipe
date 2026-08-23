#!/bin/sh
# run_prefix_cache_full_gate.sh - pccore lane lesson 6 mechanized:
#   "re-run the FULL gate after any merge touching landed files."
#
# What this script guarantees (exit nonzero on any violation):
#   1. NO STALE-BINARY SHORTCUT: it removes both gate binaries first, forces
#      make to rebuild them from scratch, and asserts each binary was actually
#      recreated AFTER the run started (newer than a start marker).
#   2. BOTH GATES PASS: build/test_prefix_cache_core AND
#      build/test_qwen36_prefix_cache must print their terminal PASS lines.
#   3. ZERO COUNTER DRIFT: measured counters/digest counts are normalized to
#      KEY=VALUE lines and diffed against the committed baseline
#      (tools/prefix_cache_full_gate_baseline.txt). Any addition, removal or
#      value change fails the gate. Baseline drift must be a deliberate,
#      reviewed re-baseline (edit the baseline file in the same commit).
#
# Usage:
#   sh tools/run_prefix_cache_full_gate.sh            # from repo root
# Env overrides:
#   PCCORE_GATE_BASELINE  path to baseline file (default: committed one)
#   PCCORE_GATE_OUTDIR    where logs/measured files go (default: build/gate)
#   TMPDIR                respected as-is (sandboxed boxes: keep inside workspace)
set -u

fail() { printf 'FULL-GATE FAIL: %s\n' "$*" >&2; exit 1; }

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || fail "cannot locate repo root"
cd "$REPO_ROOT" || fail "cannot cd to repo root"

CORE_BIN=build/test_prefix_cache_core
Q36_BIN=build/test_qwen36_prefix_cache
BASELINE=${PCCORE_GATE_BASELINE:-tools/prefix_cache_full_gate_baseline.txt}
OUTDIR=${PCCORE_GATE_OUTDIR:-build/gate}
MARKER="$OUTDIR/rebuild.marker"

[ -f "$BASELINE" ] || fail "baseline file missing: $BASELINE"
[ -f Makefile ] || fail "run from a checkout with the root Makefile"
command -v make >/dev/null || fail "make not found"

mkdir -p "$OUTDIR" || fail "cannot create $OUTDIR"
: > "$MARKER" || fail "cannot write start marker $MARKER"
# Marker must be STRICTLY older than anything built afterwards, even within
# the same wall-clock second (mtime granularity).
sleep 1

echo "== prefix-cache FULL gate (from-scratch rebuild + counter diff) =="

# --- 1. force rebuild: rm-first so no prebuilt binary can answer -------------
rm -f "$CORE_BIN" "$Q36_BIN"
echo "-- removing both gate binaries; rebuilding dependency closure --"
if ! make "$CORE_BIN" "$Q36_BIN" >"$OUTDIR/rebuild.log" 2>&1; then
    tail -40 "$OUTDIR/rebuild.log" >&2
    fail "rebuild failed (see $OUTDIR/rebuild.log)"
fi

# --- stale-binary shortcut tripwire ------------------------------------------
for bin in "$CORE_BIN" "$Q36_BIN"; do
    [ -x "$bin" ] || fail "binary absent after rebuild: $bin"
    [ "$bin" -nt "$MARKER" ] ||
        fail "STALE-BINARY SHORTCUT: $bin older than this run (was not rebuilt)"
done
echo "-- rebuild OK: both binaries recreated after run start --"

# --- 2. run both gates --------------------------------------------------------
TMPDIR=${TMPDIR:-$OUTDIR} sh -c "'./$CORE_BIN'" >"$OUTDIR/core.out" 2>&1
CORE_RC=$?
TMPDIR=${TMPDIR:-$OUTDIR} sh -c "'./$Q36_BIN'" >"$OUTDIR/qwen36.out" 2>&1
Q36_RC=$?
[ "$CORE_RC" -eq 0 ] || { tail -20 "$OUTDIR/core.out" >&2; fail "core gate exited $CORE_RC (see $OUTDIR/core.out)"; }
[ "$Q36_RC" -eq 0 ] || { tail -20 "$OUTDIR/qwen36.out" >&2; fail "qwen36 gate exited $Q36_RC (see $OUTDIR/qwen36.out)"; }

grep -q '^prefix-cache-core PASS ' "$OUTDIR/core.out" ||
    fail "core gate missing terminal 'prefix-cache-core PASS' line"
grep -q '^qwen36 prefix-cache gate PASS' "$OUTDIR/qwen36.out" ||
    fail "qwen36 gate missing terminal PASS line"
grep -q '^f1_conservation PASS' "$OUTDIR/qwen36.out" ||
    fail "qwen36 gate missing f1_conservation PASS"

# --- 3. normalize measured counters and diff against baseline ----------------
{
    # core gate: completeness-matrix cells + spec-conservation + digest totals
    awk '
        /^spec-conservation/ {
            key = "spec_" $2; gsub(/[ \t]/, "", key);
            for (i=1;i<=NF;i++) if ($i ~ /^(pool|peak_blocks|evicted)=/) print key "_" $i;
        }
        /^B[0-9]+/ {
            key = "cell_" $1 "_" $2; gsub(/[ \t]/, "", key);
            for (i=1;i<=NF;i++) if ($i ~ /^(rows|stream|matched_blocks|evicted|stalls)=/) print key "_" $i;
        }
        /^prefix-cache-core PASS/ { for (i=2;i<=NF;i++) if ($i ~ /=/) print $i; }
    ' "$OUTDIR/core.out" | tr 'A-Z' 'a-z'
    # qwen36 gate: pressure, speculation, conservation feasibility
    awk '
        /^b25_pressure/ {
            for (i=2;i<=NF;i++) if ($i ~ /=/) print "b25_" $i;
        }
        /^spec_decode ok=/ {
            match($0, /ok=[0-9]+/);
            print "spec_decode_" substr($0, RSTART, RLENGTH);
        }
        /^f1_conservation reuse-on:/ {
            match($0, /max attached=[0-9]+/); print "f1_on_" substr($0, RSTART, RLENGTH);
            match($0, /final free=[0-9]+/);   print "f1_on_" substr($0, RSTART, RLENGTH);
        }
        /^f1_conservation reuse-off:/ {
            match($0, /max attached=[0-9]+/); print "f1_off_" substr($0, RSTART, RLENGTH);
            match($0, /final free=[0-9]+/);   print "f1_off_" substr($0, RSTART, RLENGTH);
        }
    ' "$OUTDIR/qwen36.out"
} | LC_ALL=C sort > "$OUTDIR/measured.txt"

LC_ALL=C sort "$BASELINE" > "$OUTDIR/baseline.sorted"
if ! diff -u "$OUTDIR/baseline.sorted" "$OUTDIR/measured.txt" > "$OUTDIR/counter_diff.txt"; then
    cat "$OUTDIR/counter_diff.txt" >&2
    fail "MEASURED COUNTER DRIFT vs baseline $BASELINE (diff in $OUTDIR/counter_diff.txt). If intentional, re-baseline deliberately."
fi

echo "-- counters identical to baseline ($(wc -l < "$OUTDIR/measured.txt" | tr -d ' ') keys) --"
echo "prefix-cache FULL gate PASS (rebuild verified, both gates green, zero counter drift)"
