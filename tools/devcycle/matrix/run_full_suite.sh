#!/usr/bin/env bash
# run_full_suite.sh — build and run EVERY test on current HEAD without stopping
# at the first failure. Produces /tmp/head-test-suite/report.txt + per-test logs.
set -u
cd /Users/mac/dsh.sparkpipe
OUT=/tmp/head-test-suite
mkdir -p "$OUT"
: > "$OUT/report.txt"

echo "== HEAD $(git rev-parse HEAD) ==" | tee -a "$OUT/report.txt"
echo "== dirty files: $(git status --porcelain | wc -l | tr -d ' ') ==" | tee -a "$OUT/report.txt"

# ---- resolve exact test lists through make itself --------------------------
BINARIES=$(make -pn test 2>/dev/null | grep '^TEST_BINARIES' | head -1 | cut -d= -f2-)
PYTESTS=$(make -pn test 2>/dev/null | grep '^PYTHON_TESTS' | head -1 | cut -d= -f2-)

echo "== building all test binaries (-k so one bad build does not stop the rest) ==" | tee -a "$OUT/report.txt"
make -k -j8 $BINARIES > "$OUT/build.log" 2>&1
echo "build rc=$?" | tee -a "$OUT/report.txt"
grep -E 'Error [0-9]+$|error:' "$OUT/build.log" | sort -u > "$OUT/build-errors.txt"
wc -l < "$OUT/build-errors.txt" | xargs echo "distinct build error lines:" | tee -a "$OUT/report.txt"

# ---- run every binary that exists -------------------------------------------
pass=0; fail=0; missing=0
for b in $BINARIES; do
  name=$(basename "$b")
  if [ ! -x "$b" ]; then
    echo "MISSING  $name (build failed)" | tee -a "$OUT/report.txt"
    missing=$((missing+1))
    continue
  fi
  timeout 180 ./"$b" > "$OUT/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    tail_line=$(grep -E 'FAIL|Assertion|assert|error' "$OUT/$name.log" | head -1 | cut -c1-160)
    echo "FAIL rc=$rc  $name   ${tail_line}" | tee -a "$OUT/report.txt"
  fi
done

# ---- python tests ------------------------------------------------------------
for t in $PYTESTS; do
  name=$(basename "$t")
  timeout 300 python3 "$t" > "$OUT/$name.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
    tail_line=$(tail -1 "$OUT/$name.log" | cut -c1-160)
    echo "FAIL rc=$rc  $name   ${tail_line}" | tee -a "$OUT/report.txt"
  fi
done

echo "== SUMMARY pass=$pass fail=$fail missing=$missing ==" | tee -a "$OUT/report.txt"
