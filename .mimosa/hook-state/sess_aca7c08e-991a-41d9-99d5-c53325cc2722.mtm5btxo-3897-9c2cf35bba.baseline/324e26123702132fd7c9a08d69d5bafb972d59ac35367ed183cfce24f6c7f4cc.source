#!/bin/sh
# Repository metric. Code, and everything that is not code.
#
# solutions/codesize squared is only meaningful if codesize means code. Two
# thirds of this repository's FILES are captured tensor dumps from past
# debugging runs; counting them made every ratio meaningless and made deletion
# look like progress when it was housekeeping.
#
# So: three buckets, reported separately and never summed.
#
#   CODE          what ships. This is the denominator.
#   TESTS         what proves it. Tracked apart because the two trade off: a
#                 kernel whose contract is a static_assert needs no test, so
#                 test lines falling can mean the code got safer, not thinner.
#                 Counting them together hides that.
#   DOCS          prose and measurements. Grows with understanding, not debt.
#   DIAGNOSTICS   captured tensors and logs. Evidence for measured claims.
#                 Should live outside git, but the claims need it until it does.
#
# Diagnostics are not waste. This codebase's failure has been claims without
# evidence, and the handful of measured numbers in docs/ cite these captures.
# Deleting them would remove the evidence for the only trustworthy claims here.

cd "$(dirname "$0")/.." || exit 1

. "$(dirname "$0")/classify.sh"
spark_classified /tmp/lm_metric.txt

report() {
	n=$(grep -c "^$1 " /tmp/lm_metric.txt)
	l=$(grep "^$1 " /tmp/lm_metric.txt | cut -d' ' -f2- | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
	printf "  %-12s %5s files  %9s lines\n" "$1" "$n" "${l:-0}"
}

echo "REPOSITORY"
report code
report tests
report docs
report diagnostics
report other

echo
printf "  %-12s %5s   test lines per code line\n" "ratio" \
	"$(awk 'BEGIN{printf "%.2f", '"$(grep '^tests ' /tmp/lm_metric.txt | cut -d' ' -f2- | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')"' / '"$(grep '^code ' /tmp/lm_metric.txt | cut -d' ' -f2- | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')"'}')"

echo
echo "CODE ONLY, by tree"
grep "^code " /tmp/lm_metric.txt | cut -d' ' -f2- | awk -F/ '{print $1}' | sort -u | while read -r d
do
	l=$(grep "^code $d/" /tmp/lm_metric.txt | cut -d' ' -f2- | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
	n=$(grep -c "^code $d/" /tmp/lm_metric.txt)
	[ "${l:-0}" -gt 0 ] && printf "  %-18s %4s files %8s lines\n" "$d" "$n" "$l"
done | sort -k4 -rn

echo
echo "THE REWRITE, against what it replaces"
new=$(wc -l kernels/*.cuh kernels/formats/*.cuh inference/llms/*/* runtime/* 2>/dev/null | tail -1 | awk '{print $1}')
old=$(git ls-files modules model-families | grep -E '\.(c|cu|cuh|h)$' | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
printf "  %-18s %8s lines\n" "kernels+llms+runtime" "$new"
printf "  %-18s %8s lines\n" "modules+families" "$old"
[ "${new:-0}" -gt 0 ] && printf "  %-18s %8s x\n" "ratio" "$((old / new))"
