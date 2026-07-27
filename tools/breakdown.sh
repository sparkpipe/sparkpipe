#!/bin/sh
# Complete code breakdown by module. Code only - tests, docs and diagnostics are
# counted separately by tools/metric.sh because they are not the denominator.
#
# The directory list is DERIVED, not written down. It used to be a fixed list
# and it omitted node/, so 12,384 lines were missing from the total for as long
# as that directory has existed. Moving one file from api/ to node/ then made
# the total drop by 3,871 lines with nothing deleted. A hardcoded list of where
# code lives is the same defect as $(patsubst src/%.c,...) in the Makefile: it
# fails by quietly leaving things out.
cd "$(dirname "$0")/.." || exit 1
CODE='\.(c|cu|cuh|h|py|sh)$'
SKIP='^(tests|docs|diagnostics|examples|qualification)$'
total=0
# Two levels for inference/, one for everything else: inference/ holds three
# subsystems that are worth seeing apart.
dirs=$(git ls-files | grep -E "$CODE" | sed 's|/[^/]*$||' | \
	awk -F/ '{ if ($1 == "inference" && NF > 1) print $1"/"$2; else print $1 }' | \
	grep -v '\.' | sort -u)
for d in $dirs
do
	echo "$d" | grep -qE "$SKIP" && continue
	[ -d "$d" ] || continue
	files=$(git ls-files "$d" | grep -E "$CODE")
	n=$(echo "$files" | grep -c .)
	[ "${n:-0}" -eq 0 ] && continue
	l=$(echo "$files" | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
	printf "  %-22s %4s files %8s lines\n" "$d" "$n" "${l:-0}"
	total=$((total + ${l:-0}))
done
printf "  %-22s %4s       %8s lines\n" "TOTAL" "" "$total"
