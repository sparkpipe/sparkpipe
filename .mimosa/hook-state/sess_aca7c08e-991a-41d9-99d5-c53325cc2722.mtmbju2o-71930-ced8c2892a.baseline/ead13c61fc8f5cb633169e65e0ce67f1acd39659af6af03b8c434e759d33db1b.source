#!/bin/sh
# The code bucket, by top-level directory. Same classifier metric.sh uses, so
# the TOTAL here and the code line there are the same number by construction.
#
# They were not, twice. First a hardcoded directory list omitted node/ - 12,384
# lines missing until a file moved into it and the total "improved" by 3,871
# with nothing deleted. Then a second private definition of "code" put the two
# reports 1,958 lines apart. Both were the same defect: a written-down copy of
# something that already had a definition somewhere else.
cd "$(dirname "$0")/.." || exit 1
. "$(dirname "$0")/classify.sh"
spark_classified /tmp/lm_breakdown.txt
total=0
grep '^code ' /tmp/lm_breakdown.txt | cut -d' ' -f2- | \
	awk -F/ '{ if (NF == 1) print "<root>"; else if ($1 == "inference" && NF > 2) print $1"/"$2; else print $1 }' | \
	sort -u | while read -r d
do
	if [ "$d" = "<root>" ]
	then files=$(grep '^code ' /tmp/lm_breakdown.txt | cut -d' ' -f2- | grep -v '/')
	else files=$(grep "^code $d/" /tmp/lm_breakdown.txt | cut -d' ' -f2-)
	fi
	n=$(echo "$files" | grep -c .)
	[ "${n:-0}" -eq 0 ] && continue
	l=$(echo "$files" | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1+0}')
	printf "  %-22s %4s files %8s lines\n" "$d" "$n" "$l"
done | sort -k4 -rn
printf "  %-22s %4s files %8s lines\n" "TOTAL" \
	"$(grep -c '^code ' /tmp/lm_breakdown.txt)" "$(spark_lines code '' /tmp/lm_breakdown.txt)"
