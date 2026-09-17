#!/bin/sh
# THE definition of what counts as code. Sourced, never copied.
#
# metric.sh and breakdown.sh each carried their own version of this and
# disagreed by 1,958 lines on the same tree: the root Makefile and sources.mk
# (1,029) plus every modules/*/Makefile. A ratio whose denominator has two
# values cannot be optimised, and this had already produced one fake 3,871-line
# improvement when a file moved into a directory one of them did not count.
#
#   code          what ships. THE DENOMINATOR.
#   tests         what proves it. Apart, because the two trade off: a kernel
#                 whose contract is a static_assert needs no test, so test
#                 lines falling can mean the code got safer rather than thinner.
#   docs          prose and measurements. Grows with understanding, not debt.
#   diagnostics   captured tensors and logs. Evidence for measured claims.
#   other         everything else.
spark_classify() {
	case "$1" in
		diagnostics/*|*/validation-logs/*) echo diagnostics ;;
		*.bin|*.bf16|*.f32|*.i32|*.u32|*.npz|*.safetensors|*.log) echo diagnostics ;;
		SHA256SUMS|*/SHA256SUMS) echo diagnostics ;;
		tests/*) echo tests ;;
		docs/*|*.md|*.txt) echo docs ;;
		*.c|*.cu|*.cuh|*.h|*.hpp|*.cc|*.cpp|*.py|*.sh|*.mk|Makefile|*/Makefile) echo code ;;
		*) echo other ;;
	esac
}

# The classified file list, cached per run. Both reports read this and therefore
# cannot disagree.
spark_classified() {
	out=${1:-/tmp/lm_metric.txt}
	git ls-files | while read -r f
	do
		printf "%s %s\n" "$(spark_classify "$f")" "$f"
	done > "$out"
}

# Sum the lines of one bucket, or of one bucket under one top-level directory.
spark_lines() {
	if [ -n "$2" ]
	then grep "^$1 $2/" "${3:-/tmp/lm_metric.txt}" | cut -d' ' -f2-
	else grep "^$1 " "${3:-/tmp/lm_metric.txt}" | cut -d' ' -f2-
	fi | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1+0}'
}
