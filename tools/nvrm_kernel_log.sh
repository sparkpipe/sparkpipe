#!/bin/sh
set -eu

kernel_log() {
	if dmesg > /dev/null 2>&1; then
		dmesg
	elif sudo -n dmesg > /dev/null 2>&1; then
		sudo -n dmesg
	else
		echo "FAIL cannot read the kernel log with dmesg or sudo -n dmesg; the nvrm GPU gate needs it" >&2
		exit 1
	fi
}

since() {
	kernel_log | awk -v since="$1" '
		match($0, /^\[ *[0-9]+\.[0-9]+\]/) {
			stamp = substr($0, RSTART + 1, RLENGTH - 2) + 0
			if ( stamp > since + 0 )
				print
		}'
}

case "${1:-}" in
mark)
	kernel_log > /dev/null
	kernel_log | awk '
		match($0, /^\[ *[0-9]+\.[0-9]+\]/) { stamp = substr($0, RSTART + 1, RLENGTH - 2) + 0 }
		END { printf "%.6f\n", stamp + 0 }'
	;;
check)
	mark="${2:?check needs the mark from the start of the run}"
	kernel_log > /dev/null
	found=$(since "$mark" | awk '$0 ~ /NVRM:/ && $0 ~ /(Xid|ssert|Cannot|[Ee]rror|[Ff]ail|[Ff]ault)/')
	if [ -n "$found" ]; then
		echo "FAIL the open kernel module reported faults or assertions during the run:"
		echo "$found"
		exit 1
	fi
	echo "PASS the kernel log gained no NVRM Xid, assertion, refusal, error or fault line during the run"
	;;
xids)
	mark="${2:?xids needs the mark from the start of the run}"
	kernel_log > /dev/null
	since "$mark" | sed -n 's/.*NVRM: Xid ([^)]*): \([0-9][0-9]*\),.*/\1/p' | sort -n -u
	;;
*)
	echo "usage: $0 mark | check MARK | xids MARK" >&2
	exit 2
	;;
esac
