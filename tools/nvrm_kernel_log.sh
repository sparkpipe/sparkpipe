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

case "${1:-}" in
mark)
	kernel_log > /dev/null
	cut -d' ' -f1 /proc/uptime
	;;
check)
	since="${2:?check needs the mark from the start of the run}"
	found=$(kernel_log | awk -v since="$since" '
		match($0, /^\[ *[0-9]+\.[0-9]+\]/) {
			stamp = substr($0, RSTART + 1, RLENGTH - 2) + 0
			if ( stamp >= since + 0 && $0 ~ /NVRM:/ && $0 ~ /(Xid|ssert|Cannot|[Ee]rror|[Ff]ail|[Ff]ault)/ )
				print
		}')
	if [ -n "$found" ]; then
		echo "FAIL the open kernel module reported faults or assertions during the run:"
		echo "$found"
		exit 1
	fi
	echo "PASS the kernel log gained no NVRM Xid, assertion, refusal, error or fault line during the run"
	;;
*)
	echo "usage: $0 mark | check MARK" >&2
	exit 2
	;;
esac
