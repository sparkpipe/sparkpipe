#!/bin/sh
set -eu
usage() {
	echo "usage: sparkpipe-dscp-trust {set|check|clear} DEVICE" >&2
	exit 2
}
[ $# -eq 2 ] || usage
action="$1"
device="$2"
expected=$(i=0; while [ "$i" -lt 64 ]; do printf '%d:%d\n' "$i" $((i >> 3)); i=$((i + 1)); done)
current() {
	dcb -N app show dev "$device" dscp-prio | tr ' ' '\n' | grep -E '^[0-9]+:[0-7]$' | sort -t: -k1,1n || true
}
check() {
	if [ "$(current)" = "$expected" ]; then
		echo "$device trusts DSCP"
		return 0
	fi
	echo "$device does not trust DSCP: its DSCP priority map is not the 64-entry DSCP>>3 map" >&2
	return 1
}
case "$action" in
	set)
		dcb app replace dev "$device" dscp-prio $expected
		check
		;;
	check)
		check
		;;
	clear)
		dcb app flush dev "$device" dscp-prio
		[ -z "$(current)" ] || { echo "$device still holds DSCP priority entries" >&2; exit 1; }
		echo "$device trusts PCP"
		;;
	*)
		usage
		;;
esac
