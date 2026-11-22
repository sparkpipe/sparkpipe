#!/usr/bin/env bash
set -euo pipefail
root="$1"
here="$(cd "$(dirname "$0")" && pwd)"
mapfile -t units < "$here/masked-units.txt"
systemctl --root="$root" mask "${units[@]}"
for unit in "${units[@]}"; do
	[ "$(readlink "$root/etc/systemd/system/$unit")" = /dev/null ] || { echo "mask_units: $unit is not masked in $root" >&2; exit 1; }
done
