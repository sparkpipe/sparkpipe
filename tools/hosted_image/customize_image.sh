#!/usr/bin/env bash
set -euo pipefail
root="$1"
out="$2"
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
. "$here/pins.conf"
source_ref="$(cat "$out/inputs/source-commit")"
cp -a "$here/files/." "$root/"
chmod 0600 "$root/etc/netplan/01-sparkpipe.yaml"
install -m 0600 "$out/inputs/spark-firewall.nft" "$root/etc/spark-firewall.nft"
install -d "$root/lib/modules/$KERNEL_RELEASE/updates/sparkpipe"
install -m 0644 "$out/modules/nvidia.ko" "$out/modules/nvidia-uvm.ko" "$root/lib/modules/$KERNEL_RELEASE/updates/sparkpipe/"
chroot "$root" depmod -a "$KERNEL_RELEASE"
[ "$(chroot "$root" modinfo -k "$KERNEL_RELEASE" -n nvidia)" = "/lib/modules/$KERNEL_RELEASE/updates/sparkpipe/nvidia.ko" ] || { echo "customize_image: depmod does not resolve nvidia to the built module" >&2; exit 1; }
[ -f "$root/lib/firmware/nvidia/$OPEN_MODULE_TAG/gsp_ga10x.bin" ] || { echo "customize_image: $FIRMWARE_PACKAGE did not install the GSP firmware for $OPEN_MODULE_TAG" >&2; exit 1; }
install -d "$root/$RELEASE_ROOT_PATH" "$(dirname "$root/$AGENT_PATH")"
cp -a "$out/inputs/release-root/." "$root/$RELEASE_ROOT_PATH/"
git -C "$repo" show "$source_ref:tools/fleet_node_agent.sh" > "$root/$AGENT_PATH"
chmod 0755 "$root/$AGENT_PATH"
printf 'sparkpipe-node\n' > "$root/etc/hostname"
printf 'LABEL=SPARKROOT\t/\text4\terrors=remount-ro\t0\t1\nLABEL=SPARKESP\t/boot/efi\tvfat\tumask=0077\t0\t1\n' > "$root/etc/fstab"
python3 "$here/stage_baseline.py" "$root" --source-ref "$source_ref" --report "$out/baseline-report.json"
systemctl --root="$root" enable sparkpipe-host-keys.service
rm -f "$root"/etc/ssh/ssh_host_*_key "$root"/etc/ssh/ssh_host_*_key.pub "$root"/etc/dropbear/initramfs/dropbear_*_host_key
SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" chroot "$root" update-initramfs -u -k "$KERNEL_RELEASE" > "$out/initramfs.log" 2>&1 || { tail -20 "$out/initramfs.log" >&2; exit 1; }
