#!/usr/bin/env bash
set -euo pipefail
root="$1"
out="$2"
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
. "$here/pins.conf"
. "$here/owners.sh"
source_ref="$(cat "$out/inputs/source-commit")"
install_tree() {
	(cd "$1" && find . -type f -print) | sort | while read -r file; do
		if [ -x "$1/$file" ]; then mode=0755; else mode=0644; fi
		install -D -o 0 -g 0 -m "$mode" "$1/$file" "$2/${file#./}"
	done
}
[ -z "$(find "$here/files" "$out/inputs/release-root" ! -type f ! -type d -print -quit)" ] || { echo "customize_image: the image files and the release root may hold only files and directories" >&2; exit 1; }
install_tree "$here/files" "$root"
chmod 0600 "$root/etc/netplan/01-sparkpipe.yaml"
chmod 0440 "$root/etc/sudoers.d/90-sparkpipe-compact"
chroot "$root" visudo -c -q -f /etc/sudoers.d/90-sparkpipe-compact || { echo "customize_image: /etc/sudoers.d/90-sparkpipe-compact does not parse" >&2; exit 1; }
chroot "$root" getent group sparkpipe-compact > /dev/null || chroot "$root" groupadd --system sparkpipe-compact
install -m 0600 "$out/inputs/spark-firewall.nft" "$root/etc/spark-firewall.nft"
install -d "$root/lib/modules/$KERNEL_RELEASE/updates"
for module in $MODULE_FILES; do
	install -m 0644 "$out/modules/$module.ko" "$root/lib/modules/$KERNEL_RELEASE/updates/$module.ko"
done
chroot "$root" depmod -a "$KERNEL_RELEASE"
for module in $MODULE_FILES; do
	[ "$(chroot "$root" modinfo -k "$KERNEL_RELEASE" -n "$module")" = "/lib/modules/$KERNEL_RELEASE/updates/$module.ko" ] || { echo "customize_image: depmod does not resolve $module to the installed module" >&2; exit 1; }
done
[ -f "$root/lib/firmware/nvidia/$MODULE_VERSION/gsp_ga10x.bin" ] || { echo "customize_image: $FIRMWARE_PACKAGE did not install the GSP firmware for $MODULE_VERSION" >&2; exit 1; }
install -d -o 0 -g 0 -m 0755 "$root/$RELEASE_ROOT_PATH" "$(dirname "$root/$AGENT_PATH")"
install_tree "$out/inputs/release-root" "$root/$RELEASE_ROOT_PATH"
git -C "$repo" show "$source_ref:tools/fleet_node_agent.sh" > "$root/$AGENT_PATH"
chmod 0755 "$root/$AGENT_PATH"
printf 'sparkpipe-node\n' > "$root/etc/hostname"
printf 'LABEL=SPARKROOT\t/\text4\terrors=remount-ro\t0\t1\nLABEL=SPARKESP\t/boot/efi\tvfat\tumask=0077,nofail,x-systemd.device-timeout=10s\t0\t1\n' > "$root/etc/fstab"
install -m 0644 "$out/tool/tailscale.deb" "$root/tmp/tailscale.deb"
chroot "$root" dpkg -i /tmp/tailscale.deb > "$out/tailscale-install.log" 2>&1 || { tail -20 "$out/tailscale-install.log" >&2; exit 1; }
rm -f "$root/tmp/tailscale.deb"
install -d -m 0755 "$root/run/sshd"
python3 "$here/stage_baseline.py" "$root" --source-ref "$source_ref" --report "$out/baseline-report.json"
rmdir "$root/run/sshd"
systemctl --root="$root" enable sparkpipe-host-keys.service
"$here/mask_units.sh" "$root"
! chroot "$root" dpkg-query -W -f='${Status}' unattended-upgrades 2> /dev/null | grep -q "ok installed" || { echo "customize_image: unattended-upgrades is in the image; an image node never updates itself" >&2; exit 1; }
rm -f "$root"/etc/ssh/ssh_host_*_key "$root"/etc/ssh/ssh_host_*_key.pub "$root"/etc/dropbear/initramfs/dropbear_*_host_key
SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" chroot "$root" update-initramfs -u -k "$KERNEL_RELEASE" > "$out/initramfs.log" 2>&1 || { tail -20 "$out/initramfs.log" >&2; exit 1; }
[ "$(stat -c %u:%g:%a "$root")" = "0:0:755" ] || { echo "customize_image: the image root is $(stat -c %u:%g:%a "$root"), not 0:0:755" >&2; exit 1; }
owned="$(foreign_owner "$root" "$root")"
[ -z "$owned" ] || { echo "customize_image: ${owned#"$root"} belongs to a uid or gid of 1000 or above that the image does not define; the image takes no build user's ownership" >&2; exit 1; }
