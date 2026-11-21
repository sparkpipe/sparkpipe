#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/pins.conf"

usage() {
	echo "usage: install_disk.sh --disk DEVICE --image BUILD_DIR --identity BUNDLE_DIR --work DIR" >&2
	exit 2
}

fail() {
	echo "install_disk: $*" >&2
	exit 1
}

disk="" image="" identity="" work=""
while [ $# -gt 0 ]; do
	case "$1" in
		--disk) disk="$2"; shift 2 ;;
		--image) image="$2"; shift 2 ;;
		--identity) identity="$2"; shift 2 ;;
		--work) work="$2"; shift 2 ;;
		*) usage ;;
	esac
done
[ -n "$disk" ] && [ -n "$image" ] && [ -n "$identity" ] && [ -n "$work" ] || usage
[ -f "$image/rootfs.tar.zst" ] && [ -f "$image/MANIFEST" ] || fail "$image holds no rootfs.tar.zst and MANIFEST"
grep -qx "audit PASS" "$image/MANIFEST" || fail "$image/MANIFEST does not record a passing dependency audit"
expected="$(sed -n 's/^rootfs_sha256 //p' "$image/MANIFEST")"
[ -n "$expected" ] || fail "$image/MANIFEST names no rootfs_sha256"
echo "$expected  $image/rootfs.tar.zst" | sha256sum -c --quiet - || fail "$image/rootfs.tar.zst does not match its MANIFEST"
[ -d "$identity/files" ] && [ -f "$identity/user.conf" ] && [ -f "$identity/groups.conf" ] || fail "$identity needs files/, user.conf and groups.conf"
setting() {
	sed -n "s/^$1=//p" "$identity/user.conf" | tail -1
}
USER_NAME="$(setting USER_NAME)" USER_UID="$(setting USER_UID)" USER_GID="$(setting USER_GID)" USER_GROUPS="$(setting USER_GROUPS)" USER_SHELL="$(setting USER_SHELL)"
[[ "$USER_GROUPS" =~ ^[a-z0-9,]*$ ]] || fail "$identity/user.conf USER_GROUPS must be a comma list of group names"
[[ "$USER_NAME" =~ ^[a-z][a-z0-9]*$ ]] && [[ "$USER_UID" =~ ^[0-9]+$ ]] && [[ "$USER_GID" =~ ^[0-9]+$ ]] && [ -n "$USER_SHELL" ] || fail "$identity/user.conf needs USER_NAME, USER_UID, USER_GID, USER_GROUPS and USER_SHELL"
[ -z "$(find "$identity/files" ! -type f ! -type d -print -quit)" ] || fail "$identity/files may hold only files and directories"
if [ -f "$identity/files/etc/ssh/ssh_host_ed25519_key" ] && [ ! -f "$identity/files/etc/dropbear/initramfs/dropbear_ed25519_host_key" ]; then
	fail "$identity carries SSH host keys but no initramfs SSH host keys, so the first boot would not create them"
fi
while IFS= read -r line; do
	[ -z "$line" ] || [[ "$line" =~ ^[a-z_][a-z0-9_-]*:[0-9]+$ ]] || fail "$identity/groups.conf line '$line' is not name:gid"
done < "$identity/groups.conf"
[ -b "$disk" ] || fail "$disk is not a block device"
[ "$(id -u)" -eq 0 ] || fail "run as root"
nvram="$(efibootmgr -v)" || fail "efibootmgr cannot read the firmware boot entries"
esp="${disk}p1"
root="${disk}p2"
[ "$(lsblk -dno FSTYPE "$esp")" = vfat ] || fail "$esp is not a vfat EFI system partition"
[ "$(lsblk -dno FSTYPE "$root")" = ext4 ] || fail "$root is not ext4"
for partition in "$esp" "$root"; do
	[ -z "$(findmnt -rn -S "$partition" -o TARGET)" ] || fail "$partition is mounted"
done
esp_partuuid="$(lsblk -dno PARTUUID "$esp")"
[ -n "$esp_partuuid" ] || fail "$esp has no GPT partition UUID"
boot_entry="$(awk -v partition="gpt,$esp_partuuid," '
	/^BootOrder:/ { order = "," $2 "," }
	$1 ~ /^Boot[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]\*$/ && index(tolower($0), tolower(partition)) && index(tolower($0), "file(\\efi\\ubuntu\\shimaa64.efi)") { entries[substr($1, 5, 4)] = 1 }
	END { for (entry in entries) if (index(order, "," entry ",")) { print "Boot" entry; exit } }
' <<< "$nvram")"
[ -n "$boot_entry" ] || fail "no active firmware boot entry in BootOrder loads \\EFI\\ubuntu\\shimaa64.efi from $esp (partition $esp_partuuid)"
[ ! -e "$work" ] || [ -z "$(ls -A "$work")" ] || fail "$work is not empty"

mounted=()
unmount_all() {
	local index
	for (( index=${#mounted[@]}-1; index>=0; index-- )); do
		umount "${mounted[$index]}" || echo "install_disk: could not unmount ${mounted[$index]}" >&2
	done
	mounted=()
}
unmount_last() {
	local last=$(( ${#mounted[@]} - 1 ))
	umount "${mounted[$last]}"
	unset "mounted[$last]"
	mounted=("${mounted[@]}")
}
trap unmount_all EXIT
mount_at() {
	mount "$@"
	mounted+=("${*: -1}")
}
foreign_esp_entries() {
	(cd "$1" && find . -ipath ./efi/ubuntu -prune -o -type f -exec sha256sum {} + -o -print) | sort
}

e2fsck -fp "$root" && status=0 || status=$?
[ "$status" -le 1 ] || fail "e2fsck on $root reported $status"
mkdir -p "$work/target" "$work/esp"
work="$(realpath "$work")"
target="$work/target"
mount_at "$root" "$target"
[ "$(findmnt -rn -o TARGET | awk -v top="$target" '$0 == top || index($0, top "/") == 1')" = "$target" ] || fail "something else is mounted under $target"
[ "$(df --output=avail -B1G "$target" | tail -1 | tr -d ' ')" -ge 10 ] || fail "$root has less than 10 GiB free"
[ -z "$(find "$target" -mindepth 1 -maxdepth 1 \( -name '.dgxos-*' -o -name '.hosted-*' \) -print -quit)" ] || fail "$root holds a previous aside directory; finish its acceptance or rollback first"
mount_at -o ro "$esp" "$work/esp"
[ -s "$work/esp/EFI/ubuntu/shimaa64.efi" ] || fail "$esp has no EFI/ubuntu/shimaa64.efi for the firmware boot entry to load"
esp_before="$(foreign_esp_entries "$work/esp")"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
aside="$target/.dgxos-$stamp"
mkdir -p "$aside/root" "$aside/esp/EFI"
find "$target" -mindepth 1 -maxdepth 1 ! -name home ! -name lost+found ! -name ".dgxos-$stamp" -printf '%f\n' | sort > "$work/aside.list"
while read -r entry; do
	mv "$target/$entry" "$aside/root/"
done < "$work/aside.list"
echo "install_disk: moved $(wc -l < "$work/aside.list") top-level entries into /.dgxos-$stamp/root; /home stays"
cp -a "$work/esp/EFI/ubuntu" "$aside/esp/EFI/ubuntu"
diff -r "$work/esp/EFI/ubuntu" "$aside/esp/EFI/ubuntu" > /dev/null || fail "the copy of EFI/ubuntu in /.dgxos-$stamp/esp differs from $esp"
unmount_last
echo "install_disk: copied EFI/ubuntu from $esp into /.dgxos-$stamp/esp"

tar --zstd -xpf "$image/rootfs.tar.zst" --numeric-owner --xattrs --xattrs-include='*' --exclude=./home -C "$target"
[ "$(stat -c %u:%g:%a "$target")" = "0:0:755" ] || fail "the installed root is $(stat -c %u:%g:%a "$target"), not 0:0:755"
owned="$(find "$target" -xdev \( -path "$target/home" -o -path "$aside" \) -prune -o \( \( -uid +999 -a ! -uid 65534 \) -o \( -gid +999 -a ! -gid 65534 \) \) -print -quit)"
[ -z "$owned" ] || fail "${owned#"$target"} belongs to uid or gid 1000 or above"

e2label "$root" SPARKROOT
fatlabel "$esp" SPARKESP

while IFS=: read -r name gid; do
	[ -n "$name" ] || continue
	grep -q "^$name:" "$target/etc/group" || groupadd --root "$target" -g "$gid" "$name"
done < "$identity/groups.conf"
grep -q "^$USER_NAME:" "$target/etc/group" || groupadd --root "$target" -g "$USER_GID" "$USER_NAME"
useradd --root "$target" -u "$USER_UID" -g "$USER_GID" ${USER_GROUPS:+-G "$USER_GROUPS"} -d "/home/$USER_NAME" -M -s "$USER_SHELL" "$USER_NAME"
(cd "$identity/files" && find . -mindepth 1 -type d -print) | sort | while read -r directory; do
	[ -d "$target/${directory#./}" ] || install -d -o 0 -g 0 -m "$(stat -c %a "$identity/files/$directory")" "$target/${directory#./}"
done
(cd "$identity/files" && find . -type f -print) | sort | while read -r file; do
	install -D -o 0 -g 0 -m "$(stat -c %a "$identity/files/$file")" "$identity/files/$file" "$target/${file#./}"
done

install -d -o 0 -g 0 -m 0755 "$target/boot/efi"
mount_at "$esp" "$target/boot/efi"
[ "$(foreign_esp_entries "$target/boot/efi")" = "$esp_before" ] || fail "the EFI system partition changed outside EFI/ubuntu before the bootloader install"
for directory in dev proc sys; do
	mount_at --bind "/$directory" "$target/$directory"
done
chroot "$target" visudo -cq || fail "the sudoers files in the installed root do not parse"
chroot "$target" grub-install --target=arm64-efi --efi-directory=/boot/efi --bootloader-id=ubuntu --uefi-secure-boot --no-nvram --no-extra-removable
chroot "$target" update-grub
chroot "$target" update-initramfs -u -k "$KERNEL_RELEASE"
for file in shimaa64.efi grubaa64.efi grub.cfg; do
	[ -s "$target/boot/efi/EFI/ubuntu/$file" ] || fail "the EFI system partition has no EFI/ubuntu/$file"
done
[ "$(foreign_esp_entries "$target/boot/efi")" = "$esp_before" ] || fail "the EFI system partition changed outside EFI/ubuntu"
grep -q "$(blkid -s UUID -o value "$root")" "$target/boot/efi/EFI/ubuntu/grub.cfg" || fail "EFI/ubuntu/grub.cfg does not search the installed root"
grep -q "vmlinuz-$KERNEL_RELEASE" "$target/boot/grub/grub.cfg" || fail "/boot/grub/grub.cfg has no entry for $KERNEL_RELEASE"
unmount_all
[ "$(efibootmgr -v)" = "$nvram" ] || fail "the firmware boot entries changed during the install"
[ -z "$(findmnt -rn -o TARGET | awk -v top="$work" '$0 == top || index($0, top "/") == 1')" ] || fail "mounts remain under $work"
echo "install_disk: installed $(sed -n 's/^source_commit //p' "$image/MANIFEST") on $root (SPARKROOT) with $esp (SPARKESP), booted by the unchanged firmware entry $boot_entry; DGX OS is in /.dgxos-$stamp until acceptance"
