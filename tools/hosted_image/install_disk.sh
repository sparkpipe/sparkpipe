#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/pins.conf"
. "$here/owners.sh"

usage() {
	echo "usage: install_disk.sh --disk DEVICE --image BUILD_DIR --identity BUNDLE_DIR --work DIR [--carry LIST] [--carry-accounts NAME,...]" >&2
	exit 2
}

fail() {
	echo "install_disk: $*" >&2
	exit 1
}

disk="" image="" identity="" work="" carry="" accounts=""
while [ $# -gt 0 ]; do
	case "$1" in
		--disk) disk="$2"; shift 2 ;;
		--image) image="$2"; shift 2 ;;
		--identity) identity="$2"; shift 2 ;;
		--work) work="$2"; shift 2 ;;
		--carry) carry="$2"; shift 2 ;;
		--carry-accounts) accounts="$2"; shift 2 ;;
		*) usage ;;
	esac
done
[ -n "$disk" ] && [ -n "$image" ] && [ -n "$identity" ] && [ -n "$work" ] || usage
tailscale_state=var/lib/tailscale
reserved_path() {
	case "$1" in
		home|home/*|lost+found|lost+found/*|.dgxos-*|.hosted-*|"$tailscale_state"|"$tailscale_state"/*) return 0 ;;
	esac
	return 1
}
carry_patterns=()
if [ -n "$carry" ]; then
	[ -f "$carry" ] || fail "$carry is not a carry list"
	while IFS= read -r pattern; do
		[ -n "$pattern" ] || continue
		[[ "$pattern" =~ ^[A-Za-z0-9_.@*-]+(/[A-Za-z0-9_.@*-]+)*$ ]] && [[ "/$pattern/" != */../* ]] && [[ "/$pattern/" != */./* ]] || fail "$carry line '$pattern' is not a path relative to the root"
		! reserved_path "$pattern" || fail "$carry may not name /$pattern"
		carry_patterns+=("$pattern")
	done < "$carry"
fi
[[ "$accounts" =~ ^([a-z_][a-z0-9_-]*(,[a-z_][a-z0-9_-]*)*)?$ ]] || fail "--carry-accounts takes a comma list of account names"
IFS=, read -r -a carry_accounts <<< "$accounts"
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
[ ! -e "$identity/files/var/lib/tailscale" ] || fail "$identity carries var/lib/tailscale; the node's Tailscale state comes from its previous root, never from the bundle"
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
if [ -e "$target/$tailscale_state" ] || [ -L "$target/$tailscale_state" ]; then
	[ -d "$target/$tailscale_state" ] && [ ! -L "$target/$tailscale_state" ] || fail "$root's /$tailscale_state is not a directory"
	[ -z "$(find "$target/$tailscale_state" ! -type f ! -type d -print -quit)" ] || fail "$root's /$tailscale_state may hold only files and directories"
fi
carried=()
for pattern in "${carry_patterns[@]}"; do
	while IFS= read -r path; do
		[ -n "$path" ] || continue
		! reserved_path "$path" || fail "the carry list matches /$path"
		[ -z "$(find "$target/$path" ! -type f ! -type d ! -type l -print -quit)" ] || fail "$root's /$path holds something other than files, directories and links"
		carried+=("$path")
	done < <(cd "$target" && shopt -s nullglob && for match in $pattern; do if [ -e "$match" ] || [ -L "$match" ]; then printf '%s\n' "$match"; fi; done)
done
carried_kib="$( [ ${#carried[@]} -eq 0 ] || (cd "$target" && du -sk --apparent-size "${carried[@]}" | awk '{total += $1} END {print total}') )"
[ "$(df --output=avail -B1K "$target" | tail -1 | tr -d ' ')" -ge $(( ${carried_kib:-0} + 10485760 )) ] || fail "$root lacks 10 GiB free beyond the ${carried_kib:-0} KiB the carry list copies"
for name in "${carry_accounts[@]}"; do
	grep -q "^$name:" "$target/etc/passwd" && grep -q "^$name:" "$target/etc/shadow" || fail "$root has no account $name to carry"
done
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

excluded=(--exclude=./home)
for path in "${carried[@]}"; do
	excluded+=("--exclude=./$path")
done
tar --zstd -xpf "$image/rootfs.tar.zst" --numeric-owner --xattrs --xattrs-include='*' "${excluded[@]}" -C "$target"
[ ! -e "$target/$tailscale_state" ] || [ -z "$(ls -A "$target/$tailscale_state")" ] || fail "the image carries Tailscale state in /$tailscale_state"
if [ -d "$aside/root/$tailscale_state" ]; then
	cp -a "$aside/root/$tailscale_state" "$target/${tailscale_state%/*}/"
	diff -r --no-dereference "$aside/root/$tailscale_state" "$target/$tailscale_state" > /dev/null || fail "the carried /$tailscale_state differs from /.dgxos-$stamp/root/$tailscale_state"
	echo "install_disk: carried the node's Tailscale state from /.dgxos-$stamp/root/$tailscale_state, so the node rejoins the tailnet as itself"
else
	echo "install_disk: the previous root has no /$tailscale_state, so the node joins the tailnet at bring-up"
fi
carry_listing() {
	(cd "$1" && find "$2" -printf '%y %U:%G %m %s %l %p\n' | sort)
}
skip=(\( -path "$target/home" -o -path "$aside")
for path in "${carried[@]}"; do
	[ ! -e "$target/$path" ] && [ ! -L "$target/$path" ] || fail "the image wrote /$path although the extraction excluded it"
	mkdir -p "$target/$(dirname "$path")"
	cp -a "$aside/root/$path" "$target/$path"
	[ "$(carry_listing "$aside/root" "$path")" = "$(carry_listing "$target" "$path")" ] && diff -r --no-dereference "$aside/root/$path" "$target/$path" > /dev/null || fail "the carried /$path differs from /.dgxos-$stamp/root/$path"
	skip+=(-o -path "$target/$path")
	echo "install_disk: carried /$path from the previous root"
done
skip+=(\) -prune -o)
[ "$(stat -c %u:%g:%a "$target")" = "0:0:755" ] || fail "the installed root is $(stat -c %u:%g:%a "$target"), not 0:0:755"
owned="$(foreign_owner "$target" "$target" "${skip[@]}")"
[ -z "$owned" ] || fail "${owned#"$target"} belongs to a uid or gid of 1000 or above that the image does not define"

e2label "$root" SPARKROOT
fatlabel "$esp" SPARKESP

while IFS=: read -r name gid; do
	[ -n "$name" ] || continue
	grep -q "^$name:" "$target/etc/group" || groupadd --root "$target" -g "$gid" "$name"
done < "$identity/groups.conf"
grep -q "^$USER_NAME:" "$target/etc/group" || groupadd --root "$target" -g "$USER_GID" "$USER_NAME"
useradd --root "$target" -u "$USER_UID" -g "$USER_GID" ${USER_GROUPS:+-G "$USER_GROUPS"} -d "/home/$USER_NAME" -M -s "$USER_SHELL" "$USER_NAME"
for name in "${carry_accounts[@]}"; do
	user="$(grep "^$name:" "$aside/root/etc/passwd")"
	IFS=: read -r _ _ uid gid _ <<< "$user"
	! grep -q "^$name:" "$target/etc/passwd" || fail "the installed root already has an account $name"
	! cut -d: -f3 "$target/etc/passwd" | grep -qx "$uid" || fail "uid $uid of $name is taken in the installed root"
	group="$(awk -F: -v gid="$gid" '$3 == gid' "$aside/root/etc/group")"
	[ -n "$group" ] || fail "the previous root has no group $gid for $name"
	if ! cut -d: -f3 "$target/etc/group" | grep -qx "$gid"; then
		! grep -q "^${group%%:*}:" "$target/etc/group" || fail "the installed root has a group ${group%%:*} with another gid"
		printf '%s\n' "$group" >> "$target/etc/group"
		grep "^${group%%:*}:" "$aside/root/etc/gshadow" >> "$target/etc/gshadow" || printf '%s:!::\n' "${group%%:*}" >> "$target/etc/gshadow"
	fi
	printf '%s\n' "$user" >> "$target/etc/passwd"
	grep "^$name:" "$aside/root/etc/shadow" >> "$target/etc/shadow"
	echo "install_disk: carried the account $name (uid $uid, gid $gid) from the previous root"
done
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
