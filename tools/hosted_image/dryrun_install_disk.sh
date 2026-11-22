#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "usage: sudo dryrun_install_disk.sh --image BUILD_DIR --dir NEW_DIR" >&2
	exit 2
}
fail() {
	echo "dryrun: FAIL: $*" >&2
	exit 1
}
pass() {
	echo "dryrun: ok: $*"
}

here="$(cd "$(dirname "$0")" && pwd)"
src="$(cd "$here/../.." && pwd)"
image="" dir=""
while [ $# -gt 0 ]; do
	case "$1" in
		--image) image="$2"; shift 2 ;;
		--dir) dir="$2"; shift 2 ;;
		*) usage ;;
	esac
done
[ -n "$image" ] && [ -n "$dir" ] || usage
installer="$here/install_disk.sh"
audit="$src/tools/hosted_dependency_audit.py"
[ -f "$installer" ] && [ -f "$audit" ] && [ -f "$here/pins.conf" ] || fail "$src lacks the installer, pins or audit"
[ "$(id -u)" -eq 0 ] || fail "run as root"
[ -e /dev/loop-control ] || fail "this host has no loop device support (/dev/loop-control)"
for tool in sgdisk losetup udevadm mkfs.vfat mkfs.ext4 e2label fatlabel blkid lsblk findmnt gcc ssh-keygen dropbearkey lsinitramfs efibootmgr python3 zstd; do
	command -v "$tool" > /dev/null || fail "this host lacks $tool"
done
[ -f "$image/rootfs.tar.zst" ] && [ -f "$image/MANIFEST" ] || fail "$image is not a build directory"
[ ! -e "$dir" ] || fail "$dir exists"
parent="$(dirname "$dir")"
[ "$(df --output=avail -B1G "$parent" | tail -1 | tr -d ' ')" -ge 40 ] || fail "$parent has less than 40 GiB free"
KERNEL_RELEASE="$(sed -n 's/^KERNEL_RELEASE=//p' "$here/pins.conf")"
[ -n "$KERNEL_RELEASE" ] || fail "no KERNEL_RELEASE in pins.conf"
mkdir -p "$dir"
dir="$(realpath "$dir")"
exec > >(tee -a "$dir/dryrun.log") 2>&1
echo "dryrun: start $(date -u +%FT%TZ) src=$src image=$image installer_sha=$(sha256sum "$installer" | cut -c1-16)"

loop=""
mounts=()
cleanup() {
	local status=$? index
	for (( index=${#mounts[@]}-1; index>=0; index-- )); do
		umount "${mounts[$index]}" || echo "dryrun: could not unmount ${mounts[$index]}" >&2
	done
	mounts=()
	if [ -n "$loop" ]; then
		losetup -d "$loop" || echo "dryrun: could not detach $loop" >&2
	fi
	echo "dryrun: mounts left under $dir: [$(under | tr '\n' ' ')]; loop devices on the image: [$(losetup -nO NAME -j "$dir/disk.img" 2>/dev/null | tr '\n' ' ')]"
	echo "$status" > "$dir/exit"
}
esp_outside_own() {
	(cd "$1" && find . -ipath ./efi/ubuntu -prune -o -type f -exec sha256sum {} + -o -print) | sort
}
tree_sums() {
	(cd "$1" && find . -type f -exec sha256sum {} + -o -print) | sort
}
under() {
	findmnt -rn -o TARGET | awk -v top="$dir" '$0 == top || index($0, top "/") == 1'
}
mount_at() {
	mount "$@"
	mounts+=("${*: -1}")
}
unmount_last() {
	local last=$(( ${#mounts[@]} - 1 ))
	umount "${mounts[$last]}"
	unset "mounts[$last]"
	mounts=("${mounts[@]}")
}
only_mount() {
	[ "$(under)" = "$1" ] || fail "mounts under $dir are [$(under | tr '\n' ' ')], expected only $1"
}
trap cleanup EXIT

touch "$dir/start"
nvram_before="$(efibootmgr -v)"
truncate -s 24G "$dir/disk.img"
sgdisk -n1:0:+512M -t1:ef00 -n2:0:0 -t2:8300 "$dir/disk.img" > /dev/null
[ "$(sgdisk -p "$dir/disk.img" | grep -cE '^ +[12] ')" = 2 ] || fail "sgdisk wrote no two-partition table"
loop="$(losetup -P --show -f "$dir/disk.img")"
[ "$(losetup -nO BACK-FILE "$loop")" = "$dir/disk.img" ] || fail "$loop is not backed by $dir/disk.img"
udevadm settle
[ -b "${loop}p1" ] && [ -b "${loop}p2" ] || fail "$loop has no p1 and p2"
mkfs.vfat -F 32 -n EFI "${loop}p1" > /dev/null
mkfs.ext4 -q -L root "${loop}p2"
pass "disk image $dir/disk.img on $loop: p1 vfat EFI, p2 ext4 root"

firmware="$dir/firmware"
mkdir "$firmware"
esp_partuuid="$(lsblk -dno PARTUUID "${loop}p1")"
[ -n "$esp_partuuid" ] || fail "${loop}p1 has no partition UUID"
printf 'Boot%s* dryrun\tHD(1,GPT,%s,0x800,0x100000)/File(%s)\n' FFFE "$esp_partuuid" '\EFI\ubuntu\shimaa64.efi' > "$firmware/entry"
printf '#!/bin/bash\n[ "$*" = -v ] || { echo "dryrun: the firmware wrapper only reads with -v" >&2; exit 1; }\nlisting="$(%s -v)" || exit\nprintf "%%s\\n" "$listing" | sed "s/^BootOrder: /&FFFE,/"\ncat %s\n' "$(command -v efibootmgr)" "$firmware/entry" > "$firmware/efibootmgr"
chmod 755 "$firmware/efibootmgr"
[ "$(PATH="$firmware:$PATH" efibootmgr -v | grep -c "^BootOrder: FFFE,\|^BootFFFE\* dryrun")" = 2 ] || fail "the firmware wrapper does not add BootFFFE for ${loop}p1"
pass "firmware wrapper: this host's efibootmgr -v plus an active BootFFFE in BootOrder loading \\EFI\\ubuntu\\shimaa64.efi from ${loop}p1 ($esp_partuuid)"

seed="$dir/seed"
mkdir "$seed"
mount_at "${loop}p2" "$seed"
mkdir -p "$seed"/{etc,usr/bin,var/log,boot/efi,root,opt/vendor,srv,tmp,home/node/sparkdata/packs,home/node/.ssh,home/node/releases/old/bin}
chmod 1777 "$seed/tmp"
chmod 700 "$seed/root"
echo 'PRETTY_NAME="previous system"' > "$seed/etc/os-release"
echo previous > "$seed/etc/hostname"
echo previous-binary > "$seed/usr/bin/previous"
head -c 64M /dev/zero > "$seed/swap.img"
echo vendor > "$seed/opt/vendor/owned"
chown -R 1000:1000 "$seed/opt/vendor"
head -c 256M /dev/urandom > "$seed/home/node/sparkdata/packs/pack.bin"
echo "ssh-ed25519 AAAAdryrun user" > "$seed/home/node/.ssh/authorized_keys"
build="$dir/build-elf"
mkdir "$build"
echo 'int stub(void) { return 0; }' > "$build/stub.c"
echo 'int stub(void); int main(void) { return stub(); }' > "$build/main.c"
gcc -shared -fPIC -Wl,-soname,libcuda.so.1 -o "$build/libcuda.so.1" "$build/stub.c"
gcc -Wl,--no-as-needed -o "$seed/home/node/releases/old/bin/engine" "$build/main.c" "$build/libcuda.so.1"
chown -R 1000:1000 "$seed/home/node"
chmod 700 "$seed/home/node/.ssh"
chmod 600 "$seed/home/node/.ssh/authorized_keys"
(cd "$seed" && find . -path ./home -prune -o -path ./lost+found -prune -o -printf '%P %U:%G %m %s\n' | grep -v '^ ' | sort) > "$dir/seed-root.list"
(cd "$seed/home" && find . -printf '%P %U:%G %m %s %T@\n' | sort) > "$dir/home-before.list"
sha256sum "$seed/home/node/sparkdata/packs/pack.bin" | cut -d' ' -f1 > "$dir/pack.sha"
unmount_last
esp_seed="$dir/esp-seed"
mkdir "$esp_seed"
mount_at "${loop}p1" "$esp_seed"
mkdir -p "$esp_seed/EFI/ubuntu" "$esp_seed/EFI/BOOT" "$esp_seed/EFI/vendor"
echo previous-shim > "$esp_seed/EFI/ubuntu/shimaa64.efi"
echo previous-grub > "$esp_seed/EFI/ubuntu/grubaa64.efi"
echo previous-fallback > "$esp_seed/EFI/BOOT/BOOTAA64.EFI"
echo previous-fallback-loader > "$esp_seed/EFI/BOOT/fbaa64.efi"
echo vendor-loader > "$esp_seed/EFI/vendor/loader.efi"
esp_outside_own "$esp_seed" > "$dir/esp-seed.list"
tree_sums "$esp_seed" > "$dir/esp-seed-full.list"
tree_sums "$esp_seed/EFI/ubuntu" > "$dir/esp-ubuntu-seed.list"
unmount_last
pass "seeded a previous system ($(wc -l < "$dir/seed-root.list") entries outside /home), /home/node with a 256 MiB pack and an old CUDA-linked release root, and an EFI partition with EFI/ubuntu, EFI/BOOT and EFI/vendor"

bundle="$dir/bundle"
files="$bundle/files"
mkdir -p "$files"/etc/{ssh,dropbear/initramfs,sudoers.d,netplan} "$files/var/lib/tailscale" "$files/root/.ssh"
echo node > "$files/etc/hostname"
echo 1 > "$files/etc/ds4-node-rank"
ssh-keygen -q -t ed25519 -N '' -C dryrun -f "$files/etc/ssh/ssh_host_ed25519_key"
dropbearkey -t ed25519 -f "$files/etc/dropbear/initramfs/dropbear_ed25519_host_key" > /dev/null 2>&1
echo 'node ALL=(ALL) NOPASSWD: ALL' > "$files/etc/sudoers.d/90-admins"
chmod 440 "$files/etc/sudoers.d/90-admins"
printf 'network:\n  version: 2\n' > "$files/etc/netplan/90-dryrun.yaml"
chmod 600 "$files/etc/netplan/90-dryrun.yaml"
echo '{}' > "$files/var/lib/tailscale/tailscaled.state"
chmod 600 "$files/var/lib/tailscale/tailscaled.state"
chmod 700 "$files/var/lib/tailscale" "$files/root/.ssh"
echo "ssh-ed25519 AAAAdryrun root" > "$files/root/.ssh/authorized_keys"
chmod 600 "$files/root/.ssh/authorized_keys"
printf 'USER_NAME=node\nUSER_UID=1000\nUSER_GID=1000\nUSER_GROUPS=sudo,adm,admins\nUSER_SHELL=/bin/bash\n' > "$bundle/user.conf"
printf 'admins:1001\n' > "$bundle/groups.conf"
pass "fake identity bundle with $(find "$files" -type f | wc -l) files"

install_run() {
	local name="$1" expect="$2" path="$3" message="${4:-}" status
	echo "dryrun: == $name: install_disk.sh, expecting exit $expect"
	set +e
	PATH="$path" bash "$installer" --disk "$loop" --image "$image" --identity "$bundle" --work "$dir/work-$name" 2>&1 | tee "$dir/install-$name.log"
	status="${PIPESTATUS[0]}"
	set -e
	[ "$status" = "$expect" ] || fail "$name exited $status, expected $expect"
	[ -z "$message" ] || grep -q "$message" "$dir/install-$name.log" || fail "$name did not report: $message"
	[ -z "$(under)" ] || fail "$name left mounts: $(under | tr '\n' ' ')"
	pass "$name exited $status and left no mount"
}

verify_install() {
	local name="$1" check="$dir/check-$1" esp="$dir/esp-$1" aside uuid shim grub
	[ "$(blkid -p -s LABEL -o value "${loop}p2")" = SPARKROOT ] || fail "p2 label is not SPARKROOT"
	[ "$(blkid -p -s LABEL -o value "${loop}p1")" = SPARKESP ] || fail "p1 label is not SPARKESP"
	uuid="$(blkid -p -s UUID -o value "${loop}p2")"
	mkdir "$check" "$esp"
	mount_at -o ro "${loop}p2" "$check"
	[ "$(sha256sum "$check/home/node/sparkdata/packs/pack.bin" | cut -d' ' -f1)" = "$(cat "$dir/pack.sha")" ] || fail "the pack changed"
	(cd "$check/home" && find . -printf '%P %U:%G %m %s %T@\n' | sort) > "$dir/home-$name.list"
	diff "$dir/home-before.list" "$dir/home-$name.list" || fail "/home changed"
	pass "/home is unchanged: $(wc -l < "$dir/home-$name.list") entries with owners, modes, sizes and mtimes, pack sha256 $(cut -c1-16 "$dir/pack.sha")"
	[ "$(find "$check" -mindepth 1 -maxdepth 1 -name '.dgxos-*' | wc -l)" = 1 ] || fail "expected one .dgxos-* directory"
	aside="$(find "$check" -mindepth 1 -maxdepth 1 -name '.dgxos-*')"
	(cd "$aside/root" && find . -printf '%P %U:%G %m %s\n' | grep -v '^ ' | sort) > "$dir/aside-$name.list"
	diff "$dir/seed-root.list" "$dir/aside-$name.list" || fail "the set-aside system differs from the previous root"
	tree_sums "$aside/esp/EFI/ubuntu" > "$dir/aside-esp-$name.list"
	diff "$dir/esp-ubuntu-seed.list" "$dir/aside-esp-$name.list" || fail "the saved EFI/ubuntu differs from the previous one"
	[ "$(ls -A "$aside")" = "$(printf 'esp\nroot')" ] || fail "/${aside##*/} holds more than esp and root: $(ls -A "$aside" | tr '\n' ' ')"
	pass "the previous root is intact in /${aside##*/}/root and the previous EFI/ubuntu in /${aside##*/}/esp"
	grep -q 'Ubuntu 24.04' "$check/etc/os-release" || fail "the installed root is not Ubuntu 24.04"
	[ "$(cat "$check/etc/hostname")" = node ] || fail "hostname not restored"
	cmp -s "$check/etc/ssh/ssh_host_ed25519_key" "$files/etc/ssh/ssh_host_ed25519_key" || fail "SSH host key not restored"
	grep -q '^node:x:1000:1000::/home/node:/bin/bash$' "$check/etc/passwd" || fail "node passwd line: $(grep '^node:' "$check/etc/passwd")"
	grep -q '^admins:x:1001:node$' "$check/etc/group" || fail "admins group: $(grep '^admins:' "$check/etc/group")"
	grep -q '^sudo:.*node' "$check/etc/group" || fail "node is not in sudo"
	for entry in "etc/sudoers.d/90-admins 0:0:440" "etc/ssh/ssh_host_ed25519_key 0:0:600" "etc/netplan/90-dryrun.yaml 0:0:600" "root/.ssh 0:0:700" "root/.ssh/authorized_keys 0:0:600" "var/lib/tailscale/tailscaled.state 0:0:600" ". 0:0:755"; do
		[ "$(stat -c %u:%g:%a "$check/${entry% *}")" = "${entry#* }" ] || fail "${entry% *} is $(stat -c %u:%g:%a "$check/${entry% *}"), expected ${entry#* }"
	done
	echo "dryrun: var/lib/tailscale is $(stat -c %u:%g:%a "$check/var/lib/tailscale")"
	pass "identity restored: hostname, host key, user 1000 with sudo and admins, file modes"
	[ "$(awk '($1 == "LABEL=SPARKROOT" && $2 == "/") || ($1 == "LABEL=SPARKESP" && $2 == "/boot/efi")' "$check/etc/fstab" | wc -l)" = 2 ] || fail "fstab does not name SPARKROOT on / and SPARKESP on /boot/efi"
	[ -z "$(find "$check" -xdev \( -path "$check/home" -o -path "$aside" \) -prune -o \( \( -uid +999 -a ! -uid 65534 \) -o \( -gid +999 -a ! -gid 65534 \) \) -print -quit)" ] || fail "a path outside /home has uid or gid >= 1000"
	[ "$check/boot/initrd.img-$KERNEL_RELEASE" -nt "$dir/start" ] || fail "the initramfs was not rebuilt"
	lsinitramfs "$check/boot/initrd.img-$KERNEL_RELEASE" > "$dir/initramfs-$name.list"
	grep -qx 'etc/dropbear/dropbear_ed25519_host_key' "$dir/initramfs-$name.list" || fail "the initramfs has no dropbear ed25519 host key"
	grep -q "linux.*/boot/vmlinuz-$KERNEL_RELEASE root=UUID=$uuid" "$check/boot/grub/grub.cfg" || fail "grub.cfg has no $KERNEL_RELEASE entry on root UUID $uuid"
	grep -q 'iommu.passthrough=0' "$check/boot/grub/grub.cfg" && grep -q 'initcall_blacklist=tegra234_cbb_init' "$check/boot/grub/grub.cfg" || fail "grub.cfg lacks the GB10 arguments"
	echo "dryrun: kernel line: $(grep -m1 "linux.*vmlinuz-$KERNEL_RELEASE" "$check/boot/grub/grub.cfg" | sed 's/^[[:space:]]*//')"
	pass "initramfs rebuilt with the dropbear key; grub.cfg boots $KERNEL_RELEASE from the root UUID with the GB10 arguments"
	mount_at -o ro "${loop}p1" "$esp"
	for file in shimaa64.efi grubaa64.efi grub.cfg; do
		[ -s "$esp/EFI/ubuntu/$file" ] || fail "EFI/ubuntu/$file missing"
	done
	shim="$(sha256sum "$esp/EFI/ubuntu/shimaa64.efi" | cut -d' ' -f1)"
	sha256sum "$check"/usr/lib/shim/shimaa64.efi* > "$dir/shim-$name.sums"
	grep -q "^$shim " "$dir/shim-$name.sums" || fail "EFI/ubuntu/shimaa64.efi is not one of the image's shim builds"
	echo "dryrun: EFI/ubuntu/shimaa64.efi, which the unchanged firmware entry loads, is $(grep "^$shim " "$dir/shim-$name.sums" | head -1 | sed "s|.*$check||")"
	grub="$(sha256sum "$esp/EFI/ubuntu/grubaa64.efi" | cut -d' ' -f1)"
	[ "$grub" = "$(sha256sum "$check/usr/lib/grub/arm64-efi-signed/grubaa64.efi.signed" | cut -d' ' -f1)" ] || fail "EFI/ubuntu/grubaa64.efi is not the image's signed GRUB"
	grep -q "$uuid" "$esp/EFI/ubuntu/grub.cfg" || fail "EFI/ubuntu/grub.cfg does not search $uuid"
	esp_outside_own "$esp" > "$dir/esp-$name.list"
	diff "$dir/esp-seed.list" "$dir/esp-$name.list" || fail "the EFI partition changed outside EFI/ubuntu"
	echo "dryrun: EFI/ubuntu: $(ls "$esp/EFI/ubuntu" | tr '\n' ' ')"
	echo "dryrun: EFI/ubuntu/grub.cfg: $(tr '\n' ';' < "$esp/EFI/ubuntu/grub.cfg")"
	pass "EFI partition: the image's signed shim and GRUB in EFI/ubuntu, the stub searches the root, EFI/BOOT and EFI/vendor byte-identical"
	unmount_last
	unmount_last
}

install_run no-entry 1 "$PATH" "no active firmware boot entry"
install_run first 0 "$firmware:$PATH"
grep -q "booted by the unchanged firmware entry BootFFFE" "$dir/install-first.log" || fail "the install did not name BootFFFE as the entry that boots it"
verify_install first

echo "dryrun: == rollback"
roll="$dir/rollback"
mkdir "$roll"
mount_at "${loop}p2" "$roll"
only_mount "$roll"
aside="$(find "$roll" -mindepth 1 -maxdepth 1 -name '.dgxos-*')"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
hosted="$roll/.hosted-$stamp"
mkdir "$hosted"
find "$roll" -mindepth 1 -maxdepth 1 ! -name home ! -name lost+found ! -name '.dgxos-*' ! -name ".hosted-$stamp" -printf '%f\n' | while read -r entry; do mv "$roll/$entry" "$hosted/"; done
find "$aside/root" -mindepth 1 -maxdepth 1 -printf '%f\n' | while read -r entry; do mv "$aside/root/$entry" "$roll/"; done
rmdir "$aside/root"
roll_esp="$dir/rollback-esp"
mkdir "$roll_esp"
mount_at "${loop}p1" "$roll_esp"
mkdir -p "$hosted/esp/EFI"
mv "$roll_esp/EFI/ubuntu" "$hosted/esp/EFI/"
cp -r "$aside/esp/EFI/ubuntu" "$roll_esp/EFI/"
tree_sums "$roll_esp" > "$dir/rollback-esp.list"
diff "$dir/esp-seed-full.list" "$dir/rollback-esp.list" || fail "the rolled-back EFI partition differs from the previous one"
unmount_last
only_mount "$roll"
rm -r --one-file-system "$aside"
(cd "$roll" && find . -path ./home -prune -o -path ./lost+found -prune -o -path "./.hosted-$stamp" -prune -o -printf '%P %U:%G %m %s\n' | grep -v '^ ' | sort) > "$dir/rollback-root.list"
diff "$dir/seed-root.list" "$dir/rollback-root.list" || fail "the rolled-back root differs from the previous root"
unmount_last
e2label "${loop}p2" root
fatlabel "${loop}p1" EFI > /dev/null
pass "rollback restored the previous root and the whole EFI partition exactly; the image waits in /.hosted-$stamp"

install_run refused 1 "$firmware:$PATH" "holds a previous aside directory"

mount_at "${loop}p2" "$roll"
only_mount "$roll"
[ -d "$roll/.hosted-$stamp" ] || fail "no .hosted-$stamp to remove"
rm -rf --one-file-system "$roll/.hosted-$stamp"
unmount_last
pass "removed /.hosted-$stamp after the refusal"

install_run second 0 "$firmware:$PATH"
verify_install second

echo "dryrun: == acceptance gate"
gate="$dir/gate"
mkdir "$gate"
mount_at "${loop}p2" "$gate"
only_mount "$gate"
aside="$(find "$gate" -mindepth 1 -maxdepth 1 -name '.dgxos-*')"
[ -n "$aside" ] || fail "no set-aside directory at the gate"
rm -rf --one-file-system "$aside"
pass "deleted /${aside##*/}"
if python3 -B "$audit" image --rootfs "$gate" --release-root opt/sparkpipe/release > "$dir/gate-1.txt"; then status=0; else status=$?; fi
cat "$dir/gate-1.txt"
[ "$status" = 1 ] || fail "the gate with the old CUDA-linked release root in /home exited $status, expected 1"
grep -q 'home/node/releases/old/bin/engine' "$dir/gate-1.txt" || fail "the gate did not name the old engine"
pass "the gate refuses the node while /home holds a CUDA-linked release root"
rm -rf --one-file-system "$gate/home/node/releases"
if python3 -B "$audit" image --rootfs "$gate" --release-root opt/sparkpipe/release > "$dir/gate-2.txt"; then status=0; else status=$?; fi
cat "$dir/gate-2.txt"
[ "$status" = 0 ] || fail "the gate exited $status after the old release root went"
pass "the gate passes over the whole installed root with no exclusion"
unmount_last

[ "$(efibootmgr -v)" = "$nvram_before" ] || fail "this host's firmware boot entries changed"
pass "efibootmgr -v unchanged on this host"
echo "dryrun: PASS $(date -u +%FT%TZ)"
