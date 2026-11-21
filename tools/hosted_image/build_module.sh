#!/usr/bin/env bash
set -euo pipefail
root="$1"
out="$2"
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/pins.conf"
headers="/usr/src/linux-headers-$KERNEL_RELEASE"
[ -x "$root$headers/scripts/sign-file" ] || { echo "build_module: $headers/scripts/sign-file is missing from the build root" >&2; exit 1; }
mkdir -p "$root/usr/src/open-module" "$root/root/module-key"
cp -a "$out/upstream/open-gpu-kernel-modules/." "$root/usr/src/open-module/"
rm -rf "$root/usr/src/open-module/.git"
install -m 0600 "$out/inputs/module-key.pem" "$root/root/module-key/key.pem"
install -m 0644 "$out/inputs/module-cert" "$root/root/module-key/cert"
chroot "$root" nice -n 19 make -C /usr/src/open-module modules -j"$(nproc)" SYSSRC="$headers" TARGET_ARCH=aarch64 > "$out/module-build.log" 2>&1 || { tail -40 "$out/module-build.log" >&2; exit 1; }
mkdir -p "$out/modules"
for module in nvidia nvidia-uvm; do
	chroot "$root" "$headers/scripts/sign-file" sha512 /root/module-key/key.pem /root/module-key/cert "/usr/src/open-module/kernel-open/$module.ko"
	install -m 0644 "$root/usr/src/open-module/kernel-open/$module.ko" "$out/modules/$module.ko"
done
chroot "$root" modinfo -F vermagic /usr/src/open-module/kernel-open/nvidia.ko | grep -q "^$KERNEL_RELEASE " || { echo "build_module: nvidia.ko was not built for $KERNEL_RELEASE" >&2; exit 1; }
rm -rf "$root/root/module-key"
