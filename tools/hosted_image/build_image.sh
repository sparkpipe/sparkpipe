#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
. "$here/pins.conf"
export SOURCE_DATE_EPOCH
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$repo"

usage() {
	echo "usage: build_image.sh --out DIR --source-ref REF --release-root DIR --firewall-ruleset FILE" >&2
	exit 2
}

mounts_under() {
	findmnt -rn -o TARGET | awk -v top="$1" '$0 == top || index($0, top "/") == 1'
}

remove_tree() {
	local held
	held="$(mounts_under "$1")"
	[ -z "$held" ] || { echo "build_image: refusing to remove $1; mounted under it: $held" >&2; exit 1; }
	rm -rf --one-file-system "$1"
}

out="" source_ref="" release_root="" firewall=""
while [ $# -gt 0 ]; do
	case "$1" in
		--out) out="$2"; shift 2 ;;
		--source-ref) source_ref="$2"; shift 2 ;;
		--release-root) release_root="$2"; shift 2 ;;
		--firewall-ruleset) firewall="$2"; shift 2 ;;
		*) usage ;;
	esac
done
[ -n "$out" ] && [ -n "$source_ref" ] && [ -n "$release_root" ] && [ -n "$firewall" ] || usage
[ "$(id -u)" -eq 0 ] || { echo "build_image: run as root; mmdebstrap builds the root in root mode" >&2; exit 1; }
[ -d "$release_root" ] || { echo "build_image: release root $release_root is not a directory" >&2; exit 1; }
[ -z "$(find "$release_root" -type l -print -quit)" ] || { echo "build_image: release root $release_root holds a symbolic link" >&2; exit 1; }
[ -s "$firewall" ] || { echo "build_image: firewall ruleset $firewall is missing or empty" >&2; exit 1; }
[ ! -e "$out" ] || { echo "build_image: $out exists; give a new output directory" >&2; exit 1; }
out="$(realpath -m "$out")"
commit="$(git -C "$repo" rev-parse --verify "$source_ref^{commit}")"
[ "$(git -C "$repo" rev-parse HEAD)" = "$commit" ] && [ -z "$(git -C "$repo" status --porcelain)" ] || { echo "build_image: the checkout must be clean at $source_ref" >&2; exit 1; }
python3 "$repo/tools/hosted_dependency_audit.py" root "$release_root" > /dev/null || { python3 "$repo/tools/hosted_dependency_audit.py" root "$release_root" >&2; echo "build_image: the release root fails the dependency audit" >&2; exit 1; }
start="$(date +%s)"

mkdir -p "$out/inputs/release-root" "$out/tool" "$out/modules"
echo "$commit" > "$out/inputs/source-commit"
install -m 0600 "$firewall" "$out/inputs/spark-firewall.nft"
cp -a "$release_root/." "$out/inputs/release-root/"

curl -fsSL "$SNAPSHOT_URL/$MMDEBSTRAP_DEB" -o "$out/tool/mmdebstrap.deb"
echo "$MMDEBSTRAP_SHA256  $out/tool/mmdebstrap.deb" | sha256sum -c --quiet
dpkg-deb -x "$out/tool/mmdebstrap.deb" "$out/tool/root"
mmdebstrap="$out/tool/root/usr/bin/mmdebstrap"
[ "$("$mmdebstrap" --version)" = "mmdebstrap $MMDEBSTRAP_VERSION" ] || { echo "build_image: mmdebstrap is not $MMDEBSTRAP_VERSION" >&2; exit 1; }

curl -fsSL "$SNAPSHOT_URL/$MODULE_DEB" -o "$out/modules/module.deb"
echo "$MODULE_SHA256  $out/modules/module.deb" | sha256sum -c --quiet
[ "$(dpkg-deb -f "$out/modules/module.deb" Package)" = "$MODULE_PACKAGE" ] || { echo "build_image: $MODULE_DEB is not $MODULE_PACKAGE" >&2; exit 1; }
dpkg-deb -x "$out/modules/module.deb" "$out/modules/package"
for module in $MODULE_FILES; do
	found="$(find "$out/modules/package/lib/modules/$KERNEL_RELEASE" -name "$module.ko" -type f)"
	[ "$(printf '%s\n' "$found" | grep -c .)" -eq 1 ] || { echo "build_image: $MODULE_PACKAGE holds no single $module.ko for $KERNEL_RELEASE" >&2; exit 1; }
	[ "$(modinfo -F version "$found")" = "$MODULE_VERSION" ] || { echo "build_image: $module.ko is not version $MODULE_VERSION" >&2; exit 1; }
	[ "$(modinfo -F vermagic "$found" | cut -d' ' -f1)" = "$KERNEL_RELEASE" ] || { echo "build_image: $module.ko is not built for $KERNEL_RELEASE" >&2; exit 1; }
	[ "$(modinfo -F signer "$found")" = "$MODULE_SIGNER" ] || { echo "build_image: $module.ko is not signed by $MODULE_SIGNER" >&2; exit 1; }
	install -m 0644 "$found" "$out/modules/$module.ko"
done
remove_tree "$out/modules/package"

mirrors=()
for suite in "$SUITE" "$SUITE-updates" "$SUITE-security"; do
	mirrors+=("deb $SNAPSHOT_URL $suite $COMPONENTS")
done
"$mmdebstrap" --mode=root --architectures="$ARCHITECTURE" --format=directory --variant=apt \
	--aptopt='Acquire::Check-Valid-Until "false"' --aptopt='APT::Install-Recommends "false"' \
	--include="$(paste -sd, "$here/image-packages.txt")" \
	--customize-hook="$here/customize_image.sh \"\$1\" $out" \
	"$SUITE" "$out/rootfs" "${mirrors[@]}"
held="$(mounts_under "$out")"
[ -z "$held" ] || { echo "build_image: mounts remain under $out after mmdebstrap: $held" >&2; exit 1; }

if ! python3 "$repo/tools/hosted_dependency_audit.py" --json image --rootfs "$out/rootfs" --release-root "$RELEASE_ROOT_PATH" > "$out/audit.json"; then
	python3 "$repo/tools/hosted_dependency_audit.py" image --rootfs "$out/rootfs" --release-root "$RELEASE_ROOT_PATH" >&2 || true
	echo "build_image: the image fails the dependency audit; it is kept at $out/rootfs and not packed" >&2
	exit 1
fi

tar --sort=name --mtime="@$SOURCE_DATE_EPOCH" --clamp-mtime --numeric-owner --xattrs --xattrs-include='*' \
	--pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime -C "$out/rootfs" -cf - . | zstd -q -T0 -19 -o "$out/rootfs.tar.zst"
{
	echo "source_commit $commit"
	sed 's/^\([A-Z_]*\)=/\1 /' "$here/pins.conf"
	echo "rootfs_sha256 $(sha256sum "$out/rootfs.tar.zst" | cut -d' ' -f1)"
	echo "audit $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/audit.json")"
	for module in $MODULE_FILES; do
		echo "module $module $(sha256sum "$out/modules/$module.ko" | cut -d' ' -f1)"
	done
	(cd "$out/inputs/release-root" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sed 's/^/release_file /')
	dpkg-query --admindir="$out/rootfs/var/lib/dpkg" -W -f='package ${Package} ${Version}\n' | sort
} > "$out/MANIFEST"
echo "build_image: $out/rootfs.tar.zst passes the dependency audit in $(( $(date +%s) - start )) s; manifest $out/MANIFEST"
