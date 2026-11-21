#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
. "$here/pins.conf"
export SOURCE_DATE_EPOCH
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$repo"

usage() {
	echo "usage: build_image.sh --out DIR --source-ref REF --release-root DIR --firewall-ruleset FILE --module-key FILE --module-cert FILE" >&2
	exit 2
}

out="" source_ref="" release_root="" firewall="" module_key="" module_cert=""
while [ $# -gt 0 ]; do
	case "$1" in
		--out) out="$2"; shift 2 ;;
		--source-ref) source_ref="$2"; shift 2 ;;
		--release-root) release_root="$2"; shift 2 ;;
		--firewall-ruleset) firewall="$2"; shift 2 ;;
		--module-key) module_key="$2"; shift 2 ;;
		--module-cert) module_cert="$2"; shift 2 ;;
		*) usage ;;
	esac
done
[ -n "$out" ] && [ -n "$source_ref" ] && [ -n "$release_root" ] && [ -n "$firewall" ] && [ -n "$module_key" ] && [ -n "$module_cert" ] || usage
[ "$(id -u)" -eq 0 ] || { echo "build_image: run as root; mmdebstrap builds the roots in root mode" >&2; exit 1; }
[ -d "$release_root" ] || { echo "build_image: release root $release_root is not a directory" >&2; exit 1; }
[ -z "$(find "$release_root" -type l -print -quit)" ] || { echo "build_image: release root $release_root holds a symbolic link" >&2; exit 1; }
[ -s "$firewall" ] || { echo "build_image: firewall ruleset $firewall is missing or empty" >&2; exit 1; }
[ -s "$module_key" ] && openssl x509 -in "$module_cert" -noout 2> /dev/null || { echo "build_image: the module signing key and a PEM certificate are required" >&2; exit 1; }
[ ! -e "$out" ] || { echo "build_image: $out exists; give a new output directory" >&2; exit 1; }
commit="$(git -C "$repo" rev-parse --verify "$source_ref^{commit}")"
[ "$(git -C "$repo" rev-parse HEAD)" = "$commit" ] && [ -z "$(git -C "$repo" status --porcelain)" ] || { echo "build_image: the checkout must be clean at $source_ref" >&2; exit 1; }
python3 "$repo/tools/hosted_dependency_audit.py" root "$release_root" > /dev/null || { python3 "$repo/tools/hosted_dependency_audit.py" root "$release_root" >&2; echo "build_image: the release root fails the dependency audit" >&2; exit 1; }

mkdir -p "$out/inputs/release-root" "$out/tool" "$out/upstream"
echo "$commit" > "$out/inputs/source-commit"
install -m 0600 "$firewall" "$out/inputs/spark-firewall.nft"
install -m 0600 "$module_key" "$out/inputs/module-key.pem"
install -m 0644 "$module_cert" "$out/inputs/module-cert"
openssl x509 -in "$module_cert" -outform DER -out "$out/module-cert.der"
cp -a "$release_root/." "$out/inputs/release-root/"

curl -fsSL "$SNAPSHOT_URL/$MMDEBSTRAP_DEB" -o "$out/tool/mmdebstrap.deb"
echo "$MMDEBSTRAP_SHA256  $out/tool/mmdebstrap.deb" | sha256sum -c --quiet
dpkg-deb -x "$out/tool/mmdebstrap.deb" "$out/tool/root"
mmdebstrap="$out/tool/root/usr/bin/mmdebstrap"
[ "$("$mmdebstrap" --version)" = "mmdebstrap $MMDEBSTRAP_VERSION" ] || { echo "build_image: mmdebstrap is not $MMDEBSTRAP_VERSION" >&2; exit 1; }

git clone --quiet --depth 1 --branch "$OPEN_MODULE_TAG" "$OPEN_MODULE_REPOSITORY" "$out/upstream/open-gpu-kernel-modules"
[ "$(git -C "$out/upstream/open-gpu-kernel-modules" rev-parse HEAD)" = "$OPEN_MODULE_COMMIT" ] || { echo "build_image: $OPEN_MODULE_TAG is not commit $OPEN_MODULE_COMMIT" >&2; exit 1; }

mirrors=()
for suite in "$SUITE" "$SUITE-updates" "$SUITE-security"; do
	mirrors+=("deb $SNAPSHOT_URL $suite $COMPONENTS")
done
options=(--mode=root --architectures="$ARCHITECTURE" --format=directory
	--aptopt='Acquire::Check-Valid-Until "false"' --aptopt='APT::Install-Recommends "false"')

"$mmdebstrap" "${options[@]}" --variant=buildd --include="$(paste -sd, "$here/build-packages.txt")" \
	--customize-hook="$here/build_module.sh \"\$1\" $out" \
	"$SUITE" "$out/build-root" "${mirrors[@]}"
rm -rf "$out/build-root" "$out/inputs/module-key.pem"

"$mmdebstrap" "${options[@]}" --variant=apt --include="$(paste -sd, "$here/image-packages.txt")" \
	--customize-hook="$here/customize_image.sh \"\$1\" $out" \
	"$SUITE" "$out/rootfs" "${mirrors[@]}"

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
	echo "module_signer $(openssl x509 -in "$out/inputs/module-cert" -noout -fingerprint -sha256 | cut -d= -f2)"
	echo "rootfs_sha256 $(sha256sum "$out/rootfs.tar.zst" | cut -d' ' -f1)"
	echo "audit $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/audit.json")"
	(cd "$out/inputs/release-root" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sed 's/^/release_file /')
	dpkg-query --admindir="$out/rootfs/var/lib/dpkg" -W -f='package ${Package} ${Version}\n' | sort
} > "$out/MANIFEST"
echo "build_image: $out/rootfs.tar.zst passes the dependency audit; manifest $out/MANIFEST"
