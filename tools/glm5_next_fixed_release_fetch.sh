#!/bin/bash
# Fetch + verify the fixed shared-serving release for the lane-0 M3 A/B
# (glm-m3-fixed-build2, SOURCE_COMMIT ed9ff7f5 - main + the mesh
# register-skip fix #1135 + repro tooling). Idempotent per node.
# Queue: --per-node --resources cpu --memory-mib 512 --cmd 'bash tools/glm5_next_fixed_release_fetch.sh'
set -euo pipefail
DEST="$HOME/glm-m3-fixed"
TARBALL="/tmp/glm53-fixed-release.tar.gz"
[ -f "$DEST/glm53_release/SOURCE_COMMIT" ] && [ "$(cat "$DEST/glm53_release/SOURCE_COMMIT")" = "ed9ff7f52f2721091e01fd804833450c68910377" ] && {
  echo "already present: $(cat "$DEST/glm53_release/SOURCE_COMMIT")"
  exit 0
}
scp -o BatchMode=yes -o ConnectTimeout=8 spark0:/tmp/glm53-fixed-release.tar.gz "$TARBALL"
mkdir -p "$DEST"
tar xzf "$TARBALL" -C "$DEST"
cd "$DEST/glm53_release"
sha256sum -c SHA256SUMS >/dev/null
echo "verified SOURCE_COMMIT=$(cat SOURCE_COMMIT) on $(hostname)"
