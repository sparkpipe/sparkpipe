#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
need SOURCE_REPO BUILD_HOST BUILD_DIR RELEASE_WORK RELEASE_BUILD_SCRIPT PERF_HOLDER_FILE RELEASE_GPU_HOLDER_BLOCK
start_log build
clone="$RELEASE_WORK/clone-$S7"
[ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" = "$SHA" ] || die "$SOURCE_REPO is not at $SHA"
git -C "$SOURCE_REPO" diff --quiet HEAD || die "$SOURCE_REPO is dirty"
if [ ! -d "$clone" ]; then
    git init -q "$clone"
    git -C "$clone" fetch -q --depth 1 "$SOURCE_REPO" HEAD
    git -C "$clone" checkout -q FETCH_HEAD
    git -C "$clone" remote add origin https://github.com/sparkpipe/sparkpipe
fi
[ "$(git -C "$clone" rev-parse HEAD)" = "$SHA" ] || die "clone is not at $SHA"
holder=$(cat "$PERF_HOLDER_FILE" 2>/dev/null || true)
for p in $RELEASE_GPU_HOLDER_BLOCK; do
    case "$holder" in "$p"*) die "PERF_HOLDER is '$holder' (blocks $RELEASE_GPU_HOLDER_BLOCK): the build's publish validators use the GPU for <1 min; wait";; esac
done
$RELEASE_SSH "$BUILD_HOST" "test ! -e $BUILD_DIR" || die "$BUILD_HOST:$BUILD_DIR exists; move it aside first"
$RELEASE_SSH "$BUILD_HOST" "mkdir -p $(dirname "$BUILD_DIR")" || die "mkdir on $BUILD_HOST failed"
rsync -a "$clone/" "$BUILD_HOST:$BUILD_DIR/" || die "rsync to $BUILD_HOST failed"
unit="sp-release-$S7-build"
$RELEASE_SSH "$BUILD_HOST" "cd $BUILD_DIR && [ \"\$(git rev-parse HEAD)\" = $SHA ] && git diff --quiet HEAD && systemd-run --user --unit=$unit --collect --wait -p MemoryMax=32G -p CPUWeight=10 -p Nice=19 --setenv=SPARK_QUEUE_ID=release-$S7-build-$ts --working-directory=$BUILD_DIR bash -c 'bash $RELEASE_BUILD_SCRIPT > $BUILD_DIR.build.out 2>&1'; tail -n 5 $BUILD_DIR.build.out; cat $BUILD_DIR/build/$RELEASE_ARTIFACT.tar.gz.sha256" || die "build failed (read $BUILD_HOST:$BUILD_DIR.build.out)"
echo "BUILD DONE $(date -u +%H:%M:%SZ). Register $BUILD_HOST:$BUILD_DIR in that node's ~/KEEP. Next: stage.sh. log $log"
