#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
need SOURCE_REPO BUILD_HOST BUILD_DIR RELEASE_WORK RELEASE_API_MAKE
start_log stage
local_stage="$RELEASE_WORK/stage-$S7"
hub "test ! -e ~/$ASSEMBLE && test ! -e ~/$API_BUILD" || die "hub ~/$ASSEMBLE or ~/$API_BUILD exists"
[ "$(git -C "$SOURCE_REPO" rev-parse HEAD)" = "$SHA" ] || die "$SOURCE_REPO is not at $SHA"
hub "mkdir -p ~/$ASSEMBLE ~/$API_BUILD" || die "mkdir on the hub failed"
mkdir -p "$local_stage"
scp -q "$BUILD_HOST:$BUILD_DIR/build/$RELEASE_ARTIFACT.tar.gz" "$BUILD_HOST:$BUILD_DIR/build/$RELEASE_ARTIFACT.tar.gz.sha256" "$local_stage/" || die "tarball copy from $BUILD_HOST failed"
(cd "$local_stage" && [ "$(shasum -a 256 "$RELEASE_ARTIFACT.tar.gz" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$RELEASE_ARTIFACT.tar.gz.sha256")" ]) || die "tarball copy bad"
scp -q "$local_stage/$RELEASE_ARTIFACT.tar.gz" "$local_stage/$RELEASE_ARTIFACT.tar.gz.sha256" "$KIT/stage_config_patch.py" "$RELEASE_HUB:$ASSEMBLE/" || die "tarball copy to the hub failed"
if [ "$STAGE_CONFIG_ADD" != none ]; then
    need NEW_CONFIG_DIR
    (cd "$NEW_CONFIG_DIR" && shasum -a 256 config/stage_*.json > SHA256SUMS)
    hub "mkdir -p ~/$ASSEMBLE/newcfg" && scp -q -r "$NEW_CONFIG_DIR/config" "$NEW_CONFIG_DIR/SHA256SUMS" "$RELEASE_HUB:$ASSEMBLE/newcfg/" || die "newcfg copy failed"
fi
hub_script assemble_root.sh $HUB_VARS || die "assemble failed"
git -C "$SOURCE_REPO" archive --format=tar "$SHA" | hub "tar -xf - -C ~/$API_BUILD && echo $SHA > ~/$API_BUILD/SOURCE_COMMIT.build" || die "source copy to the hub failed"
hub "cd ~/$API_BUILD && nice -n 15 bash -c $(printf %q "$RELEASE_API_MAKE") > api-make.log 2>&1 && echo api \$(sha256sum < build/sparkpipe_model_api | cut -c1-16) adapter \$(sha256sum < $RELEASE_ADAPTER_BUILD | cut -c1-16) && { ldd -r build/sparkpipe_model_api $RELEASE_ADAPTER_BUILD 2>&1 | grep -E 'not found|undefined symbol' || echo ldd-clean; }" || die "x86 api/adapter build failed (hub ~/$API_BUILD/api-make.log)"
hub_script stage_channel.sh $HUB_VARS || die "channel staging failed"
echo "STAGE DONE $(date -u +%H:%M:%SZ): fill NEW_MANIFEST_SHA, NEW_RESIDENTD, NEW_DRIVER, NEW_API, NEW_ADAPTER, NEW_CHANNEL_DEPLOYMENT, EXPECT_CHANGED from the lines above, then precheck.sh. log $log"
