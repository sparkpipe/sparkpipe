#!/usr/bin/env bash
set -euo pipefail

FAMILY="${1:?module family}"
CODEC="${2:?codec}"
ROOT_NAME="${3:?release root name}"
TREE="$HOME/sparkpipe-build"
RELEASE="$HOME/release/$ROOT_NAME"
ADAPTER_SO="$TREE/build/modules/$FAMILY/$CODEC/libglm5_next_serving_adapter_$CODEC.so"

DEPLOY_JSON="$RELEASE/model_resident.json"
if [ -f "$DEPLOY_JSON" ]; then
    BAD=$(grep -c "sparkdata/[^\"]*$ROOT_NAME" "$DEPLOY_JSON")
    TOTAL=$(grep -c "runtime_root" "$DEPLOY_JSON")
    [ "$BAD" = "$TOTAL" ] || {
        echo "deployment runtime_root does not match release root $ROOT_NAME ($BAD/$TOTAL)"
        grep -o "sparkdata/[a-z0-9._]*tp16" "$DEPLOY_JSON" | sort -u | head -3
        exit 1
    }
fi

mkdir -p "$RELEASE/stages/stage_000" "$RELEASE/lib" "$RELEASE/bin"

install -m 644 "$HOME/sparkdata/out/stages/stage_000/model_driver.so" \
    "$RELEASE/stages/stage_000/model_driver.so"
install -m 644 "$TREE/build/libhidden_transport_spark_host_rdma_verbs.so" \
    "$RELEASE/lib/hidden_transport.so"
install -m 644 "$ADAPTER_SO" "$RELEASE/lib/model_serving_adapter.so"
install -m 755 "$TREE/build/sparkpipe_model_residentd" "$RELEASE/bin/sparkpipe_model_residentd"
install -m 755 "$TREE/build/sparkpipe_model_api" "$RELEASE/bin/sparkpipe_model_api"

cd "$RELEASE"
find lib bin stages config model_resident.json -type f ! -name stage.json ! -name MANIFEST 2>/dev/null |
    sort | xargs sha256sum > MANIFEST.tmp
mv MANIFEST.tmp MANIFEST

rm -f /srv/qpn/*.rec 2>/dev/null || true
touch UPDATE
echo "published $ROOT_NAME -> $RELEASE (local hub sparkf)"
