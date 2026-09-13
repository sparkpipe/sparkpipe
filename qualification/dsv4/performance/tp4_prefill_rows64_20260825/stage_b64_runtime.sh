#!/usr/bin/env bash
# Stage the B64 candidate runtime on spark4-7 (rank-local packs + B64 libs).
# Part of the prefill row-batching experiment; see
# .agents/coord/dsv4flash_prefill_batching_result.md
set -euo pipefail

DRIVERS="$(cd "$(dirname "$0")/../../../.." && pwd)/tools/devcycle/drivers/prefill-b64"
ROOT=/tmp/dsv4-b64-runtime
CONTROL=/tmp/dsv4-integrated-lean-3d962820-runtime

pack_for() {
    case "$1" in
        spark4) echo /home/spark4/sparkdata/dsv4-staging/packtool/tp4b1_rankpacks/dsv4_flash.tp4b1.rank0.spstage ;;
        spark5) echo /home/spark5/sparkdata/dsv4-staging/tp4b1_rankpacks/dsv4_flash.tp4b1.rank1.spstage ;;
        spark6) echo /home/spark6/extnvme/dsv4_packtool/tp4b1_rankpacks/dsv4_flash.tp4b1.rank2.spstage ;;
        spark7) echo /home/spark7/sparkdata/dsv4-staging/tp4b1_rankpacks/dsv4_flash.tp4b1.rank3.spstage ;;
        *) return 1 ;;
    esac
}

for h in spark4 spark5 spark6 spark7; do
    p="$(pack_for "$h")"
    ssh -o BatchMode=yes "$h" "
        set -e
        rm -rf '$ROOT'
        mkdir -p '$ROOT'/bin '$ROOT'/lib '$ROOT'/config '$ROOT'/kv '$ROOT'/packs
        test -r '$p' || { echo missing-pack-$h; exit 9; }
        ln -s '$p' '$ROOT'/packs/dsv4_flash_stage.spstage
    "
    # identical control binaries on all ranks (from the pinned lean runtime on spark4)
    scp -q -o BatchMode=yes \
        "spark4:$CONTROL/bin/sparkpipe_model_residentd" \
        "spark4:$CONTROL/bin/sparkpipe_model_batch" "$h:$ROOT/bin/"
    scp -q -o BatchMode=yes \
        "$DRIVERS/model_driver.so" "$DRIVERS/model_serving_adapter.so" \
        "$DRIVERS/hidden_transport.so" "$h:$ROOT/lib/"
    scp -q -o BatchMode=yes /tmp/b64-model_resident.json "$h:$ROOT/config/model_resident.json"
    repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
    scp -q -o BatchMode=yes "$repo/tools/devcycle/templates/dsv4_flash_tp4_stage.template.json" \
        "$h:$ROOT/config/dsv4_flash_tp4_stage.json"
    echo "stage $h ok"
done

ssh -o BatchMode=yes spark4 "ls -la $ROOT/packs/ && sha256sum $ROOT/lib/*.so | cut -c1-20"
echo STAGING-DONE
