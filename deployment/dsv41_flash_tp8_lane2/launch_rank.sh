#!/bin/sh
set -eu
fail() { printf 'dsv41 launch_rank: %s\n' "$*" >&2; exit 1; }
[ "$#" -eq 3 ] || fail "usage: $0 RANK RELEASE_DIR full|smoke"
rank=$1
release=$2
wset_kind=$3
case "$rank" in [0-7]) ;; *) fail "rank must be 0..7";; esac
case "$wset_kind" in full|smoke) ;; *) fail "wset must be full or smoke";; esac
here=$(cd "$(dirname "$0")" && pwd)
root=$HOME/sparkdata/dsv41flash.mxfp4.tp8
socket=/tmp/spark_weightd.sock
pack=$root/packs/rank$rank.spstage
revision=dba1be0a40aa45a94ad051997016db3960a90277
for f in "$pack" "$pack.sha256" "$pack.experts" "$release/bin/sparkpipe_model_residentd" "$release/bin/weightd_warm" "$release/lib/model_serving_adapter.so" "$release/lib/hidden_transport.so" "$release/stages/stage_000/model_driver.so"; do
    [ -f "$f" ] || fail "missing $f"
done
[ -S "$socket" ] || fail "no weightd socket at $socket"
digests=$(find -L "$root/packs" -maxdepth 1 -name '*.sha256' | wc -l)
[ "$digests" -eq 1 ] || fail "packs/ holds $digests *.sha256 digests, want 1"
sha=$(cut -d' ' -f1 "$pack.sha256")
mkdir -p "$root/config" "$root/kvcache" "$root/bin" "$root/lib" "$root/stages/stage_000" "$root/wset"
cp "$here/config/stage_0$rank.json" "$root/config/stage.json"
cp "$here/model_resident.json" "$root/model_resident.json"
cp "$release/bin/sparkpipe_model_residentd" "$release/bin/weightd_warm" "$root/bin/"
cp "$release/lib/model_serving_adapter.so" "$release/lib/hidden_transport.so" "$root/lib/"
cp "$release/stages/stage_000/model_driver.so" "$root/stages/stage_000/"
if [ "$wset_kind" = smoke ]; then
    cp "$here/wset/rank$rank.smoke.wset" "$root/wset/rank$rank.wset"
else
    python3 -c 'import struct,sys; open(sys.argv[1],"wb").write(b"".join(struct.pack("<II",l,e) for l in range(40) for e in range(48)))' "$root/wset/rank$rank.wset"
fi
SPARK_WEIGHTD_EXPERT_POOL_BYTES=${SPARK_WEIGHTD_EXPERT_POOL_BYTES:?set the expert pool bytes for this lane} \
    "$root/bin/weightd_warm" "$socket" "$pack" "$sha" "$revision" 8 --wset "$root/wset/rank$rank.wset" --family dsv41_flash
systemd-run --user --unit="sp-dsv41-rd-$rank" --working-directory="$root" \
    -p MemoryMax=8G -p MemorySwapMax=0 \
    --setenv=SPARK_WEIGHTD_SOCKET="$socket" --setenv=SPARK_TP_MESH_RANKS=0,1,2,3,4,5,6,7 \
    "$root/bin/sparkpipe_model_residentd" --deployment "$root/model_resident.json" --rank-index "$rank"
