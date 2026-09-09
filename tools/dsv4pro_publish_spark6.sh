#!/bin/bash
# DSV4 Pro publish on the dedicated CUDA node (spark6): waits for the
# rank00 pack copy, writes the accepted-pack sidecar, runs the full
# publish (one-time GPU validator + module + driver + adapter).
set -uo pipefail
cd $HOME/dsv4pro_checkout
mkdir -p /home/spark6/sparkdata/dsv4_pro.tp4pp4/packs /mnt/model-warm/packbuild/dsv4pro
LOG=/mnt/model-warm/packbuild/dsv4pro/publish_spark6.log
exec > "$LOG" 2>&1
export PATH=/usr/local/cuda/bin:$PATH
PACK=$HOME/sparkdata/dsv4_pro.tp4pp4/packs/dsv4_pro.tp4_pp4.rank00.spstage
for i in $(seq 1 60); do [ -s "$PACK" ] && break; sleep 30; done
[ -s "$PACK" ] || { echo "MISSING-PACK"; exit 1; }
sha256sum "$PACK" | awk '{print $1}' > "$PACK.sha256.tmp" && mv "$PACK.sha256.tmp" "$PACK.sha256"
echo "== accepted-pack sidecar: $(cat "$PACK.sha256")"
bash tools/dsv4pro_driver_rebuild.sh
rc=$?
echo "PUBLISH-RC=$rc" | tee /home/spark6/dsv4pro_checkout/publish.exit
exit $rc
