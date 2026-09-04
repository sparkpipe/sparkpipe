#!/bin/bash
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
REF=$(sha256sum < /tmp/g5mesh_driver.so | cut -d' ' -f1)
ok=0
for h in $ALL; do
    s=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        "sha256sum < \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so" 2>/dev/null | cut -d' ' -f1)
    if [ "$s" != "$REF" ]; then
        timeout 150 scp -q -o BatchMode=yes -o ConnectTimeout=8 \
            /tmp/g5mesh_driver.so "$h:/tmp/g5mesh_driver.so" && \
        timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
            "mv /tmp/g5mesh_driver.so \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so" && ok=$((ok+1))
    else
        ok=$((ok+1))
    fi
done
n=0
for h in $ALL; do
    s=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        "sha256sum < \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so" 2>/dev/null | cut -d' ' -f1)
    [ "$s" = "$REF" ] && n=$((n+1)) || echo "SHA-MISMATCH $h"
done
echo "DEPLOY10 updated_ok=$ok verified=$n/16 ref=$REF"
