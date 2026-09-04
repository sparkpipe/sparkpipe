#!/bin/bash
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
REC=e919a35b8c63bf20093b1f4e3e72d1f6e2af419cd5e76a3b97a641323da62122.json
UNIT=$(sed 's/UNIT=//' /tmp/g5m_unit_name)
DRV_SHA=$(sha256sum < /tmp/g5m_driver.so | cut -d' ' -f1)
ok=0
for h in $ALL; do
    if [ "$h" = spark5 ]; then
        rr=$HOME/sparkdata/glm5_next.tp16
        cp /tmp/g5m_driver.so $rr/lib/model_driver.so
        cp /tmp/g5m_rec.json $rr/build/module_library/active/$REC
        mkdir -p $rr/build/module_library/link_units
        cp /tmp/g5m_unit.a $rr/build/module_library/link_units/$UNIT.a
        cp /tmp/g5m_pkg.json $rr/model_package.json
        ok=$((ok+1)); continue
    fi
    timeout 150 scp -q -o BatchMode=yes -o ConnectTimeout=8 /tmp/g5m_driver.so "$h:/tmp/g5m_driver.so" && \
    timeout 60 scp -q -o BatchMode=yes -o ConnectTimeout=8 /tmp/g5m_rec.json "$h:/tmp/g5m_rec.json" && \
    timeout 90 scp -q -o BatchMode=yes -o ConnectTimeout=8 /tmp/g5m_unit.a "$h:/tmp/g5m_unit.a" && \
    timeout 60 scp -q -o BatchMode=yes -o ConnectTimeout=8 /tmp/g5m_pkg.json "$h:/tmp/g5m_pkg.json" && \
    timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" '
        rr=$HOME/sparkdata/glm5_next.tp16
        mv /tmp/g5m_driver.so $rr/lib/model_driver.so
        mv /tmp/g5m_rec.json $rr/build/module_library/active/'"$REC"'
        mkdir -p $rr/build/module_library/link_units
        mv /tmp/g5m_unit.a $rr/build/module_library/link_units/'"$UNIT"'.a
        mv /tmp/g5m_pkg.json $rr/model_package.json
        echo ok
    ' >/dev/null 2>&1 && ok=$((ok+1))
done
n=0
for h in $ALL; do
    s=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "sha256sum < \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so" 2>/dev/null | cut -d' ' -f1)
    [ "$s" = "$DRV_SHA" ] && n=$((n+1)) || echo "SHA-MISMATCH $h"
done
echo "DEPLOY13 installed=$ok/16 verified=$n/16 unit=$UNIT drv=${DRV_SHA:0:16}"
