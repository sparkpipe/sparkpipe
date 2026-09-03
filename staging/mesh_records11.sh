#!/bin/bash
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
REC=e919a35b8c63bf20093b1f4e3e72d1f6e2af419cd5e76a3b97a641323da62122.json
SRC_REC="$HOME/g5mesh-src/build/module_library/active/$REC"
SRC_PKG="$HOME/sparkdata/glm5_next.tp16/model_package.json"
ok=0
for h in $ALL; do
    [ "$h" = spark5 ] && { ok=$((ok+1)); continue; }
    timeout 60 scp -q -o BatchMode=yes -o ConnectTimeout=8 "$SRC_REC" "$h:/tmp/g5_rec.json" && \
    timeout 60 scp -q -o BatchMode=yes -o ConnectTimeout=8 "$SRC_PKG" "$h:/tmp/g5_pkg.json" && \
    timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" '
        rr=$HOME/sparkdata/glm5_next.tp16
        cp $rr/build/module_library/active/'"$REC"' $rr/build/module_library/active/'"$REC"'.prev-mesh 2>/dev/null || true
        mv /tmp/g5_rec.json $rr/build/module_library/active/'"$REC"'
        [ -f $rr/model_package.json ] && cp $rr/model_package.json $rr/model_package.json.prev-mesh 2>/dev/null || true
        mv /tmp/g5_pkg.json $rr/model_package.json
        echo updated
    ' >/dev/null 2>&1 && ok=$((ok+1))
done
echo "RECORDS11 updated=$ok/16"
