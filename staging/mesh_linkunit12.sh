#!/bin/bash
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
SRC_A="$HOME/g5mesh-src/build/module_library/link_units/f071686b23cac0ba2fe3f19d6e07786505728bff8257c7af01da96343d049b44.a"
ok=0
for h in $ALL; do
    [ "$h" = spark5 ] && { mkdir -p "$HOME/sparkdata/glm5_next.tp16/build/module_library/link_units"; cp -f "$SRC_A" "$HOME/sparkdata/glm5_next.tp16/build/module_library/link_units/"; ok=$((ok+1)); continue; }
    timeout 90 scp -q -o BatchMode=yes -o ConnectTimeout=8 "$SRC_A" "$h:/tmp/g5_unit.a" && \
    timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        "mkdir -p \$HOME/sparkdata/glm5_next.tp16/build/module_library/link_units && mv /tmp/g5_unit.a \$HOME/sparkdata/glm5_next.tp16/build/module_library/link_units/f071686b23cac0ba2fe3f19d6e07786505728bff8257c7af01da96343d049b44.a" \
        && ok=$((ok+1))
done
echo "LINKUNIT12 installed=$ok/16"
