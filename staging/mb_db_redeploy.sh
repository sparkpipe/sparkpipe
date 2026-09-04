#!/bin/bash
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
B=$(md5sum "$HOME/mb_doorbell.new" | cut -d' ' -f1)
echo "deploy md5=$B"
ok=0; miss=""
for h in $ALL; do
    [ "$h" = "spark5" ] && { mv -f "$HOME/mb_doorbell.new" "$HOME/mb_doorbell"; ok=$((ok+1)); continue; }
    timeout 60 scp -q -o BatchMode=yes -o ConnectTimeout=8 "$HOME/mb_doorbell" "$h:~/mb_doorbell.new" >/dev/null 2>&1 \
        && timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "mv -f ~/mb_doorbell.new ~/mb_doorbell" >/dev/null 2>&1 \
        || { miss="$miss $h"; continue; }
done
sleep 2
bad=""
for h in $ALL; do
    r=$(timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" "md5sum \$HOME/mb_doorbell 2>/dev/null | cut -d' ' -f1" 2>/dev/null)
    [ "$r" = "$B" ] && ok=$((ok+1)) || bad="$bad $h"
done
echo "verify $ok/16 ok bad:$bad"
[ -z "$bad" ] && echo DEPLOY-OK || echo DEPLOY-PARTIAL
