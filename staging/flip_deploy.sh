#!/bin/bash
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
RR="$HOME/sparkdata/glm5_next.tp16"
LOG=/tmp/flip_deploy.log
exec >> "$LOG" 2>&1
echo "=== FLIP-DEPLOY $(date -u +%H:%M:%S) ==="

bash "$HOME/mesh_task13a.sh" || { echo TASK13-FAIL; exit 1; }

UNIT=$(cat /tmp/g5m_unit_name | cut -d= -f2)
DRV=$(sha256sum < /tmp/g5m_driver.so | cut -d' ' -f1)
for f in /tmp/g5m_driver.so /tmp/g5m_rec.json /tmp/g5m_unit.a /tmp/g5m_pkg.json "$HOME/mb_transport/hidden_transport.so"; do
    [ -s "$f" ] || { echo MISSING "$f"; exit 1; }
done

ok=0; miss=""
for h in $ALL; do
    [ "$h" = "spark5" ] || timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" "echo up" >/dev/null 2>&1 || { miss="$miss $h"; continue; }
    (
    timeout 150 scp -q -o BatchMode=yes -o ConnectTimeout=8 /tmp/g5m_driver.so "$h:/tmp/f_driver.so"
    timeout 60 scp -q -o BatchMode=yes -o ConnectTimeout=8 /tmp/g5m_rec.json "$h:/tmp/f_rec.json"
    timeout 60 scp -q -o BatchMode=yes -o ConnectTimeout=8 /tmp/g5m_unit.a "$h:/tmp/f_unit.a"
    timeout 60 scp -q -o BatchMode=yes -o ConnectTimeout=8 /tmp/g5m_pkg.json "$h:/tmp/f_pkg.json"
    timeout 60 scp -q -o BatchMode=yes -o ConnectTimeout=8 "$HOME/mb_transport/hidden_transport.so" "$h:/tmp/f_transport.so"
    timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "
cd \$HOME/sparkdata/glm5_next.tp16
cp -f lib/model_driver.so lib/model_driver.so.nccl-era 2>/dev/null
cp -f lib/hidden_transport.so lib/hidden_transport.so.aug31 2>/dev/null
mv /tmp/f_driver.so lib/model_driver.so
mv /tmp/f_transport.so lib/hidden_transport.so
mv /tmp/f_pkg.json model_package.json
mkdir -p build/module_library/active build/module_library/link_units
rm -f build/module_library/active/*.json
mv /tmp/f_rec.json build/module_library/active/
mv /tmp/f_unit.a build/module_library/link_units/$UNIT.a
echo done
" ) >/dev/null 2>&1 && ok=$((ok+1))
done
echo "DEPLOY updated=$ok unreachable:$miss"

n=0; bad=""
for h in $ALL; do
    [ "$h" = "spark5" ] || echo "$miss" | grep -q "$h" && continue
    s=$(timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" "sha256sum < \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so; strings \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so | grep -c rail_peer_hosts" 2>/dev/null)
    ds=$(echo "$s" | head -1 | cut -d' ' -f1)
    rs=$(echo "$s" | tail -1)
    if [ "$ds" = "$DRV" ] && [ "${rs:-0}" -ge 1 ]; then n=$((n+1)); else bad="$bad $h(ds=$ds,rail=$rs)"; fi
done
echo "VERIFY $n/16 driver+rails bad:$bad"
[ "$n" -ge 15 ] && echo FLIP-DEPLOY-OK || echo FLIP-DEPLOY-PARTIAL
