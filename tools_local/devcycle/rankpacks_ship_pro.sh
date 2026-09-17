#!/usr/bin/env bash
# rankpacks_ship_pro.sh — run ON spark3 after the full pack completes.
# Shards the full GA pack one TP4xPP4 rank at a time and ships each rank
# pack straight to its host, keeping the local footprint at one rank pack.
set -euo pipefail

FULL=/home/spark3/pro-repo/dsv4_pro_ga.full.spstage
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
WORK=/home/spark3/extnvme/rankpacks-ga
mkdir -p "$WORK"

rank=0
failed=0
for host in "${HOSTS[@]}"; do
    if ! ssh -o BatchMode=yes -o ConnectTimeout=6 "$host" "echo ok" >/dev/null 2>&1; then
        echo "rank $rank -> $host SKIPPED (unreachable)"
        rank=$((rank + 1))
        failed=$((failed + 1))
        continue
    fi
    name="dsv4_pro.tp4_pp4.rank$(printf '%02d' $rank).spstage"
    python3 - "$FULL" "$WORK/$name" "$((rank % 4))" "$((rank / 4))" <<'PYEOF'
from pathlib import Path
import importlib.util, sys
spec = importlib.util.spec_from_file_location('sh', '/home/spark3/pro-repo/tools/dsv4_pro_tp16_stagepack.py')
sh = importlib.util.module_from_spec(spec)
sys.modules['sh'] = sh
spec.loader.exec_module(sh)
sh.TP_DEGREE = 4
result = sh.shard_pack(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), 4, int(sys.argv[4]))
print(result['sha256'])
PYEOF
    ssh -o BatchMode=yes "$host" "mkdir -p /home/$host/sparkdata/dsv4_pro.tp4pp4/packs"
    scp -q -o BatchMode=yes "$WORK/$name" "$host:/home/$host/sparkdata/dsv4_pro.tp4pp4/packs/dsv4_pro_tp4_pp4_stage.spstage"
    rm -f "$WORK/$name"
    echo "rank $rank -> $host done"
    rank=$((rank + 1))
done
echo "RANKS-SHIPPED failed=$failed"
