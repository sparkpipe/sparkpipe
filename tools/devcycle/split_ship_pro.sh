#!/usr/bin/env bash
# Split pro rank packs one at a time, shipping each to its host and deleting
# the local copy so sparkb's disk stays bounded.
set -euo pipefail

HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
INPUT=/home/sparkb/sparkdata/dsv4_pro.full.spstage
WORK=/home/sparkb/sparkdata/dsv4_pro.rankwork
mkdir -p "$WORK"

rank=0
for host in "${HOSTS[@]}"; do
    name="dsv4_pro.tp4_pp4.rank$(printf '%02d' $rank).spstage"
    echo "splitting $name (tp=$((rank % 4)) pp=$((rank / 4))) ..."
    python3 - <<PYEOF
from pathlib import Path
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('sh', '/home/sparkb/pro-repo/tools/dsv4_pro_tp16_stagepack.py')
sh = importlib.util.module_from_spec(spec)
sys.modules['sh'] = sh
spec.loader.exec_module(sh)
sh.TP_DEGREE = 4
result = sh.shard_pack(Path('$INPUT'), Path('$WORK/$name'), $((rank % 4)), 4, $((rank / 4)))
print('shard', $rank, 'ok', str(result.get('sha256', '?'))[:16])
PYEOF
    scp -q -o BatchMode=yes "$WORK/$name" "${host}:/home/${host}/sparkdata/dsv4_pro.tp4pp4/packs/dsv4_pro_tp4_pp4_stage.spstage"
    rm -f "$WORK/$name"
    echo "shipped ${name} -> ${host}"
    rank=$((rank + 1))
done
echo "SPLIT-SHIP-ALL-DONE"
