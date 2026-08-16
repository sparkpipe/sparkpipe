#!/usr/bin/env bash
# Regenerate each TP4xPP4 rank pack from the canonical full pack and compare
# its sha256 with the pack shipped on the rank's host. Byte-exact proof that
# every staged rank pack derives from the current full pack.
set -euo pipefail

HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
INPUT=/home/sparkb/sparkdata/dsv4_pro.full.spstage
WORK=/home/sparkb/sparkdata/dsv4_pro.regenwork
mkdir -p "$WORK"

rank=0
fail=0
for host in "${HOSTS[@]}"; do
    name="regen-rank$(printf '%02d' $rank).spstage"
    python3 - "$INPUT" "$WORK/$name" "$((rank % 4))" "$((rank / 4))" <<'PYEOF'
from pathlib import Path
import importlib.util
import sys
spec = importlib.util.spec_from_file_location('sh', '/home/sparkb/pro-repo/tools/dsv4_pro_tp16_stagepack.py')
sh = importlib.util.module_from_spec(spec)
sys.modules['sh'] = sh
spec.loader.exec_module(sh)
sh.TP_DEGREE = 4
result = sh.shard_pack(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), 4, int(sys.argv[4]))
print(result['sha256'])
PYEOF
    local_sha="$(sha256sum "$WORK/$name" | cut -d' ' -f1)"
    remote_sha="$(ssh -o BatchMode=yes $host "sha256sum /home/$host/sparkdata/dsv4_pro.tp4pp4/packs/dsv4_pro_tp4_pp4_stage.spstage | cut -d' ' -f1")"
    if [ "$local_sha" = "$remote_sha" ]; then
        echo "rank $rank MATCH $local_sha"
    else
        echo "rank $rank MISMATCH local=$local_sha remote=$remote_sha"
        fail=1
    fi
    rm -f "$WORK/$name"
    rank=$((rank + 1))
done
echo "REGEN-VERIFY exit=$fail"
exit $fail
