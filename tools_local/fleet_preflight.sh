#!/usr/bin/env bash
# fleet_preflight.sh — verify every node can launch THIS release before we
# launch it. Deterministic checks against the hub snapshot; fails loud with
# node + reason. Green preflight -> sub-minute launch.
#
# usage: tools/fleet_preflight.sh HUB_REF ROOT_NAME
set -uo pipefail
REF="${1:?hub reference (e.g. rtx5090:release)}"
NAME="${2:?runtime root name}"
PACK_TPL="${3:-packs/$NAME.rank%x.sp}"
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=6"
HUBHOST="${REF%%:*}"
HUBPATH="${REF#*:}"

DRV=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$HUBHOST" "md5sum '$HUBPATH/$NAME/stages/stage_000/model_driver.so' 2>/dev/null" | cut -d' ' -f1)
BIN=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$HUBHOST" "md5sum '$HUBPATH/$NAME/bin/sparkpipe_model_residentd' 2>/dev/null" | cut -d' ' -f1)
CFG=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$HUBHOST" "md5sum '$HUBPATH/$NAME/config/stage_00.json' 2>/dev/null" | cut -d' ' -f1)
[ -n "$DRV" ] || { echo "PREFLIGHT-FAIL hub driver missing at $REF"; exit 1; }
[ -n "$BIN" ] || { echo "PREFLIGHT-FAIL hub residentd missing at $REF"; exit 1; }

FAIL=0
i=0
for h in "${HOSTS[@]}"; do
    PACK=$(printf "$PACK_TPL" "$i")
    CFG=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$HUBHOST" "md5sum '$HUBPATH/$NAME/config/stage_$(printf %02d $i).json' 2>/dev/null" | cut -d' ' -f1)
    OUT=$($SSH "$h" "
        rr=\$HOME/sparkdata/$NAME
        [ \"\$(md5sum \$rr/stages/stage_000/model_driver.so 2>/dev/null | cut -d' ' -f1)\" = '$DRV' ] || echo driver-stale
        [ \"\$(md5sum \$rr/bin/sparkpipe_model_residentd 2>/dev/null | cut -d' ' -f1)\" = '$BIN' ] || echo residentd-stale
        [ \"\$(md5sum \$rr/config/stage_$(printf %02d $i).json 2>/dev/null | cut -d' ' -f1)\" = '$CFG' ] || echo config-stale
        [ \"\$(readlink \$rr/config/stage.json 2>/dev/null)\" = 'stage_$(printf %02d $i).json' ] || echo symlink-wrong
        [ -f \"\$rr/$PACK\" ] || echo pack-missing
        [ \"\$(od -An -tx1 -N4 \$rr/$PACK 2>/dev/null | tr -d ' ')\" = '474c5833' ] || echo pack-badmagic
        l=\$(ss -tln 2>/dev/null | awk '{print \$4}' | grep -oE '[0-9]+\$' | sort -un | awk '\$1>=61440 && \$1<=62000' | wc -l)
        [ \"\$l\" = '0' ] || echo ports-busy:\$l
        a=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo)
        [ \"\$a\" -ge 30 ] || echo memory-low:\${a}G
    " 2>/dev/null)
    if [ -n "$OUT" ]; then
        echo "PREFLIGHT-FAIL $h: $(echo "$OUT" | tr '\n' ' ')"
        FAIL=1
    fi
    i=$((i+1))
done
[ "$FAIL" = 0 ] && echo "PREFLIGHT-OK 16/16 (driver ${DRV:0:8} residentd ${BIN:0:8})"
exit $FAIL
