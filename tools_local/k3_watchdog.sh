#!/usr/bin/env bash
# k3 watchdog: keep the stage pack running across external kills.
# The packer journals each tensor, so restarts resume instead of restarting.
FIRST="$1"
COUNT="$2"
USERNAME="$3"
PACK="/tmp/k3_stage_${FIRST}_${COUNT}.pack"
while true; do
    if [[ -s "$PACK" ]]; then
        echo "WATCHDOG-DONE $(hostname) ${FIRST}+${COUNT}" >> /tmp/k3_stage_pack.log
        exit 0
    fi
    if ! pgrep -f "^python3 /tmp/k3_pack_slice" >/dev/null 2>&1; then
        echo "WATCHDOG-RESTART $(date -u +%H:%M:%S)" >> /tmp/k3_stage_pack.log
        setsid -f bash -c "PYTHONDONTWRITEBYTECODE=1 python3 /tmp/k3_pack_slice.py /home/${USERNAME}/srcdata/kimi_k3.mxfp4.pp13 ${PACK} ${FIRST} ${COUNT} > /tmp/k3_stage_pack.log 2>&1" </dev/null
    fi
    sleep 30
done
