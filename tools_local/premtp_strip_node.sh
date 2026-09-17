#!/usr/bin/env bash
set -e
cd ~/sparkdata/glm53flash.fp8.tp16/packs
r=$(ls *.sp 2>/dev/null | grep -vE "partial|restored|old|tmp|tail" | head -1 | sed 's/\.sp$//')
[ -n "$r" ] || { echo "$(hostname) no-pack"; exit 1; }
sz=$(stat -c %s "$r.sp")
if [ "$sz" -lt 22000000000 ]; then
    echo "$(hostname) already-premtp size=$sz"
else
    systemctl --user stop fleet-agent
    out=$(python3 /tmp/strip.py --pack "$r.sp" --family g5nsp --dry-run 2>&1) || { echo "$out"; echo "$(hostname) dry-fail"; exit 1; }
    echo "$out"
    stripped=$(echo "$out" | sed -n 's/^STRIP.* \([0-9][0-9]*\) bytes .*/\1/p' | head -1)
    [ -n "$stripped" ] || { echo "$(hostname) no-strip-line"; exit 1; }
    tail -c "$stripped" "$r.sp" > "$r.sp.mtp-tail.bin"
    python3 /tmp/strip.py --pack "$r.sp" --family g5nsp 2>&1 | tail -2
    sha256sum "$r.sp" > "$r.sp.sha256"
fi
echo "$(hostname) active=$(stat -c %s "$r.sp") sidecar=$(head -c 12 "$r.sp.sha256")"
if [ ! -f "$r.sp.experts" ]; then
    /tmp/gen_experts "$r.sp" > /tmp/gen.log 2>&1
    echo "$(hostname) manifest-rc=$? size=$(stat -c %s "$r.sp.experts" 2>/dev/null || echo missing)"
else
    echo "$(hostname) manifest-present"
fi
ls -l /proc/[0-9]*/exe 2>/dev/null | grep -q sparkpipe_weightd || \
    setsid nohup ~/sparkdata/weightd/sparkpipe_weightd --socket /tmp/spark_weightd.sock > ~/weightd.log 2>&1 < /dev/null &
sleep 1
rm -f ~/sparkdata/glm53flash.fp8.tp16/.last_boot
systemctl --user start fleet-agent
echo "$(hostname) agent-started"
