#!/usr/bin/env bash
cd ~/sparkdata/glm53flash.fp8.tp16/packs || exit 1
for l in $(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_weightd | sed 's|.*/proc/\([0-9]*\)/exe.*|\1|'); do
    kill -9 "$l" 2>/dev/null
done
sleep 2
r=$(ls *.sp 2>/dev/null | grep -vE "partial|restored|old" | head -1 | sed 's/\.sp$//')
[ -n "$r" ] || { echo no-pack; exit 1; }
sz=$(stat -c %s "$r.sp")
swapped=0
if [ "$sz" -gt 22000000000 ]; then
    chattr -i "$r.sp" "$r.sp.premtp-old" 2>/dev/null
    mv "$r.sp" "$r.sp.mtp-restored"
    mv "$r.sp.premtp-old" "$r.sp"
    chattr +i "$r.sp" 2>/dev/null
    sha256sum "$r.sp" > "$r.sp.sha256"
    rm -f "$r.sp.experts"
    swapped=1
fi
echo "$(hostname) active=$(stat -c %s "$r.sp") sidecar=$(head -c 12 "$r.sp.sha256")"
if [ "$swapped" = 1 ] || [ ! -f "$r.sp.experts" ]; then
    /tmp/gen_experts "$r.sp" > /tmp/gen.log 2>&1
    echo "manifest-gen rc=$? size=$(stat -c %s "$r.sp.experts" 2>/dev/null || echo missing)"
else
    echo manifest-present
fi
setsid nohup ~/sparkdata/weightd/sparkpipe_weightd --socket /tmp/spark_weightd.sock > ~/weightd.log 2>&1 < /dev/null &
rm -f ~/sparkdata/glm53flash.fp8.tp16/.last_boot
