#!/bin/bash
R=$1
L=/tmp/mb_db_r${R}.log
echo "== rank $R op0: $(grep -E "OP-DUMP rank=$R credit=0 " $L 2>/dev/null | head -1 | grep -oE "step=[0-9]+ .*receive_complete=[0-9a-f]+" | head -c 46)"
echo "   sends posted (SFX-ENTER by session):"
grep "SFX-ENTER" $L 2>/dev/null | sed -E 's/.*route=tp-device\.[0-9a-f]+\.([0-9]+)\.[0-9]+\.[0-9]+ .*/\1/' | sort | uniq -c | sed 's/^/     step\1x/'
echo "   doorbells recv (fixed_recv by session-step, with seq-route bits):"
grep "fixed_recv" $L 2>/dev/null | sed -E 's/.*route=tp-device\.[0-9a-f]+\.([0-9]+)\.[0-9]+\.[0-9]+ seq=([0-9]+) .*/\1 \2/' | awk '{s[$1]++; r[$1" route"(($2%256)%4)]++} END {for (k in s) printf "     step%s: %d doorbells\n", k, s[k]; for (k in r) print "     "k"="r[k]}' | sort
