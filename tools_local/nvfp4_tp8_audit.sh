#!/bin/bash
# Audit the placed nvfp4 TP8 arm: every node holds its canonical rank
# pack, sha-matching its receipt, chattr-locked.
set -u
fail=0
for pair in spark0:00 spark1:01 spark2:02 spark3:03 spark4:04 spark5:05 \
            spark6:06 spark7:07 spark8:00 spark9:01 sparka:02 sparkb:03 \
            sparkc:04 sparkd:05 sparke:06 sparkf:07; do
  n=${pair%%:*}
  nn=${pair##*:}
  out=$(timeout 60 ssh -o ConnectTimeout=10 "$n" "
    p=\$HOME/sparkdata/qwenflash.tp8.nvfp4/packs/tp8-rank$nn.q4fsp
    if [ ! -f \"\$p\" ]; then echo \"missing \$p\"; exit 1; fi
    s=\$(sudo sha256sum \"\$p\" | awk '{print \$1}')
    rs=\$(python3 -c \"import json;print(json.load(open('\$p.receipt.json'))['output_sha256'])\" 2>/dev/null)
    l=\$(sudo lsattr \"\$p\" 2>/dev/null | awk '{print \$1}')
    case \"\$l\" in *i*) lock=yes ;; *) lock=NO ;; esac
    if [ -n \"\$s\" ] && [ \"\$s\" = \"\$rs\" ] && [ \$lock = yes ]; then
      echo \"OK rank=$nn sha=\${s:0:16} locked=\$lock\"
    else
      echo \"FAIL rank=$nn sha=\${s:0:16} receipt=\${rs:0:16} locked=\$lock\"
      exit 1
    fi" 2>&1 | tail -1)
  echo "$n: $out"
  case "$out" in OK*) ;; *) fail=$((fail+1)) ;; esac
done
echo "NVFP4 PLACEMENT AUDIT: $((16-fail))/16 PASS"
[ $fail -eq 0 ] && exit 0 || exit 1
