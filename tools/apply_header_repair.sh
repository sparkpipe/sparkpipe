#!/bin/sh
cd ~/hy4-fp8-packs || exit 1
[ $# -ge 1 ] || { echo "usage: apply_header_repair.sh 00 01 ..."; exit 1; }
CKPT=/mnt/model-warm/hy4-preview-fp8-official
for r in "$@"; do
  p="model-fp8-tp16-rank-$r.safetensors"
  if [ -f "repair_${r}.done" ]; then
    echo "$r already done"
    continue
  fi
  python3 hy4_fp8_header_repair2.py --pack "$p" --checkpoint "$CKPT" \
    --manifest "manifest-rank-$r.json" > "repair_${r}.prefix.bin" \
    2>> repair.log || { echo "$r STAGE1 FAIL"; exit 1; }
  sz=$(stat -c %s "repair_${r}.prefix.bin")
  dd if="repair_${r}.prefix.bin" of="$p" bs=1M conv=notrunc seek=0 \
    status=none || { echo "$r DD FAIL"; exit 1; }
  rm -f "repair_${r}.prefix.bin"
  sha256sum "$p" > "$p.sha256.tmp" && mv "$p.sha256.tmp" "$p.sha256" \
    || { echo "$r SHA FAIL"; exit 1; }
  newsha=$(cut -d' ' -f1 "$p.sha256")
  python3 - "manifest-rank-$r.json" "$newsha" > "manifest-rank-$r.json.tmp" \
    <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
m["file_sha256"] = sys.argv[2]
m["header_repaired"] = "empirical-offsets-v2"
print(json.dumps(m, indent=1))
PYEOF
  mv "manifest-rank-$r.json.tmp" "manifest-rank-$r.json" \
    || { echo "$r MANIFEST FAIL"; exit 1; }
  touch "repair_${r}.done"
  echo "$r done prefix_bytes=$sz $(cut -c1-16 "$p.sha256")" >> repair.log
done
echo REPAIR_APPLY_DONE >> repair.log
