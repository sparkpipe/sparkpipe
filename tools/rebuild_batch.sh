#!/bin/sh
cd ~/hy4-fp8-packs || exit 1
CKPT=/mnt/model-warm/hy4-preview-fp8-official
for r in "$@"; do
  p="model-fp8-tp16-rank-$r.safetensors"
  if [ -f "rb_${r}.done" ]; then
    echo "$r done"
    continue
  fi
  python3 hy4_fp8_rank_rebuild.py --pack "$p" --out "rb_${r}.tmp" \
    --checkpoint "$CKPT" --manifest "manifest-rank-$r.json" \
    2> "rb_${r}.log" || { echo "$r REBUILD FAIL"; tail -3 "rb_${r}.log"; exit 1; }
  mv "rb_${r}.tmp" "$p" || exit 1
  sha256sum "$p" > "$p.sha256.tmp" && mv "$p.sha256.tmp" "$p.sha256" || exit 1
  newsha=$(cut -d' ' -f1 "$p.sha256")
  python3 - "manifest-rank-$r.json" "$newsha" > "m_${r}.tmp" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
m["file_sha256"] = sys.argv[2]
m["header_repaired"] = "empirical-rebuild-v3"
print(json.dumps(m, indent=1))
PYEOF
  mv "m_${r}.tmp" "manifest-rank-$r.json" || exit 1
  touch "rb_${r}.done"
  echo "$r rebuilt $(cut -c1-16 "$p.sha256")" >> rebuild.log
done
echo REBUILD_BATCH_DONE >> rebuild.log
