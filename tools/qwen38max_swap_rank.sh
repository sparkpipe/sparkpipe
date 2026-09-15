#!/bin/sh
# Rebuild-verify-swap: after tools/qwen38max_rebuild_rank.sh produced the
# verified pack + experts + receipt in /tmp/t1qmax_rebuild, swap the placed
# trio atomically (same filesystem) and archive the replaced digest.
set -eu
rank=$1
pack_dir=$HOME/sparkdata/qwenmax.nvfp4.tp16/packs
name=qwenmax.nvfp4.tp16.rank$rank
rebuilt=/tmp/t1qmax_rebuild/$name.sp
placed=$pack_dir/$name.sp
python3 - <<PY
import hashlib, json
old = json.load(open("$placed.receipt.json"))
new = json.load(open("$rebuilt.receipt.json"))
h = hashlib.sha256()
with open("$placed", "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
assert h.hexdigest() == old["output_sha256"], "placed pack drifted from its receipt"
print("old", h.hexdigest())
print("new", new["output_sha256"])
PY
mv "$placed" "/tmp/t1qmax_rebuild/$name.sp.replaced"
mv "$rebuilt" "$placed"
mv "$placed.experts" "/tmp/t1qmax_rebuild/$name.sp.experts.replaced" 2>/dev/null || true
mv "$placed.receipt.json" "/tmp/t1qmax_rebuild/$name.sp.receipt.json.replaced" 2>/dev/null || true
mv "/tmp/t1qmax_rebuild/$name.sp.experts" "$pack_dir/$name.sp.experts"
mv "/tmp/t1qmax_rebuild/$name.sp.receipt.json" "$pack_dir/$name.sp.receipt.json"
sudo -n /usr/local/sbin/sparkcap python3 /tmp/t1qmax_stage/tools/qwen38max_tp16_rank_verify.py \
	--pack "$placed" --tp-degree 16 --tp-rank "$rank" \
	--receipt "$placed.receipt.json" --recompute-file-hash 2>&1 | tail -4
