#!/usr/bin/env bash
# g5n_repack_place.sh — verify a rebuilt rank pack against the checkpoint,
# generate its .experts manifest, and place it in the serving root.
# usage: g5n_repack_place.sh <rank>
set -euo pipefail
RANK=${1:?usage: g5n_repack_place.sh <rank>}
WORK="$HOME/g5n_repack"
SRC="$WORK/out/glm5_next_stage.tp16.rank$RANK.g5nsp"
ROOT="$HOME/sparkdata/glm53flash.fp8.tp16/packs"
DEST="$ROOT/glm53flash.fp8.tp16.rank$RANK.sp"
[ -f "$SRC" ] || { echo "rank $RANK: pack missing"; exit 1; }
grep -q "tensors, .* bytes" "$WORK/build.log" || { echo "rank $RANK: build not complete"; exit 1; }

python3 - "$SRC" "$RANK" <<'PYEOF'
import struct, sys
import numpy as np
import json
from pathlib import Path
pack_path, rank = sys.argv[1], int(sys.argv[2])
f = open(pack_path, "rb")
f.seek(80)
doff, _ = struct.unpack("<QQ", f.read(16))
f.seek(doff)
tensors = {}
while True:
    raw = f.read(64)
    if len(raw) < 64:
        break
    kind, layer = struct.unpack("<II", raw[:8])
    if kind == 0 or kind > 64:
        break
    tensors[(kind, layer)] = struct.unpack("<QQ", raw[48:64])
d = Path("/mnt/model-warm/glm-5.3-flash")
idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
def raw_of(name):
    shard = idx[name]
    p = d / shard
    with p.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    b, e = h[name]["data_offsets"]
    mm = np.memmap(p, dtype=np.uint8, mode="r")
    return np.array(mm[8 + n + b:8 + n + e])
def scale_of(name):
    return np.frombuffer(raw_of(name + "_scale_inv").tobytes(), dtype=np.float32)
checked = 0
for layer in (3, 20, 44):
    pre = f"model.language_model.layers.{layer}.mlp.experts"
    w1_off, _ = tensors[(22, layer)]
    w2_off, _ = tensors[(23, layer)]
    e_w1 = 2 * 128 * 32 * 4
    e_w2 = 4096 * 4
    for expert in (0, 43, 287):
        up = np.repeat(scale_of(f"{pre}.{expert}.up_proj.weight").reshape(-1, 32), 128, 0)[rank*128:(rank+1)*128]
        gate = np.repeat(scale_of(f"{pre}.{expert}.gate_proj.weight").reshape(-1, 32), 128, 0)[rank*128:(rank+1)*128]
        exp_w1 = np.concatenate([up, gate], 0).tobytes()
        f.seek(w1_off + expert * e_w1)
        assert f.read(e_w1) == exp_w1, f"layer {layer} expert {expert} w1 scale mismatch"
        down = scale_of(f"{pre}.{expert}.down_proj.weight").reshape(32, 16)
        exp_w2 = np.repeat(down, 128, 0)[0:4096, rank:rank+1].tobytes()
        f.seek(w2_off + expert * e_w2)
        assert f.read(e_w2) == exp_w2, f"layer {layer} expert {expert} w2 scale mismatch"
        checked += 2
print(f"rank {rank}: {checked} scale planes byte-verified vs checkpoint")
PYEOF

rm -f "$SRC.experts"
"$WORK/g5n_experts_manifest" "$SRC"
[ -f "$SRC.experts" ] || { echo "rank $RANK: manifest generation failed"; exit 1; }

mkdir -p "$ROOT"
sudo -n chattr -i "$DEST" "$DEST.experts" 2>/dev/null || true
if [ -f "$DEST" ]; then
    mv "$DEST" "$DEST.old"
    mv "$DEST.experts" "$DEST.experts.old" 2>/dev/null || true
fi
mv "$SRC" "$DEST"
mv "$SRC.experts" "$DEST.experts"
sudo -n chattr +i "$DEST" "$DEST.experts"
SHA=$(sha256sum "$DEST" | cut -d" " -f1)
SIZE=$(stat -c%s "$DEST")
printf "%s  %s\n" "$SHA" "$(basename "$DEST")" > "$DEST.sha256"
python3 - "$RANK" "$SHA" "$SIZE" <<'PYEOF'
import json, os, sys
rank, sha, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
receipt = {
    "kind": "sparkpipe.g5nsp.stagepack-receipt.v1",
    "arm": "glm53flash.fp8.tp16",
    "rank": int(rank),
    "tp_degree": 16,
    "first_layer": 0,
    "layer_count": 45,
    "mtp": "none",
    "expert_codec": "fp8-source-native",
    "file_bytes": size,
    "output_sha256": sha,
    "packed_by": "scale-plane-rebuild-0911",
}
path = os.path.join(os.environ["HOME"],
    "sparkdata/glm53flash.fp8.tp16/packs",
    f"glm53flash.fp8.tp16.rank{rank}.sp.receipt.json")
open(path, "w").write(json.dumps(receipt))
PYEOF
rm -f /tmp/spark-weightd-spine/*.receipt 2>/dev/null || true
echo "rank $RANK: placed $DEST ($SIZE bytes, sha ${SHA:0:12})"
