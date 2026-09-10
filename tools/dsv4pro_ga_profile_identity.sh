#!/bin/bash
set -euo pipefail
MODEL=/mnt/model-warm/deepseek-v4-pro-0813-ga
python3 - "$MODEL" <<'PYEOF'
import hashlib, json, os, re, sys
model = sys.argv[1]

def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

identity = {
    "index": "model.safetensors.index.json",
    "config": "config.json",
    "tokenizer": "tokenizer.json",
    "reference_model": "inference/model.py",
    "reference_kernel": "inference/kernel.py",
    "reference_config": "inference/config.json",
}
shas = {}
for label, name in identity.items():
    path = os.path.join(model, name)
    shas[label] = sha(path)
    print("PRO-IDENTITY %s %s %d" % (label, shas[label], os.path.getsize(path)))

for receipt in ("ARCHIVE-RECEIPT.json", "DOWNLOAD_STATUS.json"):
    path = os.path.join(model, receipt)
    if os.path.exists(path):
        print("PRO-RECEIPT %s %s" % (receipt, open(path).read()[:800]))

index = json.load(open(os.path.join(model, "model.safetensors.index.json")))
weight_map = index["weight_map"]
wanted = re.compile(r"^(embed\.|layers\.[012]\.)")
needed_shards = sorted({shard for name, shard in weight_map.items() if wanted.match(name)})
print("PRO-SHARDS %s" % " ".join(needed_shards))
for shard in needed_shards:
    path = os.path.join(model, shard)
    print("PRO-SHARD %s %d %s" % (shard, os.path.getsize(path), sha(path)))

config = json.load(open(os.path.join(model, "inference", "config.json")))
geometry = {k: config[k] for k in ("vocab_size", "dim", "hc_mult", "n_heads", "head_dim",
    "n_routed_experts", "n_activated_experts", "n_layers", "n_hash_layers", "n_mtp_layers",
    "dspark_block_size", "dspark_markov_rank", "dspark_target_layer_ids") if k in config}
print("PRO-GEOMETRY %s" % json.dumps(geometry, sort_keys=True))
PYEOF
echo "PRO-IDENTITY-RC=0"
