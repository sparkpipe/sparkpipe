import hashlib
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

pack_path, checkpoint, contract = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
with pack_path.open("rb") as pack:
    magic, version, manifest_bytes = struct.unpack("<IIQ", pack.read(16))
    manifest = json.loads(pack.read(manifest_bytes))
    base = (16 + manifest_bytes + 127) // 128 * 128
    tensors = manifest["tensors"]
    handles, keys = [], {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        handle = safe_open(str(shard), framework="pt")
        handles.append(handle)
        for key in handle.keys():
            keys[key] = handle
    prefix = "model." if any(key.startswith("model.") for key in keys) else ""

    def source(name):
        return keys[prefix + name].get_tensor(prefix + name)

    def bf16(*names):
        return b"".join(source(name).contiguous().view(torch.int16).numpy().tobytes() for name in names)

    def f32(name):
        return source(name).float().contiguous().numpy().tobytes()

    def stored(name):
        entry = tensors[name]
        pack.seek(base + entry["offset"])
        return pack.read(entry["bytes"]), entry

    layers = sum(1 for name in tensors if name.endswith("attn_qkv"))
    stem = next(name for name in tensors if name.endswith("embed"))[:-len("embed")]
    expected = {stem + "embed": lambda: bf16("embed_tokens.weight"), stem + "head_norm": lambda: f32("norm.weight")}
    for layer in range(layers):
        name = next(key for key in tensors if key.endswith(f".{layer}.attn_qkv") or key.endswith(f"layers.{layer}.attn_qkv"))[:-len("attn_qkv")]
        at = f"layers.{layer}."
        expected.update({
            name + "attn_qkv": lambda at=at: bf16(at + "self_attn.q_proj.weight", at + "self_attn.k_proj.weight", at + "self_attn.v_proj.weight"),
            name + "attn_out": lambda at=at: bf16(at + "self_attn.o_proj.weight"),
            name + "attn_q_norm": lambda at=at: f32(at + "self_attn.q_norm.weight"),
            name + "attn_k_norm": lambda at=at: f32(at + "self_attn.k_norm.weight"),
            name + "ffn_gate_up": lambda at=at: bf16(at + "mlp.gate_proj.weight", at + "mlp.up_proj.weight"),
            name + "ffn_down": lambda at=at: bf16(at + "mlp.down_proj.weight"),
            name + "input_norm": lambda at=at: f32(at + "input_layernorm.weight"),
            name + "post_norm": lambda at=at: f32(at + "post_attention_layernorm.weight"),
        })
    wrong, total = [], 0
    for name, make in expected.items():
        have, entry = stored(name)
        want = make()
        total += len(have)
        if have != want:
            wrong.append({"tensor": name, "pack_bytes": len(have), "checkpoint_bytes": len(want), "kind": entry.get("kind"), "shape": entry.get("shape")})
    unused = sorted(set(keys) - {prefix + key for key in
                    ["embed_tokens.weight", "norm.weight"] + [f"layers.{layer}.{part}" for layer in range(layers) for part in (
                        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight",
                        "self_attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight",
                        "post_attention_layernorm.weight")]})
    print(json.dumps({"step": "pack", "magic": hex(magic), "version": version, "format": manifest.get("format"), "config": manifest.get("config"),
                      "layers": layers, "pack_tensors": len(tensors), "compared": len(expected), "pack_tensors_not_compared": sorted(set(tensors) - set(expected))[:8],
                      "bytes_compared": total, "different": wrong[:8], "checkpoint_tensors": len(keys), "checkpoint_tensors_not_in_pack": unused[:8],
                      "contract_sha256_now": hashlib.sha256(contract.read_bytes()).hexdigest(), "first_names": sorted(tensors)[:3]}))
