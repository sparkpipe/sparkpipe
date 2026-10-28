#!/usr/bin/env python3
"""Append the expanded-MHA attention tensors to existing K3 rank packs.

The expanded prefill attention needs the publisher's un-absorbed query rows
and the key and value halves of kv_b_proj, which the K3 pack does not carry
(it stores the absorbed query projection). This tool adds, per MLA layer L:

  model.layers.L.mla_q_b_weight             q_b_proj rows of the rank's heads,
                                            [heads/16 * (nope + rope), q_lora]
  model.layers.L.mla_kv_b_key_full_weight   W_kb of every head, [heads * nope, kv_lora]
  model.layers.L.mla_kv_b_value_full_weight W_v of every head, [heads * v, kv_lora]

extract: read the checkpoint once and write one small named pack of these
tensors per rank. apply: rewrite one rank pack with the tensors appended to
its payload. Tensor offsets are payload-relative, so existing entries keep
their offsets. When the new manifest fits the space the old header reserved
(rank packs pad theirs to a 1 MiB payload base), it is space-padded into it
and the payload base and the .experts sidecar's absolute offsets stay as
they were; otherwise the sidecar ranges move by the growth of header plus
manifest. The digest file next to the pack is rewritten.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k3_pack import ALIGN, KIND_BF16, MAGIC, VERSION, PackFailure, SafetensorDir  # noqa: E402
from spark_named_pack import PayloadWriter, assemble, source_identity  # noqa: E402

TP = 16
EXPERTS_RECORD = 48
ADDED = ("mla_q_b_weight", "mla_kv_b_key_full_weight", "mla_kv_b_value_full_weight")


def read_pack(path):
    with open(path, "rb") as handle:
        magic, version, manifest_bytes = struct.unpack("<IIQ", handle.read(16))
        manifest = json.loads(handle.read(manifest_bytes))
    base = 16 + manifest_bytes
    base += (-base) % ALIGN
    return magic, version, manifest, base


def text_config(checkpoint):
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    return config.get("text_config", config)


def mla_layers(reader):
    layers = set()
    for name in reader.names():
        if name.endswith(".self_attn.kv_b_proj.weight"):
            layers.add((int(name.split(".layers.")[1].split(".")[0]), name.rsplit(".self_attn.", 1)[0] + ".self_attn."))
    return sorted(layers)


def extract(args):
    reader = SafetensorDir(args.checkpoint)
    config = text_config(args.checkpoint)
    heads, q_lora, kv_lora = config["num_attention_heads"], config["q_lora_rank"], config["kv_lora_rank"]
    nope, rope, value = config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"]
    if heads % TP != 0:
        raise PackFailure(f"{heads} heads do not split over {TP} ranks")
    rank_heads, layers = heads // TP, mla_layers(reader)
    if not layers:
        raise PackFailure("no MLA layers (self_attn.kv_b_proj) in the checkpoint")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer, prefix in layers:
        q_b = reader.bf16(prefix + "q_b_proj.weight", (heads * (nope + rope), q_lora))
        kv_b = reader.bf16(prefix + "kv_b_proj.weight", (heads * (nope + value), kv_lora))
        row = kv_lora * 2
        key = b"".join(kv_b[(h * (nope + value)) * row:(h * (nope + value) + nope) * row] for h in range(heads))
        values = b"".join(kv_b[(h * (nope + value) + nope) * row:(h + 1) * (nope + value) * row] for h in range(heads))
        tensors[layer] = (q_b, key, values)
    for rank in range(TP):
        path = out / f"k3.additions.rank{rank:02d}.pack"
        payload = out / f"k3.additions.rank{rank:02d}.pack.payload"
        writer = PayloadWriter(payload, source_identity(args.checkpoint, "k3-expanded-mha", rank), align=ALIGN)
        for layer, (q_b, key, values) in sorted(tensors.items()):
            q_row = q_lora * 2
            rows = q_b[rank * rank_heads * (nope + rope) * q_row:(rank + 1) * rank_heads * (nope + rope) * q_row]
            for name, data, shape in ((ADDED[0], rows, [rank_heads * (nope + rope), q_lora]),
                                      (ADDED[1], key, [heads * nope, kv_lora]),
                                      (ADDED[2], values, [heads * value, kv_lora])):
                writer.add(f"model.layers.{layer}.{name}", data, KIND_BF16, shape, {"shard_class": "replicated" if name != ADDED[0] else "output_dim_heads"})
        writer.close()
        assemble(path, payload, MAGIC, VERSION, ALIGN, {"name": "k3-expanded-mha-additions", "rank": rank},
                 {"layers": [layer for layer, _ in layers]}, writer.manifest)
        print(f"wrote {path} ({path.stat().st_size} bytes, {len(writer.manifest)} tensors)")


def apply(args):
    pack = Path(args.pack)
    magic, version, manifest, base = read_pack(pack)
    add_magic, _, additions, add_base = read_pack(args.additions)
    if magic != MAGIC or add_magic != MAGIC:
        raise PackFailure("not a K3 pack")
    tensors = dict(manifest["tensors"])
    if any(name in tensors for name in additions["tensors"]):
        raise PackFailure(f"{pack} already carries expanded-MHA tensors")
    payload_bytes = pack.stat().st_size - base
    cursor = payload_bytes + (-payload_bytes) % ALIGN
    for name, entry in sorted(additions["tensors"].items(), key=lambda item: item[1]["offset"]):
        moved = dict(entry)
        moved["offset"] = cursor
        tensors[name] = moved
        cursor += entry["bytes"] + (-entry["bytes"]) % ALIGN
    text = json.dumps({"format": manifest["format"], "config": manifest["config"], "tensors": tensors}, separators=(",", ":")).encode()
    reserved = base - 16
    if len(text) <= reserved:
        text += b" " * (reserved - len(text))
    new_base = 16 + len(text)
    new_base += (-new_base) % ALIGN
    shift = new_base - base
    target = Path(str(pack) + ".append")
    digest = hashlib.sha256()
    with open(pack, "rb") as source, open(args.additions, "rb") as added, open(target, "wb") as out:
        def emit(data):
            out.write(data)
            digest.update(data)
        emit(struct.pack("<IIQ", magic, version, len(text)) + text + b"\0" * (new_base - 16 - len(text)))
        source.seek(base)
        while True:
            chunk = source.read(1 << 26)
            if not chunk:
                break
            emit(chunk)
        emit(b"\0" * ((-payload_bytes) % ALIGN))
        for name, entry in sorted(additions["tensors"].items(), key=lambda item: item[1]["offset"]):
            added.seek(add_base + entry["offset"])
            emit(added.read(entry["bytes"]) + b"\0" * ((-entry["bytes"]) % ALIGN))
    experts = Path(str(pack) + ".experts")
    raw = bytearray(experts.read_bytes())
    count = struct.unpack_from("<I", raw, 8)[0]
    if len(raw) != 16 + count * EXPERTS_RECORD:
        raise PackFailure(f"{experts}: {len(raw)} bytes for {count} records")
    for index in range(count):
        at = 16 + index * EXPERTS_RECORD + 16
        struct.pack_into("<Q", raw, at, struct.unpack_from("<Q", raw, at)[0] + shift)
    Path(str(target) + ".experts").write_bytes(bytes(raw))
    Path(str(target) + ".sha256").write_text(digest.hexdigest() + "\n")
    print(json.dumps({"pack": str(pack), "staged": str(target), "added": len(additions["tensors"]), "payload_shift": shift,
                      "bytes": target.stat().st_size, "sha256": digest.hexdigest()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    one = sub.add_parser("extract")
    one.add_argument("--checkpoint", required=True)
    one.add_argument("--output-dir", required=True)
    two = sub.add_parser("apply")
    two.add_argument("--pack", required=True)
    two.add_argument("--additions", required=True)
    args = parser.parse_args()
    extract(args) if args.mode == "extract" else apply(args)


if __name__ == "__main__":
    main()
