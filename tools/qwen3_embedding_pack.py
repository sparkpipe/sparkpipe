#!/usr/bin/env python3
"""Qwen3 dense checkpoints without a vocabulary head (the Qwen3-Embedding models) -> one whole-model named stage pack.

The geometry comes from the checkpoint's config.json. The publisher's
modules run in bfloat16: a norm rounds its normalised value to BF16 and
multiplies by its BF16 weight, so the norm weights are stored as the F32
values of the checkpoint's BF16 weights, with nothing folded in.

Output: <out>/<name>.tp1.rank00.pack (tools/spark_named_pack.py container,
magic SPNP), plus <pack>.sha256 and the empty-range <pack>.experts that the
weightd lazy attach reads for a dense pack. The pack holds:
  embed                         vocab rows, bf16
  head_norm                     the final norm weight, f32
  layers.N.attn_qkv             q_proj|k_proj|v_proj fused by rows, bf16
  layers.N.attn_out             o_proj, bf16
  layers.N.attn_q_norm, attn_k_norm   per-head norm weights, f32
  layers.N.ffn_gate_up          gate_proj|up_proj fused by rows, bf16
  layers.N.ffn_down             down_proj, bf16
  layers.N.input_norm, post_norm      f32
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, PayloadWriter, assemble, source_identity  # noqa: E402

EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2
EPSILON = 1e-6


class PackFailure(RuntimeError):
    pass


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        index = json.loads((self.directory / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.headers = {}

    def _header(self, shard):
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        return self.headers[shard]

    def bf16(self, name, shape):
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"checkpoint has no tensor {name}")
        base, header = self._header(shard)
        meta = header[name]
        if meta["dtype"] != "BF16" or tuple(meta["shape"]) != tuple(shape):
            raise PackFailure(f"{name}: expected BF16 {tuple(shape)}, found {meta['dtype']} {tuple(meta['shape'])}")
        begin, end = meta["data_offsets"]
        flat = np.memmap(self.directory / shard, dtype=np.uint16, mode="r", offset=base + begin, shape=((end - begin) // 2,))
        return flat.reshape(meta["shape"])


def bf16_to_f32(raw):
    return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)


def geometry(config):
    required = {"model_type": "qwen3", "hidden_act": "silu", "attention_bias": False, "rms_norm_eps": EPSILON, "rope_scaling": None,
                "use_sliding_window": False}
    for key, value in required.items():
        if config.get(key) != value:
            raise PackFailure(f"checkpoint {key} is {config.get(key)!r}; the stage is built for {value!r}")
    shape = {key: int(config[key]) for key in ("hidden_size", "num_hidden_layers", "vocab_size", "intermediate_size", "num_attention_heads",
                                               "num_key_value_heads", "head_dim", "rope_theta")}
    if shape["num_attention_heads"] % shape["num_key_value_heads"] or shape["head_dim"] % 2 or shape["intermediate_size"] % 128:
        raise PackFailure(f"the geometry {shape} is not grouped-query attention with whole rotary pairs and 128-wide FFN blocks")
    return shape


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--name", required=True, help="the pack's model name and file prefix, such as qwen3_embedding_8b")
    parser.add_argument("--revision", required=True, help="the checkpoint revision the files were fetched at")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    shape = geometry(json.loads((args.checkpoint / "config.json").read_text()))
    contract = json.loads(args.contract.read_text())
    declared = {"hidden_size": contract["model"]["hidden_dimension"], "num_hidden_layers": contract["model"]["layer_count"],
                "vocab_size": contract["model"]["vocabulary_size"], "intermediate_size": contract["model"]["dense_intermediate_dimension"],
                "num_attention_heads": contract["attention"]["query_head_count"], "num_key_value_heads": contract["attention"]["kv_head_count"],
                "head_dim": contract["attention"]["head_dimension"], "rope_theta": int(contract["attention"]["rope_theta"])}
    if declared != shape or contract["revision"] != args.revision:
        raise PackFailure(f"the checkpoint {shape} at {args.revision} is not the contract's {declared} at {contract['revision']}")
    hidden, layers, vocab, ffn = shape["hidden_size"], shape["num_hidden_layers"], shape["vocab_size"], shape["intermediate_size"]
    query, kv, head_dim = shape["num_attention_heads"] * shape["head_dim"], shape["num_key_value_heads"] * shape["head_dim"], shape["head_dim"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint)
    payload = args.output_dir / f"{args.name}.tp1.rank00.pack.payload"
    final = args.output_dir / f"{args.name}.tp1.rank00.pack"
    writer = PayloadWriter(payload, source_identity(args.checkpoint, "qwen3_embedding_pack", files=(args.contract,)), resume=args.resume and payload.exists())

    def add(name, array, kind):
        array = np.ascontiguousarray(array)
        writer.add(name, array.tobytes(), kind, array.shape)

    add("embed", checkpoint.bf16("embed_tokens.weight", (vocab, hidden)), "bf16")
    add("head_norm", bf16_to_f32(checkpoint.bf16("norm.weight", (hidden,))), "f32")
    for layer in range(layers):
        source, name = f"layers.{layer}.", f"layers.{layer}."
        add(name + "attn_qkv", np.concatenate((checkpoint.bf16(source + "self_attn.q_proj.weight", (query, hidden)),
                                               checkpoint.bf16(source + "self_attn.k_proj.weight", (kv, hidden)),
                                               checkpoint.bf16(source + "self_attn.v_proj.weight", (kv, hidden)))), "bf16")
        add(name + "attn_out", checkpoint.bf16(source + "self_attn.o_proj.weight", (hidden, query)), "bf16")
        add(name + "attn_q_norm", bf16_to_f32(checkpoint.bf16(source + "self_attn.q_norm.weight", (head_dim,))), "f32")
        add(name + "attn_k_norm", bf16_to_f32(checkpoint.bf16(source + "self_attn.k_norm.weight", (head_dim,))), "f32")
        add(name + "ffn_gate_up", np.concatenate((checkpoint.bf16(source + "mlp.gate_proj.weight", (ffn, hidden)),
                                                  checkpoint.bf16(source + "mlp.up_proj.weight", (ffn, hidden)))), "bf16")
        add(name + "ffn_down", checkpoint.bf16(source + "mlp.down_proj.weight", (hidden, ffn)), "bf16")
        add(name + "input_norm", bf16_to_f32(checkpoint.bf16(source + "input_layernorm.weight", (hidden,))), "f32")
        add(name + "post_norm", bf16_to_f32(checkpoint.bf16(source + "post_attention_layernorm.weight", (hidden,))), "f32")
        print(f"layer {layer} packed", flush=True)
    writer.close()
    config = {"model": args.name, "tp_degree": 1, "tp_rank": 0, "hidden": hidden, "layers": layers, "vocab": vocab, "vocab_first": 0,
              "vocab_rows": vocab, "ffn_first": 0, "ffn_rows": ffn}
    fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN, "linear_codec": "bf16",
           "revision": args.revision, "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest()}
    assemble(final, payload, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, config, writer.manifest)
    digest = hashlib.sha256()
    with final.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    Path(str(final) + ".sha256").write_text(f"{digest.hexdigest()}  {final.name}\n")
    Path(str(final) + ".experts").write_bytes(struct.pack("<4I", EXPERTS_MAGIC, EXPERTS_VERSION, 0, 0))
    print(f"rank 00 {final.stat().st_size} {digest.hexdigest()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen3_embedding_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
