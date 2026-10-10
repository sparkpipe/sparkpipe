#!/usr/bin/env python3
"""Gemma 4 31B (dense) BF16 checkpoint -> sixteen TP16 named stage packs.

Output per rank: <out>/gemma4.tp16.rank<NN>.pack (tools/spark_named_pack.py
container), plus <pack>.sha256 and the empty-range <pack>.experts.

Rank r holds, all bf16 unless noted:
  embed               vocab rows [r*16384, +16384) of the tied embedding
                      (the runner's head reads the same rows)
  head_norm           the final norm weight as f32 (plain w, Gemma4RMSNorm)
  per layer           input / post-attention / pre-feedforward /
                      post-feedforward norms as f32 (plain w) and the layer
                      scalar as f32 [1]
  attention           q_proj|k_proj|v_proj fused (sliding: 16384 rows,
                      full: q_proj|k_proj 18432 rows, no v_proj), rank rows
                      [r*rows/16, +rows/16); q and k norms f32; o_proj
                      columns [r*q/16, +q/16)
  MLP                 gate|up rows [r*1344, +1344) fused [2688, H]; down
                      columns [H, 1344]
The vision and audio towers are not packed.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, PayloadWriter, assemble  # noqa: E402

TP = 16
PREFIX = "model.language_model."
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2


class PackFailure(RuntimeError):
    pass


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def array(self, name, shape):
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"checkpoint has no tensor {name}")
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        base, header = self.headers[shard]
        meta = header[name]
        if meta["dtype"] != "BF16" or tuple(meta["shape"]) != tuple(shape):
            raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected BF16 {list(shape)}")
        begin, end = meta["data_offsets"]
        return np.memmap(self.directory / shard, dtype=np.uint16, mode="r", offset=base + begin,
                         shape=((end - begin) // 2,)).reshape(shape)

    def f32(self, name, shape):
        return (np.asarray(self.array(name, shape)).astype(np.uint32) << 16).view(np.float32)


class RankPacks:
    def __init__(self, out_dir, resume, ranks):
        self.out_dir = Path(out_dir)
        self.ranks = ranks
        self.writers = {rank: PayloadWriter(self.payload(rank), resume=resume and self.payload(rank).exists()) for rank in ranks}

    def payload(self, rank):
        return self.out_dir / f"gemma4.tp16.rank{rank:02d}.pack.payload"

    def final(self, rank):
        return self.out_dir / f"gemma4.tp16.rank{rank:02d}.pack"

    def add(self, rank, name, array, kind):
        if rank in self.writers:
            array = np.ascontiguousarray(array)
            self.writers[rank].add(name, array.tobytes(), kind, array.shape)

    def replicated(self, name, array, kind):
        for rank in self.ranks:
            self.add(rank, name, array, kind)


def pack_layer(checkpoint, packs, text, layer):
    hidden = text["hidden_size"]
    p = f"{PREFIX}layers.{layer}."
    name = f"layers.{layer}."
    full = text["layer_types"][layer] == "full_attention"
    head_dim = text["global_head_dim"] if full else text["head_dim"]
    heads = text["num_attention_heads"]
    kv_heads = text["num_global_key_value_heads"] if full else text["num_key_value_heads"]
    for norm, key in (("input_norm", "input_layernorm"), ("post_attn_norm", "post_attention_layernorm"),
                      ("pre_ff_norm", "pre_feedforward_layernorm"), ("post_ff_norm", "post_feedforward_layernorm")):
        packs.replicated(name + norm, checkpoint.f32(p + key + ".weight", (hidden,)), "f32")
    packs.replicated(name + "layer_scalar", checkpoint.f32(p + "layer_scalar", (1,)), "f32")
    q = checkpoint.array(p + "self_attn.q_proj.weight", (heads * head_dim, hidden))
    k = checkpoint.array(p + "self_attn.k_proj.weight", (kv_heads * head_dim, hidden))
    parts = [q, k]
    if not (full and text["attention_k_eq_v"]):
        parts.append(checkpoint.array(p + "self_attn.v_proj.weight", (kv_heads * head_dim, hidden)))
    fused = np.concatenate(parts)
    o = checkpoint.array(p + "self_attn.o_proj.weight", (hidden, heads * head_dim))
    if fused.shape[0] % TP or o.shape[1] % TP:
        raise PackFailure(f"layer {layer}: fused rows {fused.shape[0]} or o_proj columns {o.shape[1]} do not divide {TP}")
    rows, columns = fused.shape[0] // TP, o.shape[1] // TP
    packs.replicated(name + "attn_q_norm", checkpoint.f32(p + "self_attn.q_norm.weight", (head_dim,)), "f32")
    packs.replicated(name + "attn_k_norm", checkpoint.f32(p + "self_attn.k_norm.weight", (head_dim,)), "f32")
    inter = text["intermediate_size"]
    gate = checkpoint.array(p + "mlp.gate_proj.weight", (inter, hidden))
    up = checkpoint.array(p + "mlp.up_proj.weight", (inter, hidden))
    down = checkpoint.array(p + "mlp.down_proj.weight", (hidden, inter))
    ffn = inter // TP
    for rank in packs.ranks:
        packs.add(rank, name + "attn_qkv", fused[rank * rows:(rank + 1) * rows], "bf16")
        packs.add(rank, name + "attn_out", o[:, rank * columns:(rank + 1) * columns], "bf16")
        window = slice(rank * ffn, (rank + 1) * ffn)
        packs.add(rank, name + "ffn_gate_up", np.concatenate((gate[window], up[window])), "bf16")
        packs.add(rank, name + "ffn_down", down[:, window], "bf16")


def finish(packs, text, revision, contract_sha256):
    vocab_rows = text["vocab_size"] // TP
    for rank in packs.ranks:
        packs.writers[rank].close()
        config = {"model": "gemma4_31b", "tp_degree": TP, "tp_rank": rank, "hidden": text["hidden_size"],
                  "layers": text["num_hidden_layers"], "vocab": text["vocab_size"], "vocab_first": rank * vocab_rows,
                  "vocab_rows": vocab_rows, "ffn_rows": text["intermediate_size"] // TP, "full_period": 6, "full_phase": 5,
                  "sliding_window": text["sliding_window"]}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": "bf16", "attn_qkv_layout": "fused_windows", "revision": revision, "contract_sha256": contract_sha256}
        final = packs.final(rank)
        assemble(final, packs.payload(rank), NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, config,
                 packs.writers[rank].manifest)
        digest = hashlib.sha256()
        with final.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        Path(str(final) + ".sha256").write_text(f"{digest.hexdigest()}  {final.name}\n")
        Path(str(final) + ".experts").write_bytes(struct.pack("<4I", EXPERTS_MAGIC, EXPERTS_VERSION, 0, 0))
        print(f"rank {rank:02d} {final.stat().st_size} {digest.hexdigest()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ranks", default=",".join(str(rank) for rank in range(TP)))
    args = parser.parse_args()
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    text = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    if text.get("enable_moe_block") or text.get("hidden_size_per_layer_input") or text.get("num_kv_shared_layers"):
        raise PackFailure("this packer handles the dense Gemma 4 text model without per-layer inputs or shared KV")
    expected = ["full_attention" if index % 6 == 5 else "sliding_attention" for index in range(text["num_hidden_layers"])]
    if text["layer_types"] != expected or text["vocab_size"] % TP or text["intermediate_size"] % (TP * 64):
        raise PackFailure("layer pattern, vocab or intermediate size differs from the packer's TP16 layout")
    receipt = args.checkpoint / "DOWNLOAD-RECEIPT.json"
    revision = json.loads(receipt.read_text()).get("revision", "unknown") if receipt.exists() else "unknown"
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint)
    packs = RankPacks(args.output_dir, args.resume, ranks)
    hidden, vocab_rows = text["hidden_size"], text["vocab_size"] // TP
    embed = checkpoint.array(PREFIX + "embed_tokens.weight", (text["vocab_size"], hidden))
    for rank in ranks:
        packs.add(rank, "embed", embed[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
    packs.replicated("head_norm", checkpoint.f32(PREFIX + "norm.weight", (hidden,)), "f32")
    for layer in range(text["num_hidden_layers"]):
        pack_layer(checkpoint, packs, text, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, text, revision, contract_sha256)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"gemma4_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
