#!/usr/bin/env python3
"""Ling 3.0 flash (BailingMoeV3) BF16 checkpoint -> sixteen TP16 named stage packs.

Output per rank: <out>/<prefix>.tp16.rank<NN>.pack (tools/spark_named_pack.py
container) plus <pack>.sha256 and the empty-range <pack>.experts (experts are
plain tensors). The MTP layer is not packed.

Rank r holds, bf16 unless noted:
  embed, head         vocab rows [r*V/16, +V/16) of word_embeddings and lm_head
  head_norm           model.norm as f32
  every layer         input / post-attention norms as f32
  KDA layers          kda_qkv: q|k|v rows of the rank's 2 heads [768, H];
                      kda_f, kda_g [256, H]; kda_beta [2, H]; kda_conv: the
                      q|k|v depthwise taps [768, 4]; kda_a_log f32 [2];
                      kda_dt_bias f32 [256]; kda_norm [128]; kda_out: o_proj
                      columns [H, 256]
  MLA layers          mla_q: q_proj rows of the rank's 2 heads [384, H];
                      mla_kv_a [576, H] and mla_kv_norm f32 [512]
                      (replicated); mla_uk: per head kv_b nope rows
                      transposed [2, 512, 128]; mla_uv: per head kv_b value
                      rows [2, 128, 512]; mla_gate [2, H]; mla_out: dense
                      columns [H, 256]
  dense layers        ffn_gate_up [2*I/16, H]; ffn_down [H, I/16]
  MoE layers          router [E, H] and expert_bias f32 [E] (replicated);
                      experts [r*E/16, +E/16) as moe_w1 / moe_w3 [E/16, I, H]
                      and moe_w2 [E/16, H, I]; the shared expert's rank
                      slice of 48 rows padded with zero rows to 64:
                      shared_gate_up [128, H], shared_down [H, 64]
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
SHARED_ROWS = 64
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2


class PackFailure(RuntimeError):
    pass


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def raw(self, name, shape, dtype):
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"checkpoint has no tensor {name}")
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        base, header = self.headers[shard]
        meta = header[name]
        if meta["dtype"] != dtype or tuple(meta["shape"]) != tuple(shape):
            raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected {dtype} {list(shape)}")
        begin, end = meta["data_offsets"]
        kind = np.uint16 if dtype == "BF16" else np.float32
        return np.memmap(self.directory / shard, dtype=kind, mode="r", offset=base + begin,
                         shape=((end - begin) // np.dtype(kind).itemsize,)).reshape(shape)

    def bf16(self, name, shape):
        return self.raw(name, shape, "BF16")

    def f32(self, name, shape):
        return np.asarray(self.raw(name, shape, "F32"))

    def bf16_as_f32(self, name, shape):
        return (np.asarray(self.bf16(name, shape)).astype(np.uint32) << 16).view(np.float32)


class RankPacks:
    def __init__(self, out_dir, resume, ranks, prefix):
        self.out_dir = Path(out_dir)
        self.ranks = ranks
        self.prefix = prefix
        self.writers = {rank: PayloadWriter(self.payload(rank), resume=resume and self.payload(rank).exists()) for rank in ranks}

    def payload(self, rank):
        return self.out_dir / f"{self.prefix}.tp16.rank{rank:02d}.pack.payload"

    def final(self, rank):
        return self.out_dir / f"{self.prefix}.tp16.rank{rank:02d}.pack"

    def add(self, rank, name, array, kind):
        if rank in self.writers:
            array = np.ascontiguousarray(array)
            self.writers[rank].add(name, array.tobytes(), kind, array.shape)

    def replicated(self, name, array, kind):
        for rank in self.ranks:
            self.add(rank, name, array, kind)


def pad_rows(array, rows):
    padded = np.zeros((rows,) + array.shape[1:], dtype=array.dtype)
    padded[:array.shape[0]] = array
    return padded


def pack_layer(checkpoint, packs, config, layer):
    hidden, heads, head_dim = config["hidden_size"], config["num_attention_heads"], config["head_dim"]
    p = f"model.layers.{layer}."
    a = p + "attention."
    name = f"layers.{layer}."
    packs.replicated(name + "input_norm", checkpoint.bf16_as_f32(p + "input_layernorm.weight", (hidden,)), "f32")
    packs.replicated(name + "post_norm", checkpoint.bf16_as_f32(p + "post_attention_layernorm.weight", (hidden,)), "f32")
    rank_heads = heads // TP
    if (layer + 1) % config["layer_group_size"] == 0:
        nope, rope, value, latent = config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"], config["kv_lora_rank"]
        q = checkpoint.bf16(a + "q_proj.weight", (heads * (nope + rope), hidden))
        kv_b = np.asarray(checkpoint.bf16(a + "kv_b_proj.weight", (heads * (nope + value), latent))).reshape(heads, nope + value, latent)
        gate = checkpoint.bf16(a + "g_proj.weight", (heads, hidden))
        dense = checkpoint.bf16(a + "dense.weight", (hidden, heads * value))
        packs.replicated(name + "mla_kv_a", checkpoint.bf16(a + "kv_a_proj_with_mqa.weight", (latent + rope, hidden)), "bf16")
        packs.replicated(name + "mla_kv_norm", checkpoint.bf16_as_f32(a + "kv_a_layernorm.weight", (latent,)), "f32")
        for rank in packs.ranks:
            owned = slice(rank * rank_heads, (rank + 1) * rank_heads)
            packs.add(rank, name + "mla_q", q[rank * rank_heads * (nope + rope):(rank + 1) * rank_heads * (nope + rope)], "bf16")
            packs.add(rank, name + "mla_uk", kv_b[owned, :nope, :].transpose(0, 2, 1), "bf16")
            packs.add(rank, name + "mla_uv", kv_b[owned, nope:, :], "bf16")
            packs.add(rank, name + "mla_gate", gate[owned], "bf16")
            packs.add(rank, name + "mla_out", dense[:, rank * rank_heads * value:(rank + 1) * rank_heads * value], "bf16")
    else:
        width = heads * head_dim
        projections = {key: checkpoint.bf16(a + f"{key}_proj.weight", (width, hidden)) for key in ("q", "k", "v", "f", "g")}
        convs = {key: np.asarray(checkpoint.bf16(a + f"{key}_conv1d.weight", (width, 1, config["short_conv_kernel_size"])))[:, 0, :]
                 for key in ("q", "k", "v")}
        beta = checkpoint.bf16(a + "b_proj.weight", (heads, hidden))
        a_log = checkpoint.f32(a + "A_log", (heads,))
        dt_bias = checkpoint.f32(a + "dt_bias", (width,))
        out = checkpoint.bf16(a + "o_proj.weight", (hidden, width))
        packs.replicated(name + "kda_norm", checkpoint.bf16(a + "o_norm.weight", (head_dim,)), "bf16")
        for rank in packs.ranks:
            rows = slice(rank * rank_heads * head_dim, (rank + 1) * rank_heads * head_dim)
            packs.add(rank, name + "kda_qkv", np.concatenate([projections[key][rows] for key in ("q", "k", "v")]), "bf16")
            packs.add(rank, name + "kda_f", projections["f"][rows], "bf16")
            packs.add(rank, name + "kda_g", projections["g"][rows], "bf16")
            packs.add(rank, name + "kda_beta", beta[rank * rank_heads:(rank + 1) * rank_heads], "bf16")
            packs.add(rank, name + "kda_conv", np.concatenate([convs[key][rows] for key in ("q", "k", "v")]), "bf16")
            packs.add(rank, name + "kda_a_log", a_log[rank * rank_heads:(rank + 1) * rank_heads], "f32")
            packs.add(rank, name + "kda_dt_bias", dt_bias[rows], "f32")
            packs.add(rank, name + "kda_out", out[:, rows], "bf16")
    m = p + "mlp."
    if layer < config["first_k_dense_replace"]:
        inter = config["intermediate_size"]
        ffn = inter // TP
        gate = checkpoint.bf16(m + "gate_proj.weight", (inter, hidden))
        up = checkpoint.bf16(m + "up_proj.weight", (inter, hidden))
        down = checkpoint.bf16(m + "down_proj.weight", (hidden, inter))
        for rank in packs.ranks:
            window = slice(rank * ffn, (rank + 1) * ffn)
            packs.add(rank, name + "ffn_gate_up", np.concatenate((gate[window], up[window])), "bf16")
            packs.add(rank, name + "ffn_down", down[:, window], "bf16")
        return
    experts, inter, shared = config["num_experts"], config["moe_intermediate_size"], config["moe_shared_expert_intermediate_size"]
    local, shared_rows = experts // TP, shared // TP
    packs.replicated(name + "router", checkpoint.bf16(m + "gate.weight", (experts, hidden)), "bf16")
    packs.replicated(name + "expert_bias", checkpoint.f32(m + "gate.expert_bias", (experts,)), "f32")
    shared_gate = checkpoint.bf16(m + "shared_experts.gate_proj.weight", (shared, hidden))
    shared_up = checkpoint.bf16(m + "shared_experts.up_proj.weight", (shared, hidden))
    shared_down = checkpoint.bf16(m + "shared_experts.down_proj.weight", (hidden, shared))
    for rank in packs.ranks:
        owned = range(rank * local, (rank + 1) * local)
        for key, target, shape in (("gate", "moe_w1", (inter, hidden)), ("up", "moe_w3", (inter, hidden)), ("down", "moe_w2", (hidden, inter))):
            packs.add(rank, name + target, np.stack([checkpoint.bf16(m + f"experts.{expert}.{key}_proj.weight", shape) for expert in owned]), "bf16")
        window = slice(rank * shared_rows, (rank + 1) * shared_rows)
        packs.add(rank, name + "shared_gate_up", np.concatenate((pad_rows(shared_gate[window], SHARED_ROWS), pad_rows(shared_up[window], SHARED_ROWS))), "bf16")
        packs.add(rank, name + "shared_down", pad_rows(np.ascontiguousarray(shared_down[:, window].T), SHARED_ROWS).T, "bf16")


def finish(packs, config, model, revision, contract_sha256):
    vocab_rows = config["vocab_size"] // TP
    local = config["num_experts"] // TP
    for rank in packs.ranks:
        packs.writers[rank].close()
        layout = {"model": model, "tp_degree": TP, "tp_rank": rank, "hidden": config["hidden_size"], "layers": config["num_hidden_layers"],
                  "vocab": config["vocab_size"], "vocab_first": rank * vocab_rows, "vocab_rows": vocab_rows,
                  "ffn_rows": config["intermediate_size"] // TP, "experts": config["num_experts"], "local_experts": local,
                  "expert_first": rank * local, "top_k": config["num_experts_per_tok"], "shared_rows": SHARED_ROWS}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": "bf16", "expert_codec": "bf16", "revision": revision, "contract_sha256": contract_sha256}
        final = packs.final(rank)
        assemble(final, packs.payload(rank), NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, layout,
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
    parser.add_argument("--model", choices=("ling3_flash", "ling3_fin"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ranks", default=",".join(str(rank) for rank in range(TP)))
    args = parser.parse_args()
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    config = json.loads((args.checkpoint / "config.json").read_text())
    if config.get("q_lora_rank") is not None or not config.get("no_kda_lora") or not config.get("rope_interleave") or \
            config.get("gated_attention_proj_granularity_type") != "head_wise" or config.get("num_shared_experts") != 1:
        raise PackFailure("this packer handles the Ling 3.0 flash layout: no q LoRA, no KDA LoRA, interleaved rope, head-wise gate, one shared expert")
    if config["num_attention_heads"] % TP or config["num_experts"] % TP or config["vocab_size"] % TP or \
            config["intermediate_size"] % (TP * 64) or config["moe_shared_expert_intermediate_size"] // TP > SHARED_ROWS:
        raise PackFailure("heads, experts, vocab or intermediate sizes do not fit the TP16 layout")
    receipt = args.checkpoint / "DOWNLOAD-RECEIPT.json"
    revision = json.loads(receipt.read_text()).get("revision", "unknown") if receipt.exists() else "unknown"
    contract = json.loads(args.contract.read_text())
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint)
    packs = RankPacks(args.output_dir, args.resume, ranks, args.model)
    hidden, vocab_rows = config["hidden_size"], config["vocab_size"] // TP
    embed = checkpoint.bf16("model.word_embeddings.weight", (config["vocab_size"], hidden))
    head = checkpoint.bf16("lm_head.weight", (config["vocab_size"], hidden))
    for rank in ranks:
        packs.add(rank, "embed", embed[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
        packs.add(rank, "head", head[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
    packs.replicated("head_norm", checkpoint.bf16_as_f32("model.norm.weight", (hidden,)), "f32")
    for layer in range(config["num_hidden_layers"]):
        pack_layer(checkpoint, packs, config, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, config, args.model, revision if revision != "unknown" else contract.get("source_revision", "unknown"), contract_sha256)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"ling3_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
