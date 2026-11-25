#!/usr/bin/env python3
"""Hy4 preview (official MXFP8 checkpoint) -> TP named stage packs (sixteen by default).

--tp-degree sets the rank count (16 for the Sparks; 1, 2, 4 or 8 for Metal
ranks); the splits below are of 1/degree each, shown for 16. Output per rank:
<out>/hy4.tp<degree>.rank<NN>.pack (tools/spark_named_pack.py
container) plus <pack>.sha256 and the empty-range <pack>.experts (experts are
plain tensors). MXFP8 projections keep the official bytes: an "fp8" tensor
(E4M3, [rows, K]) and its "<name>_scale" "u8" tensor (E8M0 per 32 columns,
[rows, K/32]). The MTP layer is not packed.

Rank r holds:
  embed, head         vocab rows [r*V/16, +V/16), bf16
  head_norm           model.norm as f32
  head_hc_*           the final hyper-connection collapse (f32)
  every layer         attn/mlp hyper-connection fn, base, scale (f32);
                      input/post norms (bf16); q_a, kv_a (replicated); q_norm,
                      kv_norm (bf16); q_b rows of the rank's 4 heads; uk: per
                      head kv_b nope rows dequantised and transposed to bf16
                      [4, 512, 192]; uv: per head kv_b value rows [4, 256, 512];
                      gate: linear_gate rows [1024, H] bf16; out: o_proj columns
                      [H, 1024]; sinks f32 [64]
  indexer layers      index_q (wq_b), index_k (wk) replicated; index_k_norm and
                      index_k_bias f32; index_weights bf16 [32, H]
  dense layer         ffn_gate_up rows [2*I/16, H]; ffn_down columns [H, I/16]
  MoE layers          router bf16 [E, H], router_bias f32 [E]; experts
                      [r*E/16, +E/16): moe_w1 (gate rows), moe_w3 (up rows),
                      moe_w2 (down); shared expert rows [2*S/16, H] and
                      columns [H, S/16]
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

TP = 16
GROUP = 32
FP8_BLOCK = 128
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2
DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8}


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def expand_block128_rows(scale, rows):
    row_blocks = np.arange(rows, dtype=np.int64) // FP8_BLOCK
    return np.ascontiguousarray(scale[row_blocks], dtype=np.float32)


class PackFailure(RuntimeError):
    pass


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def meta(self, name):
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"checkpoint has no tensor {name}")
        if shard not in self.headers:
            with open(self.directory / shard, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                self.headers[shard] = (8 + length, json.loads(handle.read(length)))
        base, header = self.headers[shard]
        return shard, base, header[name]

    def raw(self, name, dtype, shape):
        shard, base, meta = self.meta(name)
        if meta["dtype"] != dtype or tuple(meta["shape"]) != tuple(shape):
            raise PackFailure(f"{name}: {meta['dtype']} {meta['shape']}, expected {dtype} {list(shape)}")
        begin, end = meta["data_offsets"]
        kind = DTYPES[dtype]
        return np.memmap(self.directory / shard, dtype=kind, mode="r", offset=base + begin,
                         shape=((end - begin) // np.dtype(kind).itemsize,)).reshape(shape)

    def bf16(self, name, shape):
        return self.raw(name, "BF16", shape)

    def f32(self, name, shape):
        return np.asarray(self.raw(name, "F32", shape))

    def bf16_as_f32(self, name, shape):
        return (np.asarray(self.bf16(name, shape)).astype(np.uint32) << 16).view(np.float32)

    def scaled(self, name, shape, prefix=None):
        weight = self.raw(name + (".weight" if prefix is None else ""), "F8_E4M3", shape)
        scale_name = name + (".weight_scale" if prefix is None else "_scale")
        scale = self.raw(scale_name, "U8", shape[:-1] + (shape[-1] // GROUP,))
        return ("mxfp8", weight, scale)

    def linear(self, name, shape):
        weight_name = name + ".weight"
        _, _, meta = self.meta(weight_name)
        if meta["dtype"] == "BF16":
            return ("bf16", self.bf16(weight_name, shape), None)
        if meta["dtype"] != "F8_E4M3":
            raise PackFailure(f"{weight_name}: unsupported dtype {meta['dtype']}")
        scale_name = name + ".weight_scale"
        if scale_name not in self.weight_map and (name + ".weight_scale_inv") in self.weight_map:
            scale_name = name + ".weight_scale_inv"
        _, _, smeta = self.meta(scale_name)
        if smeta["dtype"] == "U8":
            return self.scaled(name, shape)
        if smeta["dtype"] != "F32":
            raise PackFailure(f"{scale_name}: unsupported scale dtype {smeta['dtype']}")
        weight = self.raw(weight_name, "F8_E4M3", shape)
        rows, cols = shape
        if cols % FP8_BLOCK != 0:
            raise PackFailure(f"{weight_name}: cols {cols} not divisible by {FP8_BLOCK}")
        scale = np.asarray(self.raw(scale_name, "F32", (ceil_div(rows, FP8_BLOCK), cols // FP8_BLOCK)))
        return ("fp8_e4m3", weight, expand_block128_rows(scale, rows))


E4M3 = np.array([(-1.0 if code & 0x80 else 1.0) * (
    (2.0 ** (((code >> 3) & 15) - 7)) * (1.0 + (code & 7) / 8.0) if (code >> 3) & 15 else (2.0 ** -6) * ((code & 7) / 8.0))
    for code in range(256)], dtype=np.float32)
E4M3[0x7F] = np.nan
E4M3[0xFF] = np.nan


def dequantise_linear(item):
    kind, weight, scale = item
    if kind == "bf16":
        return (np.asarray(weight).astype(np.uint32) << 16).view(np.float32)
    values = E4M3[np.asarray(weight)]
    if kind == "mxfp8":
        factors = np.exp2(np.asarray(scale).astype(np.float32) - 127.0)
        return values * np.repeat(factors, GROUP, axis=-1)
    return values * np.repeat(np.asarray(scale, dtype=np.float32), FP8_BLOCK, axis=-1)


def to_bf16(values):
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


class RankPacks:
    def __init__(self, out_dir, resume, ranks, identity):
        self.out_dir = Path(out_dir)
        self.ranks = ranks
        self.writers = {rank: PayloadWriter(self.payload(rank), identity, resume=resume and self.payload(rank).exists()) for rank in ranks}

    def payload(self, rank):
        return self.out_dir / f"hy4.tp{TP}.rank{rank:02d}.pack.payload"

    def final(self, rank):
        return self.out_dir / f"hy4.tp{TP}.rank{rank:02d}.pack"

    def add(self, rank, name, array, kind):
        if rank in self.writers:
            array = np.ascontiguousarray(array)
            self.writers[rank].add(name, array.tobytes(), kind, array.shape)

    def add_parts(self, rank, name, parts, kind, shape):
        if rank in self.writers:
            self.writers[rank].add_parts(name, [np.ascontiguousarray(part) for part in parts], kind, shape)

    def linear(self, rank, name, item):
        kind, weight, scale = item
        if kind == "bf16":
            self.add(rank, name, weight, "bf16")
        elif kind == "mxfp8":
            self.add(rank, name, weight, "fp8")
            self.add(rank, name + "_scale", scale, "u8")
        else:
            self.add(rank, name, weight, "fp8_e4m3")
            self.add(rank, name + ".scale", scale, "f32")

    def replicated(self, name, array, kind):
        for rank in self.ranks:
            self.add(rank, name, array, kind)

    def replicated_linear(self, name, item):
        for rank in self.ranks:
            self.linear(rank, name, item)


def rows_of(item, window):
    kind, weight, scale = item
    return (kind, weight[window], None if scale is None else scale[window])


def concat_rows(first, second):
    if first[0] != second[0]:
        raise PackFailure(f"cannot concatenate mismatched codecs {first[0]} and {second[0]}")
    kind = first[0]
    weight = np.concatenate((first[1], second[1]))
    scale = None if first[2] is None else np.concatenate((first[2], second[2]))
    return (kind, weight, scale)


def columns_of(item, first, count):
    kind, weight, scale = item
    if kind == "bf16":
        return (kind, weight[:, first:first + count], None)
    block = GROUP if kind == "mxfp8" else FP8_BLOCK
    if first % block != 0 or count % block != 0:
        raise PackFailure(f"column slice [{first}, +{count}) not aligned to {block}")
    return (kind, weight[:, first:first + count], scale[:, first // block:(first + count) // block])


def pack_layer(checkpoint, packs, config, layer):
    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    nope, rope, value, latent, q_lora = (config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"], config["kv_lora_rank"],
                                         config["q_lora_rank"])
    rank_heads = heads // TP
    p = f"model.layers.{layer}."
    a = p + "self_attn."
    name = f"layers.{layer}."
    for target, source in (("attn_hc", "hc_attn_layer"), ("mlp_hc", "hc_mlp_layer")):
        packs.replicated(name + target + "_fn", checkpoint.f32(p + source + ".hc_pre.hc_fn", (8, 4 * hidden)), "f32")
        packs.replicated(name + target + "_base", checkpoint.f32(p + source + ".hc_pre.hc_base", (8,)), "f32")
        packs.replicated(name + target + "_scale", checkpoint.f32(p + source + ".hc_pre.hc_scale", (2,)), "f32")
    packs.replicated(name + "input_norm", checkpoint.bf16(p + "input_layernorm.weight", (hidden,)), "bf16")
    packs.replicated(name + "post_norm", checkpoint.bf16(p + "post_attention_layernorm.weight", (hidden,)), "bf16")
    packs.replicated_linear(name + "q_a", checkpoint.linear(a + "q_a_proj", (q_lora, hidden)))
    packs.replicated(name + "q_norm", checkpoint.bf16(a + "q_a_layernorm.weight", (q_lora,)), "bf16")
    packs.replicated_linear(name + "kv_a", checkpoint.linear(a + "kv_a_proj_with_mqa", (latent + rope, hidden)))
    packs.replicated(name + "kv_norm", checkpoint.bf16(a + "kv_a_layernorm.weight", (latent,)), "bf16")
    packs.replicated(name + "sinks", checkpoint.f32(a + "learnable_sink_param", (heads,)), "f32")
    q_b = checkpoint.linear(a + "q_b_proj", (heads * (nope + rope), q_lora))
    kv_b_item = checkpoint.linear(a + "kv_b_proj", (heads * (nope + value), latent))
    kv_b = (np.asarray(kv_b_item[1]) if kv_b_item[0] == "bf16" else to_bf16(dequantise_linear(kv_b_item))).reshape(heads, nope + value, latent)
    gate = checkpoint.linear(a + "linear_gate", (heads * value, hidden))
    out = checkpoint.linear(a + "o_proj", (hidden, heads * value))
    for rank in packs.ranks:
        owned = slice(rank * rank_heads, (rank + 1) * rank_heads)
        packs.linear(rank, name + "q_b", rows_of(q_b, slice(rank * rank_heads * (nope + rope), (rank + 1) * rank_heads * (nope + rope))))
        packs.add(rank, name + "uk", kv_b[owned, :nope, :].transpose(0, 2, 1), "bf16")
        packs.add(rank, name + "uv", kv_b[owned, nope:, :], "bf16")
        packs.linear(rank, name + "gate", rows_of(gate, slice(rank * rank_heads * value, (rank + 1) * rank_heads * value)))
        packs.linear(rank, name + "out", columns_of(out, rank * rank_heads * value, rank_heads * value))
    if config["indexer_types"][layer] == "full":
        index_heads, index_dim = config["index_n_heads"], config["index_head_dim"]
        i = a + "indexer."
        packs.replicated_linear(name + "index_q", checkpoint.linear(i + "wq_b", (index_heads * index_dim, q_lora)))
        packs.replicated_linear(name + "index_k", checkpoint.linear(i + "wk", (index_dim, hidden)))
        packs.replicated(name + "index_k_norm", checkpoint.bf16_as_f32(i + "k_norm.weight", (index_dim,)), "f32")
        packs.replicated(name + "index_k_bias", checkpoint.bf16_as_f32(i + "k_norm.bias", (index_dim,)), "f32")
        packs.replicated(name + "index_weights", checkpoint.bf16(i + "weights_proj.weight", (index_heads, hidden)), "bf16")
    m = p + "mlp."
    is_dense = config["mlp_layer_types"][layer] == "dense" if "mlp_layer_types" in config else layer < config.get("first_k_dense_replace", 1)
    if is_dense:
        inter = config["intermediate_size"]
        ffn = inter // TP
        gate_proj = checkpoint.linear(m + "gate_proj", (inter, hidden))
        up_proj = checkpoint.linear(m + "up_proj", (inter, hidden))
        down = checkpoint.linear(m + "down_proj", (hidden, inter))
        for rank in packs.ranks:
            window = slice(rank * ffn, (rank + 1) * ffn)
            packs.linear(rank, name + "ffn_gate_up", concat_rows(rows_of(gate_proj, window), rows_of(up_proj, window)))
            packs.linear(rank, name + "ffn_down", columns_of(down, rank * ffn, ffn))
        return
    experts, inter, shared = config["n_routed_experts"], config["moe_intermediate_size"], config["moe_intermediate_size"] * config["n_shared_experts"]
    local, shared_rows = experts // TP, shared // TP
    packs.replicated(name + "router", checkpoint.bf16(m + "gate.weight", (experts, hidden)), "bf16")
    packs.replicated(name + "router_bias", checkpoint.f32(m + "gate.e_score_correction_bias", (experts,)), "f32")
    shared_gate = checkpoint.linear(m + "shared_experts.gate_proj", (shared, hidden))
    shared_up = checkpoint.linear(m + "shared_experts.up_proj", (shared, hidden))
    shared_down = checkpoint.linear(m + "shared_experts.down_proj", (hidden, shared))
    fused_experts = (m + "experts.gate_up_proj") in checkpoint.weight_map
    if fused_experts:
        gate_up = checkpoint.scaled(m + "experts.gate_up_proj", (experts, 2 * inter, hidden), prefix=True)
        expert_down = checkpoint.scaled(m + "experts.down_proj", (experts, hidden, inter), prefix=True)
        for rank in packs.ranks:
            owned = slice(rank * local, (rank + 1) * local)
            packs.linear(rank, name + "moe_w1", ("mxfp8", gate_up[1][owned, :inter], gate_up[2][owned, :inter]))
            packs.linear(rank, name + "moe_w3", ("mxfp8", gate_up[1][owned, inter:], gate_up[2][owned, inter:]))
            packs.linear(rank, name + "moe_w2", ("mxfp8", expert_down[1][owned], expert_down[2][owned]))
    else:
        for rank in packs.ranks:
            owned = range(rank * local, (rank + 1) * local)
            for target, source, rows_out, cols_in in (("moe_w1", "gate_proj", inter, hidden), ("moe_w3", "up_proj", inter, hidden),
                                                      ("moe_w2", "down_proj", hidden, inter)):
                weights, scales, kind = [], [], None
                for expert in owned:
                    ek, ew, es = checkpoint.linear(f"{m}experts.{expert}.{source}", (rows_out, cols_in))
                    if kind is None:
                        kind = ek
                    elif kind != ek:
                        raise PackFailure(f"layer {layer} expert {expert} {source}: codec {ek} != {kind}")
                    weights.append(ew)
                    if es is not None:
                        scales.append(es)
                if kind == "bf16":
                    packs.add_parts(rank, name + target, weights, "bf16", (local, rows_out, cols_in))
                elif kind == "fp8_e4m3":
                    packs.add_parts(rank, name + target, weights, "fp8_e4m3", (local, rows_out, cols_in))
                    packs.add_parts(rank, name + target + ".scale", scales, "f32", (local, rows_out, cols_in // FP8_BLOCK))
                else:
                    packs.add_parts(rank, name + target, weights, "fp8", (local, rows_out, cols_in))
                    packs.add_parts(rank, name + target + "_scale", scales, "u8", (local, rows_out, cols_in // GROUP))
    for rank in packs.ranks:
        window = slice(rank * shared_rows, (rank + 1) * shared_rows)
        packs.linear(rank, name + "shared_gate_up", concat_rows(rows_of(shared_gate, window), rows_of(shared_up, window)))
        packs.linear(rank, name + "shared_down", columns_of(shared_down, rank * shared_rows, shared_rows))


def resolve_text_config(raw):
    cfg = dict(raw)
    if isinstance(raw.get("text_config"), dict):
        cfg.update(raw["text_config"])
    return cfg


def detect_codecs(raw):
    qcfg = raw.get("quantization_config") or {}
    if qcfg.get("quant_method") == "modelopt" and (qcfg.get("quantization") or {}).get("quant_algo") == "MXFP8":
        return "hy4_preview", "mxfp8", "mxfp8"
    if qcfg.get("quant_method") == "compressed-tensors":
        groups = qcfg.get("config_groups") or {}
        for group in groups.values():
            w = group.get("weights") or {}
            if w.get("num_bits") == 8 and w.get("type") == "float" and list(w.get("block_structure") or []) == [FP8_BLOCK, FP8_BLOCK]:
                return "hy4", "mixed_bf16_fp8", "fp8_block128"
    if not qcfg:
        return "hy4", "bf16", "bf16"
    raise PackFailure(f"unsupported Hy4 quantization_config: {qcfg.get('quant_method')}")


def finish(packs, config, model_name, linear_codec, expert_codec, revision, contract_sha256):
    vocab_rows = config["vocab_size"] // TP
    local = config["n_routed_experts"] // TP
    for rank in packs.ranks:
        packs.writers[rank].close()
        layout = {"model": model_name, "tp_degree": TP, "tp_rank": rank, "hidden": config["hidden_size"], "layers": config["num_hidden_layers"],
                  "vocab": config["vocab_size"], "vocab_first": rank * vocab_rows, "vocab_rows": vocab_rows,
                  "ffn_rows": config["intermediate_size"] // TP, "shared_rows": config["moe_intermediate_size"] * config["n_shared_experts"] // TP,
                  "experts": config["n_routed_experts"], "local_experts": local, "expert_first": rank * local,
                  "top_k": config["num_experts_per_tok"]}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": linear_codec, "expert_codec": expert_codec, "revision": revision, "contract_sha256": contract_sha256}
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tp-degree", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--ranks")
    args = parser.parse_args()
    global TP
    TP = args.tp_degree
    args.ranks = args.ranks or ",".join(str(rank) for rank in range(TP))
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    raw_config = json.loads((args.checkpoint / "config.json").read_text())
    model_name, linear_codec, expert_codec = detect_codecs(raw_config)
    config = resolve_text_config(raw_config)
    if config.get("n_group") != 1 or config.get("hc_mult") != 4 or config.get("gating_type") != "elementwise" or \
            config["num_attention_heads"] % TP or config["n_routed_experts"] % TP or config["vocab_size"] % TP or \
            config["intermediate_size"] % (TP * 64) or config["moe_intermediate_size"] * config["n_shared_experts"] % (TP * 64):
        raise PackFailure(f"this packer handles the Hy4 layout on TP{TP}")
    receipt = args.checkpoint / "DOWNLOAD-RECEIPT.json"
    revision = json.loads(receipt.read_text()).get("revision", "unknown") if receipt.exists() else "unknown"
    contract = json.loads(args.contract.read_text())
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint)
    packs = RankPacks(args.output_dir, args.resume, ranks, source_identity(args.checkpoint, "hy4_tp16_pack" if TP == 16 else f"hy4_tp{TP}_pack", files=(args.contract,)))
    hidden, vocab_rows = config["hidden_size"], config["vocab_size"] // TP
    embed = checkpoint.bf16("model.embed_tokens.weight", (config["vocab_size"], hidden))
    head = checkpoint.bf16("lm_head.weight", (config["vocab_size"], hidden))
    for rank in ranks:
        packs.add(rank, "embed", embed[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
        packs.add(rank, "head", head[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
    packs.replicated("head_norm", checkpoint.bf16_as_f32("model.norm.weight", (hidden,)), "f32")
    packs.replicated("head_hc_fn", checkpoint.f32("model.hc_head.hc_head_fn", (4, 4 * hidden)), "f32")
    packs.replicated("head_hc_base", checkpoint.f32("model.hc_head.hc_head_base", (4,)), "f32")
    packs.replicated("head_hc_scale", checkpoint.f32("model.hc_head.hc_head_scale", (1,)), "f32")
    for layer in range(config["num_hidden_layers"]):
        pack_layer(checkpoint, packs, config, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, config, model_name, linear_codec, expert_codec, revision if revision != "unknown" else contract.get("source_revision", "unknown"), contract_sha256)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"hy4_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
