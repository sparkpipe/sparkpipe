#!/usr/bin/env python3
"""MiMo V2.6 Flash / Pro (official FP8 + MXFP4 checkpoint) -> sixteen TP16 named stage packs.

Output per rank: <out>/mimo26_<model>.tp16.rank<NN>.pack (tools/spark_named_pack.py
container), <pack>.sha256 and the empty-range <pack>.experts v2 manifest (experts
are plain tensors). Every weight keeps the official bytes; nothing is requantised.

The checkpoint stores each fused qkv_proj TP-interleaved: T segments (T is the
index metadata tp_size, 4 for Flash, 8 for Pro), segment s = [q heads of s | k
heads of s | v heads of s], with the 128x128 scale_inv grid blocked per segment
and each segment padded to whole 128-row blocks. The packer proves that layout
on every layer (every 128x128 block of the segment map holds the e4m3 maximum
code 0x7e, which a per-block absmax/448 quantiser always produces) and refuses
anything else.

Rank r holds:
  embed, head           vocab rows [r*V/16, +V/16), bf16
  head_norm             model.norm as f32
  layers.L.input_norm, post_attn_norm      bf16 [H]
  layers.L.attn_qkv     fp8_e4m3 rows [r*F/16, +F/16) of the de-interleaved
                        [Q | K | V] (F = heads*192 + kv*(192+128)); attn_qkv.scale
                        f32 [F/16, H/128]: each row carries its source block's
                        scale_inv
  layers.L.attn_out     o_proj columns of the rank's heads [H, heads/16*128] bf16
  layers.L.sinks        SWA layers only: attention_sink_bias as f32 [heads]
  layer 0               ffn_gate_up fp8 [gate rows r*I/16 | up rows], ffn_down
                        fp8 columns [H, I/16], both with per-row f32 .scale
  MoE layers            router bf16 [E, H]; router_bias f32 [E]; experts
                        [r*E/16, +E/16): moe_w1 (gate) / moe_w3 (up) mxfp4_e2m1
                        [E/16, I, H/2] + .scale e8m0 [E/16, I, H/32]; moe_w2
                        [E/16, H, I/2] + .scale [E/16, H, I/32]
The MTP layers, the DFlash drafter and the vision / audio towers are not packed.
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
BLOCK = 128
MX_GROUP = 32
E4M3_MAX_CODE = 0x7E
E8M0_NAN = 0xFF
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2
HEAD_DIM = 192
VALUE_DIM = 128
DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8}
MODELS = {"flash": dict(hidden=4096, layers=48, heads=64), "pro": dict(hidden=6144, layers=70, heads=128)}


class PackFailure(RuntimeError):
    pass


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        index = json.loads((self.directory / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.metadata = index.get("metadata") or {}
        self.headers = {}

    def raw(self, name, dtype, shape):
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
        kind = DTYPES[dtype]
        return np.memmap(self.directory / shard, dtype=kind, mode="r", offset=base + begin,
                         shape=((end - begin) // np.dtype(kind).itemsize,)).reshape(shape)

    def bf16_as_f32(self, name, shape):
        return (np.asarray(self.raw(name, "BF16", shape)).astype(np.uint32) << 16).view(np.float32)


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


class Geometry:
    def __init__(self, config, model, interleave):
        self.model = model
        self.hidden = config["hidden_size"]
        self.layers = config["num_hidden_layers"]
        self.heads = config["num_attention_heads"]
        self.vocab = config["vocab_size"]
        self.pattern = list(config["hybrid_layer_pattern"])
        self.full_kv = config["num_key_value_heads"]
        self.sliding_kv = config["swa_num_key_value_heads"]
        self.dense_inter = config["intermediate_size"]
        self.moe_inter = config["moe_intermediate_size"]
        self.experts = config["n_routed_experts"]
        self.top_k = config["num_experts_per_tok"]
        self.window = config["sliding_window"]
        self.interleave = interleave
        self.local = self.experts // TP
        self.vocab_rows = self.vocab // TP
        self.dense_rows = self.dense_inter // TP

    def sliding(self, layer):
        return self.pattern[layer] == 1

    def kv_heads(self, layer):
        return self.sliding_kv if self.sliding(layer) else self.full_kv

    def fused(self, layer):
        return self.heads * HEAD_DIM + self.kv_heads(layer) * (HEAD_DIM + VALUE_DIM)

    def segment(self, layer):
        heads, kv = self.heads // self.interleave, self.kv_heads(layer) // self.interleave
        return heads * HEAD_DIM, kv * HEAD_DIM, kv * VALUE_DIM


def check_config(config, model):
    expected = MODELS[model]
    problems = []
    def need(condition, text):
        if not condition:
            problems.append(text)
    quant = config.get("quantization_config") or {}
    need(config.get("model_type") == "mimo_v2" and config.get("architectures") == ["MiMoV2ForCausalLM"], "model_type mimo_v2")
    need(config.get("attention_projection_layout") == "fused_qkv", "fused_qkv projection layout")
    need(config["hidden_size"] == expected["hidden"] and config["num_hidden_layers"] == expected["layers"] and
         config["num_attention_heads"] == expected["heads"], f"{model} hidden/layers/heads")
    need(config["swa_num_attention_heads"] == config["num_attention_heads"], "SWA and full layers share the query head count")
    need(config["head_dim"] == HEAD_DIM and config["swa_head_dim"] == HEAD_DIM and config["v_head_dim"] == VALUE_DIM and
         config["swa_v_head_dim"] == VALUE_DIM, "head dims 192/128")
    need(int(HEAD_DIM * config["partial_rotary_factor"]) == 64, "64 rotary dims")
    need(config["n_group"] == 1 and config["topk_group"] == 1 and config["scoring_func"] == "sigmoid" and
         config["topk_method"] == "noaux_tc" and config["norm_topk_prob"] is True and config.get("routed_scaling_factor") is None and
         config.get("n_shared_experts") is None, "sigmoid noaux_tc routing, no groups, no shared experts")
    need(list(config["moe_layer_freq"]) == [0] + [1] * (config["num_hidden_layers"] - 1), "layer 0 dense, every other layer MoE")
    need(len(config["hybrid_layer_pattern"]) == config["num_hidden_layers"] and config["hybrid_layer_pattern"][0] == 0 and
         set(config["hybrid_layer_pattern"]) <= {0, 1}, "hybrid pattern, layer 0 full")
    need(config["sliding_window"] == 128 and config["add_swa_attention_sink_bias"] is True and
         config["add_full_attention_sink_bias"] is False, "window 128, sinks on SWA layers only")
    need(config["tie_word_embeddings"] is False and config.get("attention_bias") is False, "untied head, no attention bias")
    need(quant.get("quant_method") == "fp8" and quant.get("fmt") == "e4m3" and list(quant.get("weight_block_size", [])) == [BLOCK, BLOCK] and
         quant.get("store_dtype") == "mxfp4" and quant.get("mxfp4_block_size") == MX_GROUP, "fp8 e4m3 128x128 spine, mxfp4 g32 experts")
    need(config["vocab_size"] % TP == 0 and config["num_attention_heads"] % TP == 0 and config["n_routed_experts"] % TP == 0 and
         config["intermediate_size"] % (TP * BLOCK) == 0, "vocab, heads, experts and the dense intermediate split over 16 ranks")
    need(config["num_key_value_heads"] >= 1 and config["swa_num_key_value_heads"] >= 1, "kv heads")
    ignored = set(quant.get("ignored_layers", []))
    need(all(f"model.layers.{layer}.self_attn.o_proj" in ignored for layer in range(config["num_hidden_layers"])),
         "every o_proj is unquantised")
    if problems:
        raise PackFailure("config is not the MiMo V2.6 layout this packer handles: " + "; ".join(problems))


def qkv_order(geometry, layer):
    q, k, v = geometry.segment(layer)
    segment = q + k + v
    blocks = ceil_div(segment, BLOCK)
    order = []
    for part_offset, part_rows in ((0, q), (q, k), (q + k, v)):
        for s in range(geometry.interleave):
            order.append(np.arange(s * segment + part_offset, s * segment + part_offset + part_rows, dtype=np.int64))
    order = np.concatenate(order)
    source = np.arange(geometry.interleave * segment, dtype=np.int64)
    block_of_row = (source // segment) * blocks + (source % segment) // BLOCK
    return order, block_of_row, segment, blocks


def verify_interleave(weight, scale, geometry, layer, segment, blocks):
    columns = geometry.hidden // BLOCK
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise PackFailure(f"layer {layer}: qkv scale_inv holds non-finite or non-positive values")
    for s in range(geometry.interleave):
        codes = np.bitwise_and(np.asarray(weight[s * segment:(s + 1) * segment]), 0x7F)
        padded = np.zeros((blocks * BLOCK, geometry.hidden), dtype=np.uint8)
        padded[:segment] = codes
        peak = padded.reshape(blocks, BLOCK, columns, BLOCK).max(axis=(1, 3))
        if not np.all(peak == E4M3_MAX_CODE):
            bad = int(np.count_nonzero(peak != E4M3_MAX_CODE))
            raise PackFailure(f"layer {layer}: segment {s} has {bad} of {peak.size} 128x128 blocks without the e4m3 maximum code; "
                              f"the qkv is not {geometry.interleave}-way interleaved as the index metadata says")


class RankPacks:
    def __init__(self, out_dir, resume, ranks, prefix, identity):
        self.out_dir = Path(out_dir)
        self.ranks = ranks
        self.prefix = prefix
        self.writers = {rank: PayloadWriter(self.payload(rank), identity, resume=resume and self.payload(rank).exists()) for rank in ranks}

    def payload(self, rank):
        return self.out_dir / f"{self.prefix}.tp16.rank{rank:02d}.pack.payload"

    def final(self, rank):
        return self.out_dir / f"{self.prefix}.tp16.rank{rank:02d}.pack"

    def add(self, rank, name, array, kind):
        if rank in self.writers:
            array = np.ascontiguousarray(array)
            self.writers[rank].add(name, array.tobytes(), kind, array.shape)

    def add_parts(self, rank, name, parts, kind, shape):
        if rank in self.writers:
            self.writers[rank].add_parts(name, [np.ascontiguousarray(part) for part in parts], kind, shape)

    def replicated(self, name, array, kind):
        for rank in self.ranks:
            self.add(rank, name, array, kind)


def check_mxfp4_scales(name, scale):
    if np.any(np.asarray(scale) == E8M0_NAN):
        raise PackFailure(f"{name}: e8m0 scale holds the NaN code")


def pack_layer(checkpoint, packs, geometry, layer):
    hidden, heads = geometry.hidden, geometry.heads
    p = f"model.layers.{layer}."
    a = p + "self_attn."
    name = f"layers.{layer}."
    packs.replicated(name + "input_norm", checkpoint.raw(p + "input_layernorm.weight", "BF16", (hidden,)), "bf16")
    packs.replicated(name + "post_attn_norm", checkpoint.raw(p + "post_attention_layernorm.weight", "BF16", (hidden,)), "bf16")
    fused = geometry.fused(layer)
    order, block_of_row, segment, blocks = qkv_order(geometry, layer)
    if order.size != fused or segment * geometry.interleave != fused:
        raise PackFailure(f"layer {layer}: fused rows {fused} do not tile {geometry.interleave} segments of {segment}")
    qkv = checkpoint.raw(a + "qkv_proj.weight", "F8_E4M3", (fused, hidden))
    qkv_scale = np.asarray(checkpoint.raw(a + "qkv_proj.weight_scale_inv", "F32", (geometry.interleave * blocks, hidden // BLOCK)))
    verify_interleave(qkv, qkv_scale, geometry, layer, segment, blocks)
    rows = fused // TP
    for rank in packs.ranks:
        source = order[rank * rows:(rank + 1) * rows]
        packs.add(rank, name + "attn_qkv", np.asarray(qkv[source]), "fp8_e4m3")
        packs.add(rank, name + "attn_qkv.scale", qkv_scale[block_of_row[source]], "f32")
    width = heads // TP * VALUE_DIM
    out = checkpoint.raw(a + "o_proj.weight", "BF16", (hidden, heads * VALUE_DIM))
    for rank in packs.ranks:
        packs.add(rank, name + "attn_out", out[:, rank * width:(rank + 1) * width], "bf16")
    sink_name = a + "attention_sink_bias"
    if geometry.sliding(layer):
        packs.replicated(name + "sinks", checkpoint.bf16_as_f32(sink_name, (heads,)), "f32")
    elif sink_name in checkpoint.weight_map:
        raise PackFailure(f"layer {layer} is a full-attention layer but the checkpoint has {sink_name}")
    m = p + "mlp."
    if layer == 0:
        inter, ffn = geometry.dense_inter, geometry.dense_rows
        grid = (inter // BLOCK, hidden // BLOCK)
        gate = checkpoint.raw(m + "gate_proj.weight", "F8_E4M3", (inter, hidden))
        gate_scale = np.asarray(checkpoint.raw(m + "gate_proj.weight_scale_inv", "F32", grid))
        up = checkpoint.raw(m + "up_proj.weight", "F8_E4M3", (inter, hidden))
        up_scale = np.asarray(checkpoint.raw(m + "up_proj.weight_scale_inv", "F32", grid))
        down = checkpoint.raw(m + "down_proj.weight", "F8_E4M3", (hidden, inter))
        down_scale = np.asarray(checkpoint.raw(m + "down_proj.weight_scale_inv", "F32", (hidden // BLOCK, inter // BLOCK)))
        for rank in packs.ranks:
            window = slice(rank * ffn, (rank + 1) * ffn)
            blocks_window = slice(rank * ffn // BLOCK, (rank + 1) * ffn // BLOCK)
            packs.add(rank, name + "ffn_gate_up", np.concatenate((np.asarray(gate[window]), np.asarray(up[window]))), "fp8_e4m3")
            packs.add(rank, name + "ffn_gate_up.scale", np.repeat(np.concatenate((gate_scale[blocks_window], up_scale[blocks_window])), BLOCK, axis=0), "f32")
            packs.add(rank, name + "ffn_down", np.asarray(down[:, window]), "fp8_e4m3")
            packs.add(rank, name + "ffn_down.scale", np.repeat(down_scale[:, blocks_window], BLOCK, axis=0), "f32")
        return
    experts, inter, local = geometry.experts, geometry.moe_inter, geometry.local
    packs.replicated(name + "router", checkpoint.raw(m + "gate.weight", "BF16", (experts, hidden)), "bf16")
    packs.replicated(name + "router_bias", np.asarray(checkpoint.raw(m + "gate.e_score_correction_bias", "F32", (experts,))), "f32")
    for rank in packs.ranks:
        owned = range(rank * local, (rank + 1) * local)
        for target, source, rows_out, cols_in in (("moe_w1", "gate_proj", inter, hidden), ("moe_w3", "up_proj", inter, hidden),
                                                  ("moe_w2", "down_proj", hidden, inter)):
            weights, scales = [], []
            for expert in owned:
                e = f"{m}experts.{expert}.{source}."
                weights.append(checkpoint.raw(e + "weight", "U8", (rows_out, cols_in // 2)))
                scale = checkpoint.raw(e + "weight_scale", "U8", (rows_out, cols_in // MX_GROUP))
                check_mxfp4_scales(e + "weight_scale", scale)
                scales.append(scale)
            packs.add_parts(rank, name + target, weights, "mxfp4_e2m1", (local, rows_out, cols_in // 2))
            packs.add_parts(rank, name + target + ".scale", scales, "e8m0", (local, rows_out, cols_in // MX_GROUP))


def finish(packs, geometry, revision, contract_sha256):
    full_layers = sum(1 for layer in range(geometry.layers) if not geometry.sliding(layer))
    for rank in packs.ranks:
        packs.writers[rank].close()
        layout = {"model": f"mimo26_{geometry.model}", "tp_degree": TP, "tp_rank": rank, "hidden": geometry.hidden, "layers": geometry.layers,
                  "full_layers": full_layers, "vocab": geometry.vocab, "vocab_first": rank * geometry.vocab_rows,
                  "vocab_rows": geometry.vocab_rows, "ffn_rows": geometry.dense_rows, "sliding_window": geometry.window,
                  "experts": geometry.experts, "local_experts": geometry.local, "expert_first": rank * geometry.local,
                  "top_k": geometry.top_k, "qkv_interleave": geometry.interleave}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": "fp8_e4m3_row_scale", "expert_codec": "mxfp4_e2m1_e8m0_g32", "revision": revision,
               "contract_sha256": contract_sha256}
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
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ranks", default=",".join(str(rank) for rank in range(TP)))
    args = parser.parse_args()
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    config = json.loads((args.checkpoint / "config.json").read_text())
    check_config(config, args.model)
    contract = json.loads(args.contract.read_text())
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    receipt = args.checkpoint / "DOWNLOAD-RECEIPT.json"
    revision = json.loads(receipt.read_text()).get("revision", "unknown") if receipt.exists() else "unknown"
    if contract.get("source_revision") not in (None, revision):
        raise PackFailure(f"checkpoint revision {revision} is not the contract's {contract.get('source_revision')}")
    checkpoint = Checkpoint(args.checkpoint)
    interleave = int(checkpoint.metadata.get("tp_size", 0))
    if interleave not in (4, 8) or config["num_attention_heads"] % interleave or config["num_key_value_heads"] % interleave or \
            config["swa_num_key_value_heads"] % interleave:
        raise PackFailure(f"index metadata tp_size {checkpoint.metadata.get('tp_size')} is not a qkv interleave this packer knows")
    geometry = Geometry(config, args.model, interleave)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packs = RankPacks(args.output_dir, args.resume, ranks, f"mimo26_{args.model}", source_identity(args.checkpoint, f"mimo26_{args.model}", files=(args.contract,)))
    hidden, vocab_rows = geometry.hidden, geometry.vocab_rows
    embed = checkpoint.raw("model.embed_tokens.weight", "BF16", (geometry.vocab, hidden))
    head = checkpoint.raw("lm_head.weight", "BF16", (geometry.vocab, hidden))
    for rank in ranks:
        packs.add(rank, "embed", embed[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
        packs.add(rank, "head", head[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
    packs.replicated("head_norm", checkpoint.bf16_as_f32("model.norm.weight", (hidden,)), "f32")
    for layer in range(geometry.layers):
        pack_layer(checkpoint, packs, geometry, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, geometry, revision, contract_sha256)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"mimo26_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
