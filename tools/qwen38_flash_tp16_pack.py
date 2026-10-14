#!/usr/bin/env python3
"""Qwen3.8 Flash (qwen4_exp) NVIDIA NVFP4 checkpoint -> sixteen TP16 named stage packs.

One pass over the checkpoint writes every rank. Output per rank:
<out>/qwen38_flash.tp16.rank<NN>.pack (tools/spark_named_pack.py container),
plus <pack>.sha256 and the empty-range <pack>.experts.

Rank r holds:
  embed / head        vocab rows [r*15520, +15520), bf16
  hyper-connections   every layer's attn and mlp mixer, and the final mixer,
                      replicated: hc_norm (1 + w) f32 [10240], down bf16
                      [320, 10240], up bf16 [10240, 320], inject bf16 [4, 10240]
  GDN                 key head r, value heads 3r..3r+2: gdn_qkvz [1024, 2560],
                      gdn_ba [6, 2560], gdn_conv [640, 4], A_log / dt_bias [3],
                      gdn_norm raw bf16 [128], gdn_out columns [2560, 384]
  attention           q_proj|k_proj|v_proj fused (13312 rows), rank rows
                      [r*832, +832); q and k norms (1 + w); o_proj columns
                      [r*384, +384); the indexer projection [640, 2560] and its
                      q / k norms replicated
  MoE                 router [512, 2560] replicated; local experts 32r..32r+31
                      as nvfp4 moe_w1 (gate) / moe_w3 (up) [32*640, 2560] and
                      moe_w2 (down) [32*2560, 640], each with a ue4m3 scale
                      plane whose 32 f32 weight_scale_2 globals come first;
                      the shared expert rows [r*40, +40), each half zero-padded
                      to 64 rows so the down projection's K is a multiple of
                      64: gate|up [128, 2560] and down columns [2560, 64];
                      the shared gate [1, 2560]
  PLE (layer 1)       key [10240, 2560], value [2560, 2560], the three norms
                      (1 + w) f32 [10240], the dilated conv [10240, 4]
                      replicated; the n-gram table rows [r*20000096, +20000096)
                      (checkpoint shards 8r..8r+7 back to back) as one fp8
                      [20000096, 160] tensor and its per-tensor scale f32 [1]
The hash multipliers, head vocab sizes and head offsets are replicated i64
tensors (and are listed in the config).
The MTP decoder and the vision tower are not packed.
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
HIDDEN = 2560
LAYERS = 48
VOCAB = 248320
VOCAB_ROWS = VOCAB // TP
HC = 4
HC_WIDTH = HC * HIDDEN
HC_LOWRANK = 320
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_KEY_DIM = 128
GDN_VALUE_DIM = 128
GDN_QK = GDN_KEY_HEADS * GDN_KEY_DIM
GDN_V = GDN_VALUE_HEADS * GDN_VALUE_DIM
ATTN_HEADS = 24
ATTN_KV_HEADS = 2
HEAD_DIM = 256
ATTN_Q = ATTN_HEADS * HEAD_DIM
ATTN_QG = 2 * ATTN_Q
ATTN_KV = ATTN_KV_HEADS * HEAD_DIM
ATTN_QKV = ATTN_QG + 2 * ATTN_KV
INDEX_HEADS = 4
INDEX_DIM = 128
INDEX_ROWS = (INDEX_HEADS + 1) * INDEX_DIM
FULL_PERIOD = 4
FULL_PHASE = 3
EXPERTS = 512
LOCAL_EXPERTS = EXPERTS // TP
TOP_K = 10
MOE_INTER = 640
SHARED_INTER = 640
SHARED_ROWS = SHARED_INTER // TP
SHARED_ROWS_PADDED = 64
PLE_LAYER = 1
PLE_SHARDS = 128
PLE_SHARDS_PER_RANK = PLE_SHARDS // TP
PLE_SHARD_ROWS = 2500012
PLE_HEAD_DIM = 160
PREFIX = "model.language_model."
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2


class PackFailure(RuntimeError):
    pass


class Checkpoint:
    DTYPES = {"BF16": np.uint16, "F8_E4M3": np.uint8, "U8": np.uint8, "F32": np.float32, "I64": np.int64}

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

    def array(self, name, dtype=None, shape=None):
        shard = self.weight_map.get(name)
        if shard is None:
            raise PackFailure(f"checkpoint has no tensor {name}")
        base, header = self._header(shard)
        meta = header[name]
        if dtype is not None and meta["dtype"] != dtype:
            raise PackFailure(f"{name}: expected {dtype}, found {meta['dtype']}")
        if shape is not None and tuple(meta["shape"]) != tuple(shape):
            raise PackFailure(f"{name}: shape {tuple(meta['shape'])} != {tuple(shape)}")
        element = self.DTYPES[meta["dtype"]]
        begin, end = meta["data_offsets"]
        count = (end - begin) // np.dtype(element).itemsize
        flat = np.memmap(self.directory / shard, dtype=element, mode="r", offset=base + begin, shape=(count,))
        return flat.reshape(meta["shape"]) if meta["shape"] else flat


def bf16_to_f32(raw):
    return (np.asarray(raw).astype(np.uint32) << 16).view(np.float32)


def folded(checkpoint, name, width):
    return bf16_to_f32(checkpoint.array(name, "BF16", (width,))) + np.float32(1.0)


def scalar_f32(checkpoint, name):
    shard = checkpoint.weight_map.get(name)
    if shard is None:
        raise PackFailure(f"checkpoint has no tensor {name}")
    _, header = checkpoint._header(shard)
    meta = header[name]
    value = checkpoint.array(name)
    if meta["dtype"] == "F32":
        result = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
    elif meta["dtype"] == "BF16":
        result = float(bf16_to_f32(np.asarray(value).reshape(-1)[:1])[0])
    else:
        raise PackFailure(f"{name}: expected an F32 or BF16 scalar, found {meta['dtype']}")
    if not np.isfinite(result) or result <= 0.0:
        raise PackFailure(f"{name}: scale {result} is not positive and finite")
    return result


class RankPacks:
    def __init__(self, out_dir, resume, ranks):
        self.out_dir = Path(out_dir)
        self.ranks = ranks
        self.writers = {rank: PayloadWriter(self.payload(rank), resume=resume and self.payload(rank).exists())
                        for rank in ranks}

    def payload(self, rank):
        return self.out_dir / f"qwen38_flash.tp16.rank{rank:02d}.pack.payload"

    def final(self, rank):
        return self.out_dir / f"qwen38_flash.tp16.rank{rank:02d}.pack"

    def add(self, rank, name, array, kind, extra=None):
        if rank not in self.writers:
            return
        array = np.ascontiguousarray(array)
        self.writers[rank].add(name, array.tobytes(), kind, array.shape, extra)

    def replicated(self, name, array, kind):
        array = np.ascontiguousarray(array)
        for rank in self.ranks:
            self.writers[rank].add(name, array.tobytes(), kind, array.shape)


def pack_globals(checkpoint, packs):
    embed = checkpoint.array(PREFIX + "embed_tokens.weight", "BF16", (VOCAB, HIDDEN))
    head = checkpoint.array("lm_head.weight", "BF16", (VOCAB, HIDDEN))
    for rank in range(TP):
        first = rank * VOCAB_ROWS
        packs.add(rank, "embed", embed[first:first + VOCAB_ROWS], "bf16")
        packs.add(rank, "head", head[first:first + VOCAB_ROWS], "bf16")
    pack_mixer(checkpoint, packs, PREFIX + "hyper_connection_mixer.", "final_hc_", False)


def pack_mixer(checkpoint, packs, source, name, inject):
    packs.replicated(name + "norm", folded(checkpoint, source + "hc_norm.weight", HC_WIDTH), "f32")
    packs.replicated(name + "down", checkpoint.array(source + "input_mix_weight_down.weight", "BF16", (HC_LOWRANK, HC_WIDTH)), "bf16")
    packs.replicated(name + "up", checkpoint.array(source + "input_mix_weight_up.weight", "BF16", (HC_WIDTH, HC_LOWRANK)), "bf16")
    if inject:
        packs.replicated(name + "inject", checkpoint.array(source + "block_inject_weight.weight", "BF16", (HC, HC_WIDTH)), "bf16")


def pack_gdn(checkpoint, packs, layer):
    p = f"{PREFIX}layers.{layer}.linear_attn."
    qkv = checkpoint.array(p + "in_proj_qkv.weight", "BF16", (2 * GDN_QK + GDN_V, HIDDEN))
    z = checkpoint.array(p + "in_proj_z.weight", "BF16", (GDN_V, HIDDEN))
    out = checkpoint.array(p + "out_proj.weight", "BF16", (HIDDEN, GDN_V))
    beta = checkpoint.array(p + "in_proj_b.weight", "BF16", (GDN_VALUE_HEADS, HIDDEN))
    decay = checkpoint.array(p + "in_proj_a.weight", "BF16", (GDN_VALUE_HEADS, HIDDEN))
    conv = checkpoint.array(p + "conv1d.weight", "BF16", (2 * GDN_QK + GDN_V, 1, 4)).reshape(2 * GDN_QK + GDN_V, 4)
    a_log = bf16_to_f32(checkpoint.array(p + "A_log", "BF16", (GDN_VALUE_HEADS,)))
    dt_bias = bf16_to_f32(checkpoint.array(p + "dt_bias", "BF16", (GDN_VALUE_HEADS,)))
    norm = checkpoint.array(p + "norm.weight", "BF16", (GDN_VALUE_DIM,))
    heads = GDN_VALUE_HEADS // TP
    for rank in range(TP):
        q = (rank * GDN_KEY_DIM, GDN_KEY_DIM)
        k = (GDN_QK + rank * GDN_KEY_DIM, GDN_KEY_DIM)
        v = (2 * GDN_QK + rank * heads * GDN_VALUE_DIM, heads * GDN_VALUE_DIM)
        zr = (rank * heads * GDN_VALUE_DIM, heads * GDN_VALUE_DIM)
        name = f"layers.{layer}."
        packs.add(rank, name + "gdn_qkvz", np.concatenate([qkv[f:f + c] for f, c in (q, k, v)] + [z[zr[0]:zr[0] + zr[1]]]), "bf16")
        packs.add(rank, name + "gdn_ba", np.concatenate((beta[rank * heads:(rank + 1) * heads], decay[rank * heads:(rank + 1) * heads])), "bf16")
        packs.add(rank, name + "gdn_conv", np.concatenate([conv[f:f + c] for f, c in (q, k, v)]), "bf16")
        packs.add(rank, name + "gdn_a_log", a_log[rank * heads:(rank + 1) * heads], "f32")
        packs.add(rank, name + "gdn_dt_bias", dt_bias[rank * heads:(rank + 1) * heads], "f32")
        packs.add(rank, name + "gdn_norm", norm, "bf16")
        packs.add(rank, name + "gdn_out", out[:, zr[0]:zr[0] + zr[1]], "bf16")


def pack_attention(checkpoint, packs, layer):
    p = f"{PREFIX}layers.{layer}.self_attn."
    q = checkpoint.array(p + "q_proj.weight", "BF16", (ATTN_QG, HIDDEN))
    k = checkpoint.array(p + "k_proj.weight", "BF16", (ATTN_KV, HIDDEN))
    v = checkpoint.array(p + "v_proj.weight", "BF16", (ATTN_KV, HIDDEN))
    o = checkpoint.array(p + "o_proj.weight", "BF16", (HIDDEN, ATTN_Q))
    fused = np.concatenate((q, k, v))
    rows = ATTN_QKV // TP
    columns = ATTN_Q // TP
    name = f"layers.{layer}."
    for rank in range(TP):
        packs.add(rank, name + "attn_qkv", fused[rank * rows:(rank + 1) * rows], "bf16")
        packs.add(rank, name + "attn_out", o[:, rank * columns:(rank + 1) * columns], "bf16")
    packs.replicated(name + "attn_q_norm", folded(checkpoint, p + "q_norm.weight", HEAD_DIM), "f32")
    packs.replicated(name + "attn_k_norm", folded(checkpoint, p + "k_norm.weight", HEAD_DIM), "f32")
    packs.replicated(name + "attn_index_qk", checkpoint.array(p + "indexer.index_qk_proj.weight", "BF16", (INDEX_ROWS, HIDDEN)), "bf16")
    packs.replicated(name + "attn_index_q_norm", folded(checkpoint, p + "indexer.q_layernorm.weight", INDEX_DIM), "f32")
    packs.replicated(name + "attn_index_k_norm", folded(checkpoint, p + "indexer.k_layernorm.weight", INDEX_DIM), "f32")


def expert_planes(checkpoint, source, proj, experts, rows, cols):
    payloads, planes = [], []
    globals_ = np.empty(len(experts), dtype=np.float32)
    for slot, expert in enumerate(experts):
        p = f"{source}{expert}.{proj}."
        payloads.append(np.asarray(checkpoint.array(p + "weight", "U8", (rows, cols // 2))))
        planes.append(np.asarray(checkpoint.array(p + "weight_scale", "F8_E4M3", (rows, cols // 16))))
        globals_[slot] = scalar_f32(checkpoint, p + "weight_scale_2")
    return np.concatenate(payloads), globals_, np.concatenate(planes)


def pack_moe(checkpoint, packs, layer):
    p = f"{PREFIX}layers.{layer}.mlp."
    name = f"layers.{layer}."
    packs.replicated(name + "moe_router", checkpoint.array(p + "gate.weight", "BF16", (EXPERTS, HIDDEN)), "bf16")
    packs.replicated(name + "moe_shared_gate", checkpoint.array(p + "shared_expert_gate.weight", "BF16", (1, HIDDEN)), "bf16")
    shared_gate = checkpoint.array(p + "shared_expert.gate_proj.weight", "BF16", (SHARED_INTER, HIDDEN))
    shared_up = checkpoint.array(p + "shared_expert.up_proj.weight", "BF16", (SHARED_INTER, HIDDEN))
    shared_down = checkpoint.array(p + "shared_expert.down_proj.weight", "BF16", (HIDDEN, SHARED_INTER))
    pad_rows = np.zeros((SHARED_ROWS_PADDED - SHARED_ROWS, HIDDEN), dtype=np.uint16)
    pad_columns = np.zeros((HIDDEN, SHARED_ROWS_PADDED - SHARED_ROWS), dtype=np.uint16)
    for rank in range(TP):
        rows = slice(rank * SHARED_ROWS, (rank + 1) * SHARED_ROWS)
        packs.add(rank, name + "moe_shared_gate_up", np.concatenate((shared_gate[rows], pad_rows, shared_up[rows], pad_rows)), "bf16")
        packs.add(rank, name + "moe_shared_down", np.concatenate((shared_down[:, rows], pad_columns), axis=1), "bf16")
        if rank not in packs.writers:
            continue
        experts = range(rank * LOCAL_EXPERTS, (rank + 1) * LOCAL_EXPERTS)
        for kind, proj, rows_, cols in (("moe_w1", "gate_proj", MOE_INTER, HIDDEN), ("moe_w3", "up_proj", MOE_INTER, HIDDEN),
                                        ("moe_w2", "down_proj", HIDDEN, MOE_INTER)):
            payload, globals_, planes = expert_planes(checkpoint, p + "experts.", proj, experts, rows_, cols)
            packs.writers[rank].add(name + kind, payload.tobytes(), "nvfp4_e2m1", (LOCAL_EXPERTS * rows_, cols))
            packs.writers[rank].add(name + kind + ".scale", globals_.tobytes() + planes.tobytes(), "ue4m3_f32_global",
                                    (LOCAL_EXPERTS, rows_, cols // 16), {"global_scales": LOCAL_EXPERTS})


def pack_ple(checkpoint, packs):
    p = f"{PREFIX}layers.{PLE_LAYER}.ple."
    name = f"layers.{PLE_LAYER}."
    packs.replicated(name + "ple_key", checkpoint.array(p + "key_proj.weight", "BF16", (HC_WIDTH, HIDDEN)), "bf16")
    packs.replicated(name + "ple_value", checkpoint.array(p + "value_proj.weight", "BF16", (HIDDEN, HIDDEN)), "bf16")
    for norm in ("norm_key", "norm_query", "norm_conv"):
        packs.replicated(name + "ple_" + norm, folded(checkpoint, p + norm + ".weight", HC_WIDTH), "f32")
    packs.replicated(name + "ple_conv", checkpoint.array(p + "conv1d.weight", "BF16", (HC_WIDTH, 1, 4)).reshape(HC_WIDTH, 4), "bf16")
    scale = np.array([scalar_f32(checkpoint, p + "ple_embedding.ngram_embedding.weight_scale")], dtype=np.float32)
    hashing = f"{p}ple_embedding."
    packs.replicated(name + "ple_multipliers", np.asarray(checkpoint.array(hashing + "layer_multipliers", "I64", (3,))), "i64")
    packs.replicated(name + "ple_vocab_sizes", np.asarray(checkpoint.array(hashing + "ngram_heads_vocab_sizes", "I64", (16,))), "i64")
    packs.replicated(name + "ple_offsets", np.asarray(checkpoint.array(hashing + "ngram_heads_offsets", "I64", (16,))), "i64")
    for rank in packs.ranks:
        shards = [checkpoint.array(f"{p}ple_embedding.ngram_embedding.shard_{shard}.weight", "F8_E4M3", (PLE_SHARD_ROWS, PLE_HEAD_DIM))
                  for shard in range(rank * PLE_SHARDS_PER_RANK, (rank + 1) * PLE_SHARDS_PER_RANK)]
        packs.writers[rank].add_parts(name + "ple_table", shards, "fp8_e4m3", (PLE_SHARDS_PER_RANK * PLE_SHARD_ROWS, PLE_HEAD_DIM))
        packs.add(rank, name + "ple_table_scale", scale, "f32")


def ple_config(checkpoint, text):
    p = f"{PREFIX}layers.{PLE_LAYER}.ple.ple_embedding."
    multipliers = [int(value) for value in checkpoint.array(p + "layer_multipliers", "I64", (3,))]
    sizes = [int(value) for value in checkpoint.array(p + "ngram_heads_vocab_sizes", "I64", (16,))]
    offsets = [int(value) for value in checkpoint.array(p + "ngram_heads_offsets", "I64", (16,))]
    if offsets != [sum(sizes[:head]) for head in range(16)]:
        raise PackFailure("ngram head offsets are not the running sum of the head vocab sizes")
    total = offsets[-1] + sizes[-1]
    divisor = text["make_ngram_vocab_size_divisible_by"]
    if -(-total // divisor) * divisor != PLE_SHARDS * PLE_SHARD_ROWS:
        raise PackFailure(f"ngram table rows {total} padded to {divisor} are not {PLE_SHARDS} shards of {PLE_SHARD_ROWS}")
    return {"ple_layer": PLE_LAYER, "ple_ngram": text["ngram_size"], "ple_heads_per_ngram": text["heads_per_ngram"],
            "ple_head_dim": PLE_HEAD_DIM, "ple_conv_kernel": text["ple_conv_kernel_size"], "ple_conv_dilation": text["ngram_size"],
            "ple_multipliers": multipliers, "ple_vocab_sizes": sizes, "ple_offsets": offsets,
            "ple_table_rows": PLE_SHARDS_PER_RANK * PLE_SHARD_ROWS, "ple_shard_rows": PLE_SHARD_ROWS,
            "ple_eos": text["eos_token_id"]}


def finish(packs, revision, contract_sha256, ple):
    for rank in packs.ranks:
        packs.writers[rank].close()
        config = {"model": "qwen38_flash", "tp_degree": TP, "tp_rank": rank, "hidden": HIDDEN, "layers": LAYERS,
                  "vocab": VOCAB, "vocab_first": rank * VOCAB_ROWS, "vocab_rows": VOCAB_ROWS,
                  "full_period": FULL_PERIOD, "full_phase": FULL_PHASE, "experts": EXPERTS,
                  "local_experts": LOCAL_EXPERTS, "expert_first": rank * LOCAL_EXPERTS, "top_k": TOP_K,
                  "moe_inter": MOE_INTER, "shared_rows": SHARED_ROWS_PADDED, "shared_rows_real": SHARED_ROWS, "hc_streams": HC, "hc_lowrank": HC_LOWRANK,
                  "index_heads": INDEX_HEADS, "index_dim": INDEX_DIM, "index_budget": 2048, "index_ratio": 4,
                  "ple_table_first": rank * PLE_SHARDS_PER_RANK * PLE_SHARD_ROWS, **ple}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": "bf16", "expert_codec": "nvfp4_e2m1", "attn_qkv_layout": "fused_windows",
               "revision": revision, "contract_sha256": contract_sha256}
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


def check_geometry(text):
    expected = {"hidden_size": HIDDEN, "num_hidden_layers": LAYERS, "vocab_size": VOCAB, "num_attention_heads": ATTN_HEADS,
                "num_key_value_heads": ATTN_KV_HEADS, "head_dim": HEAD_DIM, "linear_num_key_heads": GDN_KEY_HEADS,
                "linear_num_value_heads": GDN_VALUE_HEADS, "num_experts": EXPERTS, "num_experts_per_tok": TOP_K,
                "moe_intermediate_size": MOE_INTER, "shared_expert_intermediate_size": SHARED_INTER, "hc_count": HC,
                "hc_lowrank": HC_LOWRANK, "indexer_n_heads": INDEX_HEADS, "indexer_kv_heads": 1,
                "indexer_head_dim": INDEX_DIM, "full_attention_interval": FULL_PERIOD, "split_ngram_parts": PLE_SHARDS,
                "ple_embed_dim": HIDDEN, "ple_layer_ids": [PLE_LAYER + 1], "norm_topk_prob": True,
                "output_gate_type": "sigmoid", "tie_word_embeddings": False}
    differing = {key: (text.get(key), value) for key, value in expected.items() if text.get(key) != value}
    if differing:
        raise PackFailure(f"checkpoint geometry differs from the packer's: {differing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ranks", default=",".join(str(rank) for rank in range(TP)),
                        help="comma-separated ranks to write (default every rank)")
    args = parser.parse_args()
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    text = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    check_geometry(text)
    receipt = args.checkpoint / "DOWNLOAD-RECEIPT.json"
    revision = json.loads(receipt.read_text()).get("revision", "unknown") if receipt.exists() else "unknown"
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint)
    ple = ple_config(checkpoint, text)
    packs = RankPacks(args.output_dir, args.resume, ranks)
    pack_globals(checkpoint, packs)
    for layer in range(LAYERS):
        name = f"{PREFIX}layers.{layer}."
        pack_mixer(checkpoint, packs, name + "attn_hyper_connection.", f"layers.{layer}.attn_hc_", True)
        pack_mixer(checkpoint, packs, name + "mlp_hyper_connection.", f"layers.{layer}.mlp_hc_", True)
        if layer == PLE_LAYER:
            pack_ple(checkpoint, packs)
        if layer % FULL_PERIOD == FULL_PHASE:
            pack_attention(checkpoint, packs, layer)
        else:
            pack_gdn(checkpoint, packs, layer)
        pack_moe(checkpoint, packs, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, revision, contract_sha256, ple)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"qwen38_flash_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
