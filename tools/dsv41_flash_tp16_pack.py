#!/usr/bin/env python3
"""DeepSeek V4.1 Flash (the official FP8/FP4 release) -> named stage packs for the Metal ranks.

Output in --output-dir, from the checkpoint at its pinned revision:
  dsv41_flash.tp<T>.rank<NN>.pack   one per rank (tools/spark_named_pack.py
                                    container), with <pack>.sha256 and the
                                    empty-range <pack>.experts
  dsv41_flash.engram.pack           the two engram tables, shared by every
                                    rank and degree, with its .sha256; the
                                    Metal stage reads their rows on the host

FP8 projections keep the official E4M3 bytes as an "fp8" tensor [rows, K]
with its "<name>_scale" "u8" tensor, the UE8M0 byte of the checkpoint's
32 x 32 block repeated over the block's rows ([rows, K/32]); the routed
experts keep their MXFP4 planes ("mxfp4_e2m1" [E/T, rows, K/2] and
"<name>.scale" "e8m0" [E/T, rows, K/32]). Norm gains are widened to f32.

Rank r of T holds:
  embed, head          vocab rows [r V/T, +V/T), bf16
  final_norm           norm.weight, f32
  engram.*             the hash tables of tools/dsv41_flash_engram_tables.py:
                       token_map i32 [V], multipliers i64 [2, 4], primes and
                       offsets i64 [2, 24]
  every layer          attn_hc_* / ffn_hc_* (fn, base, scale; f32), attn_norm,
                       ffn_norm; q_a, kv (wkv) replicated, q_norm, kv_norm;
                       q_b rows of the rank's 64/T heads; sinks of those heads;
                       o_a rows of the rank's 8/T output groups; o_b columns
                       [r 8192/T, +8192/T); router bf16 and router_bias f32;
                       experts [r 384/T, +384/T) as moe_w1 / moe_w3 / moe_w2;
                       shared_gate_up (w1 and w3 rows [r 2304/T, +2304/T)) and
                       shared_down (w2 columns of the same window)
  layers 2, 8, 14, 20  c_kv, c_gate (ratio 2 only) bf16, c_norm f32
  layers 1, 14         engram_kv rows [r 25600/T, +25600/T), engram_query and
                       engram_key bf16 [4, 5120]
The indexer, the dspark blocks and the vision tower are not packed.

usage: dsv41_flash_tp16_pack.py --checkpoint DIR --output-dir DIR --contract model_contracts/dsv41_flash_authoritative.json
       --tp-degree T [--part ranks|engram|all] [--ranks 0,1,...] [--resume]
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
from dsv41_flash_engram_tables import tables as engram_hash_tables  # noqa: E402

BLOCK = 32
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2
DTYPES = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "I8": np.uint8}
ENGRAM_FILE = "dsv41_flash.engram.pack"
CHUNK_ROWS = 1 << 22


class PackFailure(RuntimeError):
    pass


class Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.weight_map = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
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

    def bf16(self, name, shape):
        return self.raw(name, "BF16", shape)

    def f32(self, name, shape):
        return np.asarray(self.raw(name, "F32", shape))

    def widened(self, name, shape):
        return (np.asarray(self.bf16(name, shape)).astype(np.uint32) << 16).view(np.float32)

    def fp8(self, name, rows, columns):
        weight = self.raw(name + ".weight", "F8_E4M3", (rows, columns))
        scale = self.raw(name + ".scale", "F8_E8M0", (rows // BLOCK, columns // BLOCK))
        return weight, scale

    def mxfp4(self, name, rows, columns):
        return self.raw(name + ".weight", "I8", (rows, columns // 2)), self.raw(name + ".scale", "F8_E8M0", (rows, columns // BLOCK))


def scaled_window(pair, first_row, rows, first_column, columns):
    weight, scale = pair
    row_scale = np.repeat(np.asarray(scale[first_row // BLOCK:(first_row + rows + BLOCK - 1) // BLOCK]), BLOCK, axis=0)
    row_scale = row_scale[first_row % BLOCK:first_row % BLOCK + rows, first_column // BLOCK:(first_column + columns) // BLOCK]
    return np.asarray(weight[first_row:first_row + rows, first_column:first_column + columns]), row_scale


def stacked(pairs, first_row, rows, first_column, columns):
    parts = [scaled_window(pair, first_row, rows, first_column, columns) for pair in pairs]
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


class RankPacks:
    def __init__(self, out_dir, tp, ranks, identity, resume):
        self.out_dir = Path(out_dir)
        self.tp = tp
        self.ranks = ranks
        self.writers = {rank: PayloadWriter(self.payload(rank), identity, resume=resume and self.payload(rank).exists()) for rank in ranks}

    def final(self, rank):
        return self.out_dir / f"dsv41_flash.tp{self.tp}.rank{rank:02d}.pack"

    def payload(self, rank):
        return Path(str(self.final(rank)) + ".payload")

    def add(self, rank, name, array, kind):
        array = np.ascontiguousarray(array)
        self.writers[rank].add(name, array.tobytes(), kind, array.shape)

    def add_parts(self, rank, name, parts, kind, shape):
        self.writers[rank].add_parts(name, parts, kind, shape)

    def scaled(self, rank, name, pair):
        self.add(rank, name, pair[0], "fp8")
        self.add(rank, name + "_scale", pair[1], "u8")

    def replicated(self, name, array, kind):
        for rank in self.ranks:
            self.add(rank, name, array, kind)


def engram_tables(config):
    text = config["text_config"]
    token_map = np.load(Path(__file__).resolve().parents[1] / "qualification/t1_reference/dsv41/engram_token_map.npy").astype(np.int64)
    derived = engram_hash_tables(text, token_map)
    if derived["rows"] != [int(v) for v in text["engram_num_embeddings"]]:
        raise PackFailure(f"engram tables give rows {derived['rows']}, the checkpoint {text['engram_num_embeddings']}")
    return {"token_map": token_map.astype(np.int32), "multipliers": np.array(derived["multipliers"], dtype=np.int64),
            "primes": np.array(derived["primes"], dtype=np.int64), "offsets": np.array(derived["offsets"], dtype=np.int64), "rows": derived["rows"],
            "pad": derived["pad"], "layers": derived["layers"]}


def pack_layer(checkpoint, packs, text, engram, layer):
    tp = packs.tp
    hidden, heads, head_dim = text["hidden_size"], text["num_attention_heads"], text["head_dim"]
    q_lora, o_lora, groups = text["q_lora_rank"], text["o_lora_rank"], text["o_groups"]
    experts, inter = text["n_routed_experts"], text["moe_intermediate_size"]
    rank_heads, rank_groups, local, shared_rows = heads // tp, groups // tp, experts // tp, inter // tp
    group_input = heads * head_dim // groups
    name, a, f = f"layers.{layer}.", f"layers.{layer}.attn.", f"layers.{layer}.ffn."
    for target, source in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        packs.replicated(name + target + "_fn", checkpoint.f32(f"layers.{layer}.{source}_fn", (24, 4 * hidden)), "f32")
        packs.replicated(name + target + "_base", checkpoint.f32(f"layers.{layer}.{source}_base", (24,)), "f32")
        packs.replicated(name + target + "_scale", checkpoint.f32(f"layers.{layer}.{source}_scale", (3,)), "f32")
    packs.replicated(name + "attn_norm", checkpoint.widened(f"layers.{layer}.attn_norm.weight", (hidden,)), "f32")
    packs.replicated(name + "ffn_norm", checkpoint.widened(f"layers.{layer}.ffn_norm.weight", (hidden,)), "f32")
    q_a, kv = checkpoint.fp8(a + "wq_a", q_lora, hidden), checkpoint.fp8(a + "wkv", head_dim, hidden)
    q_b, o_a, o_b = checkpoint.fp8(a + "wq_b", heads * head_dim, q_lora), checkpoint.fp8(a + "wo_a", groups * o_lora, group_input), \
        checkpoint.fp8(a + "wo_b", hidden, groups * o_lora)
    sinks = checkpoint.f32(a + "attn_sink", (heads,))
    for rank in packs.ranks:
        packs.scaled(rank, name + "q_a", scaled_window(q_a, 0, q_lora, 0, hidden))
        packs.add(rank, name + "q_norm", checkpoint.widened(a + "q_norm.weight", (q_lora,)), "f32")
        packs.scaled(rank, name + "q_b", scaled_window(q_b, rank * rank_heads * head_dim, rank_heads * head_dim, 0, q_lora))
        packs.scaled(rank, name + "kv", scaled_window(kv, 0, head_dim, 0, hidden))
        packs.add(rank, name + "kv_norm", checkpoint.widened(a + "kv_norm.weight", (head_dim,)), "f32")
        packs.add(rank, name + "sinks", sinks[rank * rank_heads:(rank + 1) * rank_heads], "f32")
        packs.scaled(rank, name + "o_a", scaled_window(o_a, rank * rank_groups * o_lora, rank_groups * o_lora, 0, group_input))
        packs.scaled(rank, name + "o_b", scaled_window(o_b, 0, hidden, rank * rank_groups * o_lora, rank_groups * o_lora))
    ratio = text["compress_ratios"][layer]
    if layer in text["kv_source_layer_ids"]:
        packs.replicated(name + "c_kv", checkpoint.bf16(a + "compressor.wkv.weight", (head_dim, hidden)), "bf16")
        if ratio == 2:
            packs.replicated(name + "c_gate", checkpoint.bf16(a + "compressor.wgate.weight", (head_dim, hidden)), "bf16")
        packs.replicated(name + "c_norm", checkpoint.widened(a + "compressor.norm.weight", (head_dim,)), "f32")
    packs.replicated(name + "router", checkpoint.bf16(f + "gate.weight", (experts, hidden)), "bf16")
    packs.replicated(name + "router_bias", checkpoint.f32(f + "gate.bias", (experts,)), "f32")
    for target, source, rows, columns in (("moe_w1", "w1", inter, hidden), ("moe_w3", "w3", inter, hidden), ("moe_w2", "w2", hidden, inter)):
        for rank in packs.ranks:
            owned = range(rank * local, (rank + 1) * local)
            planes = [checkpoint.mxfp4(f"{f}experts.{e}.{source}", rows, columns) for e in owned]
            packs.add_parts(rank, name + target, (plane[0] for plane in planes), "mxfp4_e2m1", (local, rows, columns // 2))
            packs.add_parts(rank, name + target + ".scale", (plane[1] for plane in planes), "e8m0", (local, rows, columns // BLOCK))
    w1, w3, w2 = (checkpoint.fp8(f"{f}shared_experts.{w}", r, c) for w, r, c in (("w1", inter, hidden), ("w3", inter, hidden), ("w2", hidden, inter)))
    for rank in packs.ranks:
        packs.scaled(rank, name + "shared_gate_up", stacked((w1, w3), rank * shared_rows, shared_rows, 0, hidden))
        packs.scaled(rank, name + "shared_down", scaled_window(w2, 0, hidden, rank * shared_rows, shared_rows))
    if layer in engram["layers"]:
        e = f"layers.{layer}.engram."
        streams = text["hc_mult"]
        kv_rows, embed = (streams + 1) * hidden, engram_embed(text)
        wkv = checkpoint.fp8(e + "wkv", kv_rows, embed)
        for rank in packs.ranks:
            packs.scaled(rank, name + "engram_kv", scaled_window(wkv, rank * kv_rows // tp, kv_rows // tp, 0, embed))
            packs.add(rank, name + "engram_query", checkpoint.bf16(e + "q_weight", (streams, hidden)), "bf16")
            packs.add(rank, name + "engram_key", checkpoint.bf16(e + "k_weight", (streams, hidden)), "bf16")


def engram_embed(text):
    return (int(text["engram_max_ngram_size"]) - 1) * int(text["engram_n_heads"]) * int(text["engram_head_dim"])


def write_sha256(final):
    digest = hashlib.sha256()
    with final.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    Path(str(final) + ".sha256").write_text(f"{digest.hexdigest()}  {final.name}\n")
    return digest.hexdigest()


def pack_ranks(args, checkpoint, config, engram, revision, contract_sha256):
    text, tp = config["text_config"], args.tp_degree
    ranks = sorted({int(rank) for rank in args.ranks.split(",")}) if args.ranks else list(range(tp))
    if not ranks or ranks[0] < 0 or ranks[-1] >= tp:
        raise PackFailure(f"--ranks must name ranks in 0..{tp - 1}")
    packs = RankPacks(args.output_dir, tp, ranks, source_identity(args.checkpoint, "dsv41_flash_tp16_pack", tp, files=(args.contract,)), args.resume)
    hidden, vocab = text["hidden_size"], text["vocab_size"]
    vocab_rows = vocab // tp
    embed, head = checkpoint.bf16("embed.weight", (vocab, hidden)), checkpoint.bf16("head.weight", (vocab, hidden))
    for rank in ranks:
        packs.add(rank, "embed", embed[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
        packs.add(rank, "head", head[rank * vocab_rows:(rank + 1) * vocab_rows], "bf16")
    packs.replicated("final_norm", checkpoint.widened("norm.weight", (hidden,)), "f32")
    packs.replicated("engram.token_map", engram["token_map"], "i32")
    packs.replicated("engram.multipliers", engram["multipliers"], "i64")
    packs.replicated("engram.primes", engram["primes"], "i64")
    packs.replicated("engram.offsets", engram["offsets"], "i64")
    for layer in range(text["num_hidden_layers"]):
        pack_layer(checkpoint, packs, text, engram, layer)
        print(f"layer {layer} packed", flush=True)
    local = text["n_routed_experts"] // tp
    for rank in ranks:
        packs.writers[rank].close()
        layout = {"model": "dsv41_flash", "tp_degree": tp, "tp_rank": rank, "hidden": hidden, "layers": text["num_hidden_layers"], "vocab": vocab,
                  "vocab_first": rank * vocab_rows, "vocab_rows": vocab_rows, "heads": text["num_attention_heads"] // tp,
                  "experts": text["n_routed_experts"], "local_experts": local, "expert_first": rank * local, "top_k": text["num_experts_per_tok"],
                  "shared_rows": text["moe_intermediate_size"] // tp, "engram_rows_0": engram["rows"][0], "engram_rows_1": engram["rows"][1],
                  "engram_pad": engram["pad"]}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN, "linear_codec": "mxfp8",
               "expert_codec": "mxfp4", "revision": revision, "contract_sha256": contract_sha256, "engram_file": ENGRAM_FILE}
        final = packs.final(rank)
        assemble(final, packs.payload(rank), NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, layout, packs.writers[rank].manifest)
        digest = write_sha256(final)
        Path(str(final) + ".experts").write_bytes(struct.pack("<4I", EXPERTS_MAGIC, EXPERTS_VERSION, 0, 0))
        print(f"rank {rank:02d} {final.stat().st_size} {digest}", flush=True)


def pack_engram(args, checkpoint, config, engram, revision, contract_sha256):
    text = config["text_config"]
    final = args.output_dir / ENGRAM_FILE
    payload = Path(str(final) + ".payload")
    writer = PayloadWriter(payload, source_identity(args.checkpoint, "dsv41_flash_engram", files=(args.contract,)), resume=args.resume and payload.exists())
    dim = int(text["engram_head_dim"])
    for which, (layer, rows) in enumerate(zip(engram["layers"], engram["rows"])):
        e = f"layers.{layer}.engram.embed."
        weight = checkpoint.raw(e + "weight", "F8_E4M3", (rows, dim))
        scale = checkpoint.raw(e + "scale", "F8_E8M0", (rows, dim // BLOCK))
        writer.add_parts(f"engram.{which}.payload", (weight[first:first + CHUNK_ROWS] for first in range(0, rows, CHUNK_ROWS)), "fp8", (rows, dim))
        writer.add_parts(f"engram.{which}.scale", (scale[first:first + CHUNK_ROWS] for first in range(0, rows, CHUNK_ROWS)), "e8m0", (rows, dim // BLOCK))
        print(f"engram {which} (layer {layer}) packed: {rows} rows", flush=True)
    writer.close()
    layout = {"model": "dsv41_flash", "engram_rows_0": engram["rows"][0], "engram_rows_1": engram["rows"][1], "engram_pad": engram["pad"]}
    fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN, "revision": revision,
           "contract_sha256": contract_sha256}
    assemble(final, payload, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, layout, writer.manifest)
    print(f"engram {final.stat().st_size} {write_sha256(final)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tp-degree", type=int, required=True, choices=(1, 2, 4, 8))
    parser.add_argument("--part", choices=("ranks", "engram", "all"), default="all")
    parser.add_argument("--ranks")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    text = config["text_config"]
    contract = json.loads(args.contract.read_text())
    if hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest() != contract["sources"]["pinned_files"]["config.json"]["sha256"]:
        raise PackFailure("the checkpoint's config.json is not the contract's pinned revision")
    revision = contract["sources"]["hf_revision"]
    receipt = args.checkpoint / f".RECEIPT-{revision[:8]}.json"
    if not receipt.exists():
        raise PackFailure(f"{receipt} is missing: pack only a checkpoint with its revision receipt")
    if config.get("quantization_config", {}).get("weight_block_size") != [BLOCK, BLOCK] or text["hc_mult"] != 4 or text["num_key_value_heads"] != 1 or \
            text["o_groups"] % args.tp_degree or text["num_attention_heads"] % args.tp_degree or text["n_routed_experts"] % args.tp_degree or \
            text["vocab_size"] % args.tp_degree or text["moe_intermediate_size"] % (args.tp_degree * BLOCK):
        raise PackFailure("this packer handles the DeepSeek V4.1 Flash layout at TP 1, 2, 4 or 8")
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint(args.checkpoint)
    engram = engram_tables(config)
    if args.part in ("engram", "all"):
        pack_engram(args, checkpoint, config, engram, revision, contract_sha256)
    if args.part in ("ranks", "all"):
        pack_ranks(args, checkpoint, config, engram, revision, contract_sha256)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"dsv41_flash_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
