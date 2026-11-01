#!/usr/bin/env python3
"""GLM 5.3 (Full) or A.X K2 FP8 checkpoint -> TP16 (or --tp-degree N) named stage packs for the device stage.

Output per rank: <out>/<model>.tp<N>.rank<NN>.pack (tools/spark_named_pack.py
container) plus <pack>.sha256 and the empty-range <pack>.experts. The MTP layer
and the sparse-attention indexer are not packed: the device stage refuses rows
that would see more keys than the indexer selects, where attention is dense.

The checkpoint is read through tools/glm52_resident_stagepack.py's
Fp8SourceReader. Projections keep their E4M3 codes with the 128x128 block
scale expanded to one FP32 scale per row and 128 inputs, as the CUDA fp8lin
packs and the skinny FP8 kernels read them; kv_b decodes to BF16 as
bf16(e4m3 x scale), as the CUDA pack does.

Rank r holds (H hidden, R = 1/TP of a sharded dimension, Q the query
LoRA width, q the rows per query head: nope + rope (+ gate on A.X K2)):
  embed, head          bf16 vocab rows [r*V/TP, +V/TP); head_norm f32 [H]
  every layer          input_norm, post_norm f32 [H]; A.X K2 adds the
                       rank-16 norm gates input_gate_down bf16 [16, H] and
                       input_gate_up [H, 16] (post_gate_* on routed layers)
                       mla_q_a fp8_e4m3 [Q, H], mla_kv_a [576, H]
                       (replicated); mla_q_b the rank's heads [heads*q, Q, or
                       2Q on A.X K2 whose q_b reads the normed and raw
                       query]; mla_out [H, heads*value]; each with a .scale
                       f32 [rows, cols/128]; mla_q_norm f32 [Q], mla_kv_norm
                       f32 [512]; mla_uk bf16 [heads, 512, nope]; mla_uv bf16
                       [heads, value, 512]
  dense layers         ffn_gate_up fp8_e4m3: gate then up rows [2*I/TP, H];
                       ffn_down [H, I/TP]; with .scale
  routed layers        router bf16 [256, H], router_bias f32 [256]
                       (replicated); all 256 experts split by intermediate
                       width: moe_w1 (gate) and moe_w3 (up) fp8 [256,
                       2048/TP, H] with *_scale f32 [256, 2048/TP, H/128];
                       moe_w2 fp8 [256, H, 2048/TP] with moe_w2_scale f32;
                       shared_gate_up fp8_e4m3 gate then up [2*2048/TP, H];
                       shared_down [H, 2048/TP]; with .scale
"""

import argparse
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm52_resident_stagepack import Fp8SourceReader  # noqa: E402
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, PayloadWriter, assemble, source_identity  # noqa: E402
from spark_pack_common import PackFailure  # noqa: E402

MODELS = {
    "glm53full": {"name": "glm-5.3-full", "revision": "935644c05e76fc198714f4cca449fd8b970ff6d7", "hidden": 6144, "layers": 78, "vocab": 154880,
                  "heads": 64, "nope": 192, "rope": 64, "value": 256, "latent": 512, "query_lora": 2048, "query_gate": 0, "gate_rank": 0,
                  "experts": 256, "top_k": 8, "expert_inter": 2048, "dense_inter": 12288, "first_routed": 3},
    "axk2": {"name": "ax-k2", "revision": "d0805cfd1fc647b4078c0d9f232c80d34ee0c893", "hidden": 7168, "layers": 61, "vocab": 163840,
             "heads": 64, "nope": 128, "rope": 64, "value": 128, "latent": 512, "query_lora": 1536, "query_gate": 128, "gate_rank": 16,
             "experts": 256, "top_k": 8, "expert_inter": 2048, "dense_inter": 18432, "first_routed": 1},
}
TP = 16
M = MODELS["glm53full"]
PREFIX = "glm53full"
BLOCK = 128
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2


class RankPacks:
    def __init__(self, out_dir, resume, ranks, identity):
        self.out_dir = Path(out_dir)
        self.ranks = ranks
        self.writers = {rank: PayloadWriter(self.payload(rank), identity, resume=resume and self.payload(rank).exists()) for rank in ranks}

    def payload(self, rank):
        return self.out_dir / f"{PREFIX}.tp{TP}.rank{rank:02d}.pack.payload"

    def final(self, rank):
        return self.out_dir / f"{PREFIX}.tp{TP}.rank{rank:02d}.pack"

    def add(self, rank, name, array, kind):
        array = np.ascontiguousarray(array)
        self.writers[rank].add(name, array.tobytes(), kind, array.shape)

    def add_bytes(self, rank, name, blob, kind, shape):
        self.writers[rank].add(name, blob, kind, shape)

    def replicated(self, name, array, kind):
        for rank in self.ranks:
            self.add(rank, name, array, kind)


def window(total, rank):
    count = total // TP
    return rank * count, (rank + 1) * count


def widen(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def norm_f32(source, name):
    return widen(source.spine_bf16(name)).reshape(-1)


def fp8_linear(source, packs, rank, key, parts, columns):
    codes = b"".join(source.expert_payload(name, r0, r1, c0, c1) for name, r0, r1, c0, c1 in parts)
    scales = b"".join(source.expert_scale(name, r0, r1, c0, c1) for name, r0, r1, c0, c1 in parts)
    rows = sum(r1 - r0 for _, r0, r1, _, _ in parts)
    packs.add_bytes(rank, key, codes, "fp8_e4m3", (rows, columns))
    packs.add_bytes(rank, key + ".scale", scales, "f32", (rows, columns // BLOCK))


def pack_norm(source, packs, prefix, name, gated):
    if not gated:
        packs.replicated(name + "_norm", norm_f32(source, prefix + ".weight"), "f32")
        return
    packs.replicated(name + "_norm", norm_f32(source, prefix + ".norm.weight"), "f32")
    packs.replicated(name + "_gate_down", source.spine_bf16(prefix + ".W_down.weight"), "bf16")
    packs.replicated(name + "_gate_up", source.spine_bf16(prefix + ".W_up.weight"), "bf16")


def pack_attention(source, packs, a, name):
    heads, nope, rope, value, latent, query_lora = M["heads"], M["nope"], M["rope"], M["value"], M["latent"], M["query_lora"]
    query_head = nope + rope + M["query_gate"]
    query_input = 2 * query_lora if M["query_gate"] else query_lora
    kv_b = source.spine_bf16(a + "kv_b_proj.weight").reshape(heads, nope + value, latent)
    packs.replicated(name + "mla_q_norm", norm_f32(source, a + "q_a_layernorm.weight"), "f32")
    packs.replicated(name + "mla_kv_norm", norm_f32(source, a + "kv_a_layernorm.weight"), "f32")
    local = heads // TP
    for rank in packs.ranks:
        first, last = rank * local, (rank + 1) * local
        fp8_linear(source, packs, rank, name + "mla_q_a", [(a + "q_a_proj.weight", 0, query_lora, 0, M["hidden"])], M["hidden"])
        fp8_linear(source, packs, rank, name + "mla_kv_a", [(a + "kv_a_proj_with_mqa.weight", 0, latent + rope, 0, M["hidden"])], M["hidden"])
        fp8_linear(source, packs, rank, name + "mla_q_b", [(a + "q_b_proj.weight", first * query_head, last * query_head, 0, query_input)], query_input)
        fp8_linear(source, packs, rank, name + "mla_out", [(a + "o_proj.weight", 0, M["hidden"], first * value, last * value)], local * value)
        packs.add(rank, name + "mla_uk", kv_b[first:last, :nope, :].transpose(0, 2, 1), "bf16")
        packs.add(rank, name + "mla_uv", kv_b[first:last, nope:, :], "bf16")


def pack_gated(source, packs, m, name, gate_up, down, inter):
    for rank in packs.ranks:
        r0, r1 = window(inter, rank)
        fp8_linear(source, packs, rank, name + gate_up, [(m + "gate_proj.weight", r0, r1, 0, M["hidden"]), (m + "up_proj.weight", r0, r1, 0, M["hidden"])],
                   M["hidden"])
        fp8_linear(source, packs, rank, name + down, [(m + "down_proj.weight", 0, M["hidden"], r0, r1)], r1 - r0)


def pack_experts(source, packs, m, name):
    prefix, hidden, experts = f"{m}experts", M["hidden"], M["experts"]
    for rank in packs.ranks:
        r0, r1 = window(M["expert_inter"], rank)
        rows = r1 - r0
        for target, projection in (("moe_w1", "gate"), ("moe_w3", "up")):
            names = [f"{prefix}.{expert}.{projection}_proj.weight" for expert in range(experts)]
            packs.add_bytes(rank, name + target, b"".join(source.expert_payload(n, r0, r1, 0, hidden) for n in names), "fp8", (experts, rows, hidden))
            packs.add_bytes(rank, name + target + "_scale", b"".join(source.expert_scale(n, r0, r1, 0, hidden) for n in names), "f32",
                            (experts, rows, hidden // BLOCK))
        names = [f"{prefix}.{expert}.down_proj.weight" for expert in range(experts)]
        packs.add_bytes(rank, name + "moe_w2", b"".join(source.expert_payload(n, 0, hidden, r0, r1) for n in names), "fp8", (experts, hidden, rows))
        packs.add_bytes(rank, name + "moe_w2_scale", b"".join(source.expert_scale(n, 0, hidden, r0, r1) for n in names), "f32",
                        (experts, hidden, rows // BLOCK))


def pack_layer(source, packs, layer):
    p = f"model.layers.{layer}."
    m, name = p + "mlp.", f"layers.{layer}."
    routed, gated = layer >= M["first_routed"], M["gate_rank"] != 0
    pack_norm(source, packs, p + "input_layernorm", name + "input", gated)
    pack_norm(source, packs, p + "post_attention_layernorm", name + "post", gated and routed)
    pack_attention(source, packs, p + "self_attn.", name)
    if not routed:
        pack_gated(source, packs, m, name, "ffn_gate_up", "ffn_down", M["dense_inter"])
        return
    packs.replicated(name + "router", source.spine_bf16(m + "gate.weight"), "bf16")
    packs.replicated(name + "router_bias", np.asarray(source.raw(m + "gate.e_score_correction_bias")).view(np.float32).copy(), "f32")
    pack_experts(source, packs, m, name)
    pack_gated(source, packs, m + "shared_experts.", name, "shared_gate_up", "shared_down", M["expert_inter"])


def finish(packs, revision, contract_sha256):
    vocab_rows = M["vocab"] // TP
    for rank in packs.ranks:
        packs.writers[rank].close()
        layout = {"model": M["name"], "tp_degree": TP, "tp_rank": rank, "hidden": M["hidden"], "layers": M["layers"], "vocab": M["vocab"],
                  "vocab_first": rank * vocab_rows, "vocab_rows": vocab_rows, "ffn_rows": M["dense_inter"] // TP, "experts": M["experts"],
                  "local_experts": M["experts"], "expert_first": 0, "top_k": M["top_k"], "shared_rows": M["expert_inter"] // TP,
                  "moe_rows": M["expert_inter"] // TP}
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": "fp8_e4m3", "expert_codec": "fp8", "revision": revision, "contract_sha256": contract_sha256}
        final = packs.final(rank)
        assemble(final, packs.payload(rank), NAMED_PACK_MAGIC, NAMED_PACK_VERSION, NAMED_PACK_ALIGN, fmt, layout, packs.writers[rank].manifest)
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
    parser.add_argument("--tp-degree", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--ranks")
    args = parser.parse_args()
    global TP, M, PREFIX
    TP, M, PREFIX = args.tp_degree, MODELS[args.model], args.model
    if args.ranks is None:
        args.ranks = ",".join(str(rank) for rank in range(TP))
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    if (M["expert_inter"] // TP) % BLOCK or (M["dense_inter"] // TP) % BLOCK or (M["heads"] // TP * M["value"]) % BLOCK:
        raise PackFailure(f"TP{TP} splits a projection inside a 128-wide scale block")
    source = Fp8SourceReader(args.checkpoint)
    config = source.config.get("text_config", source.config)
    expect = {"num_hidden_layers": M["layers"], "hidden_size": M["hidden"], "vocab_size": M["vocab"], "q_lora_rank": M["query_lora"],
              "kv_lora_rank": M["latent"], "qk_nope_head_dim": M["nope"], "qk_rope_head_dim": M["rope"], "v_head_dim": M["value"],
              "num_attention_heads": M["heads"], "n_routed_experts": M["experts"], "moe_intermediate_size": M["expert_inter"],
              "intermediate_size": M["dense_inter"], "first_k_dense_replace": M["first_routed"], "n_shared_experts": 1}
    for key, value in expect.items():
        if config.get(key) != value:
            raise PackFailure(f"config {key} is {config.get(key)!r}, the packer pins {value!r}")
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packs = RankPacks(args.output_dir, args.resume, ranks, source_identity(args.checkpoint, PREFIX, files=(args.contract,)))
    embed = source.spine_bf16("model.embed_tokens.weight")
    head = source.spine_bf16("lm_head.weight")
    for rank in ranks:
        r0, r1 = window(M["vocab"], rank)
        packs.add(rank, "embed", embed[r0:r1], "bf16")
        packs.add(rank, "head", head[r0:r1], "bf16")
    packs.replicated("head_norm", norm_f32(source, "model.norm.weight"), "f32")
    for layer in range(M["layers"]):
        pack_layer(source, packs, layer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, M["revision"], contract_sha256)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"glm52_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
