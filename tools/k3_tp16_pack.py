#!/usr/bin/env python3
"""Kimi K3 or K3.1 Pro checkpoint -> TP16 (or --tp-degree N) named stage packs for the device stage.

Output per rank: <out>/k3.tp<N>.rank<NN>.pack, k31.tp<N>.rank<NN>.pack for
K3.1 Pro (tools/spark_named_pack.py
container) plus <pack>.sha256 and the empty-range <pack>.experts. The
checkpoint is read through tools/k3_pack.py's SafetensorDir; the AttnRes
weights are its gamma folds, bf16(gamma x proj). Unlike the CUDA pack, the
query keeps q_b and kv_b's key rows apart (no q_up absorption) and the
routed experts stay in the checkpoint's MXFP4 planes (E2M1 two per byte, low
nibble first, an E8M0 scale per 32 inputs) instead of interleaved cells.

Rank r holds (H hidden 7168, W = 96/TP heads x 128, R = 1/TP of a sharded
dimension):
  embed, head          bf16 vocab rows [r*V/TP, +V/TP); head_norm f32 [H];
                       attnres_out bf16 [H]
  every layer          attn_norm, mlp_norm f32 [H]; attnres_attn,
                       attnres_mlp bf16 [H]
  KDA layers           kda_qkv: q|k|v rows of the rank's heads [3W, H];
                       kda_beta [heads, H]; kda_f_a [128, H] (replicated);
                       kda_f [W, 128]; kda_g [W, H] (full rank); kda_conv f32
                       q|k|v taps [3W, 4]; kda_a_log f32 [heads] (of the
                       first 96); kda_dt_bias f32 [W]; kda_norm f32 [128];
                       kda_out [H, W]
  MLA layers           mla_q_a [1536, H], mla_kv_a [576, H] (replicated);
                       mla_q_norm f32 [1536], mla_kv_norm f32 [512];
                       mla_q_b [heads*192, 1536]; mla_uk [heads, 512, 128];
                       mla_uv [heads, 128, 512]; mla_gate [heads*128, H];
                       mla_out [H, heads*128]
  layer 0              ffn_gate_up: gate then up rows [2*33792/TP, H];
                       ffn_down [H, 33792/TP]
  routed layers        router [896, H], router_bias f32 [896] (replicated);
                       routed_down [3584/TP, H]; routed_norm f32 [3584];
                       routed_up [H, 3584/TP]; all 896 experts split by
                       intermediate width: moe_w1 (gate), moe_w3 (up)
                       mxfp4_e2m1 [896, 3072/TP, 1792] with *_scale e8m0
                       [896, 3072/TP, 112]; moe_w2 [896, 3584, 3072/TP/2]
                       with moe_w2_scale [896, 3584, 3072/TP/32];
                       shared_gate_up: gate then up [2*6144/TP, H];
                       shared_down [H, 6144/TP]
  K3.1 MLA layers      the DSA indexer, replicated and bit-preserved: bf16
                       mla_index_q [32*128, 1536] (wq_b), mla_index_k
                       [128, H] (wk), mla_index_w [32, H] (weights_proj);
                       mla_index_norm and mla_index_norm_bias f32 [128]
                       (k_norm's LayerNorm, the bf16 widened exactly);
                       the layout adds index_topk 2048, index_heads 32 and
                       index_dim 128 and names the model kimi-k3.1-pro
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k3_pack import SafetensorDir, check_scales, gamma_fold_bf16, quant_pair  # noqa: E402
from spark_named_pack import NAMED_PACK_ALIGN, NAMED_PACK_MAGIC, NAMED_PACK_VERSION, PayloadWriter, assemble, source_identity  # noqa: E402
from spark_pack_common import PackFailure  # noqa: E402

TP = 16
PREFIX = "k3"
H, LAYERS, VOCAB = 7168, 93, 163840
HEADS, KEY = 96, 128
Q_LORA, KV_LORA, NOPE, ROPE, VALUE = 1536, 512, 128, 64, 128
EXPERTS, TOP_K, INTER, LATENT, SHARED, DENSE = 896, 16, 3072, 3584, 6144, 33792
GROUP, CONV = 32, 4
INDEX_TOPK, INDEX_HEADS, INDEX_DIM = 2048, 32, 128
DSA = {"index_topk": INDEX_TOPK, "index_num_heads": INDEX_HEADS, "index_head_dim": INDEX_DIM, "index_rope_dim": 0, "index_topk_freq": 1,
       "index_k_norm_type": "layernorm"}
INDEXER = (("mla_index_q", "wq_b.weight", (INDEX_HEADS * INDEX_DIM, Q_LORA)), ("mla_index_k", "wk.weight", (INDEX_DIM, H)),
           ("mla_index_w", "weights_proj.weight", (INDEX_HEADS, H)), ("mla_index_norm", "k_norm.weight", (INDEX_DIM,)),
           ("mla_index_norm_bias", "k_norm.bias", (INDEX_DIM,)))
MODELS = {False: ("kimi-k3", "k3", "f831ab66814297da540d832a5235f8e904f29d06"),
          True: ("kimi-k3.1-pro", "k31", "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e")}
EXPERTS_MAGIC = 0x58504557
EXPERTS_VERSION = 2
S = "language_model.model."


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

    def replicated(self, name, array, kind):
        for rank in self.ranks:
            self.add(rank, name, array, kind)


class Reader:
    def __init__(self, directory):
        self.st = SafetensorDir(directory)
        self.names = self.st.names()

    def u16(self, name, shape):
        return np.frombuffer(self.st.bf16(name), dtype=np.uint16).reshape(shape)

    def u8(self, name, shape):
        return np.frombuffer(self.st.u8(name), dtype=np.uint8).reshape(shape)

    def f32(self, name, shape):
        return np.frombuffer(self.st.f32(name), dtype=np.float32).reshape(shape)

    def gain(self, name, count):
        return (self.u16(name, (count,)).astype(np.uint32) << 16).view(np.float32)

    def fold(self, gamma, proj):
        return np.frombuffer(gamma_fold_bf16(self.st.bf16(proj, (1, H)), self.st.bf16(gamma, (H,)), H), dtype=np.uint16)


def mla_layer(layer):
    return layer % 4 == 3 or layer == LAYERS - 1


def indexed(config):
    dsa = config.get("dsa_config")
    if dsa is None:
        return False
    wrong = {key: dsa.get(key) for key, value in DSA.items() if dsa.get(key) != value}
    if wrong:
        raise PackFailure(f"dsa_config {wrong} is not Kimi K3.1 Pro's indexer {DSA}: the device stage runs only that one")
    return True


def check_indexer_census(names, indexer):
    found = sorted(name for name in names if ".self_attn.indexer." in name)
    expected = sorted(f"{S}layers.{layer}.self_attn.indexer.{source}" for layer in range(LAYERS) if mla_layer(layer) for _, source, _ in INDEXER) \
        if indexer else []
    if found != expected:
        missing, extra = sorted(set(expected) - set(found)), sorted(set(found) - set(expected))
        raise PackFailure(f"the checkpoint's indexer tensors are not one indexer per MLA layer: missing {missing[:4]}, unexpected {extra[:4]}")


def pack_indexer(reader, packs, a, name):
    i = a + "indexer."
    for target, source, shape in INDEXER:
        bits = np.frombuffer(reader.st.bf16(i + source, shape), dtype=np.uint16).reshape(shape)
        if len(shape) == 1:
            packs.replicated(name + target, (bits.astype(np.uint32) << 16).view(np.float32), "f32")
        else:
            packs.replicated(name + target, bits, "bf16")


def window(total, rank):
    count = total // TP
    return slice(rank * count, (rank + 1) * count)


def pack_kda(reader, packs, a, name):
    width = HEADS * KEY
    q, k, v = (reader.u16(a + f"{key}_proj.weight", (width, H)) for key in "qkv")
    convs = [reader.f32(a + f"{key}_conv1d.weight", (width, 1, CONV)).reshape(width, CONV) for key in "qkv"]
    beta = reader.u16(a + "b_proj.weight", (HEADS, H))
    f_b = reader.u16(a + "f_b_proj.weight", (width, KEY))
    gate = reader.u16(a + "g_proj.weight", (width, H))
    a_log = reader.f32(a + "A_log", (128,))[:HEADS]
    dt_bias = reader.f32(a + "dt_bias", (width,))
    out = reader.u16(a + "o_proj.weight", (H, width))
    packs.replicated(name + "kda_f_a", reader.u16(a + "f_a_proj.weight", (KEY, H)), "bf16")
    packs.replicated(name + "kda_norm", reader.f32(a + "o_norm.weight", (KEY,)), "f32")
    for rank in packs.ranks:
        rows, heads = window(width, rank), window(HEADS, rank)
        packs.add(rank, name + "kda_qkv", np.concatenate((q[rows], k[rows], v[rows])), "bf16")
        packs.add(rank, name + "kda_beta", beta[heads], "bf16")
        packs.add(rank, name + "kda_f", f_b[rows], "bf16")
        packs.add(rank, name + "kda_g", gate[rows], "bf16")
        packs.add(rank, name + "kda_conv", np.concatenate([conv[rows] for conv in convs]), "f32")
        packs.add(rank, name + "kda_a_log", a_log[heads], "f32")
        packs.add(rank, name + "kda_dt_bias", dt_bias[rows], "f32")
        packs.add(rank, name + "kda_out", out[:, rows], "bf16")


def pack_mla(reader, packs, a, name):
    q_b = reader.u16(a + "q_b_proj.weight", (HEADS * (NOPE + ROPE), Q_LORA))
    kv_b = reader.u16(a + "kv_b_proj.weight", (HEADS * (NOPE + VALUE), KV_LORA)).reshape(HEADS, NOPE + VALUE, KV_LORA)
    gate = reader.u16(a + "g_proj.weight", (HEADS * VALUE, H))
    out = reader.u16(a + "o_proj.weight", (H, HEADS * VALUE))
    packs.replicated(name + "mla_q_a", reader.u16(a + "q_a_proj.weight", (Q_LORA, H)), "bf16")
    packs.replicated(name + "mla_q_norm", reader.gain(a + "q_a_layernorm.weight", Q_LORA), "f32")
    packs.replicated(name + "mla_kv_a", reader.u16(a + "kv_a_proj_with_mqa.weight", (KV_LORA + ROPE, H)), "bf16")
    packs.replicated(name + "mla_kv_norm", reader.gain(a + "kv_a_layernorm.weight", KV_LORA), "f32")
    for rank in packs.ranks:
        heads = window(HEADS, rank)
        packs.add(rank, name + "mla_q_b", q_b[window(HEADS * (NOPE + ROPE), rank)], "bf16")
        packs.add(rank, name + "mla_uk", kv_b[heads, :NOPE, :].transpose(0, 2, 1), "bf16")
        packs.add(rank, name + "mla_uv", kv_b[heads, NOPE:, :], "bf16")
        packs.add(rank, name + "mla_gate", gate[window(HEADS * VALUE, rank)], "bf16")
        packs.add(rank, name + "mla_out", out[:, window(HEADS * VALUE, rank)], "bf16")


def pack_gated(reader, packs, m, name, gate_up, down, inter):
    gate = reader.u16(m + "gate_proj.weight", (inter, H))
    up = reader.u16(m + "up_proj.weight", (inter, H))
    projection = reader.u16(m + "down_proj.weight", (H, inter))
    for rank in packs.ranks:
        rows = window(inter, rank)
        packs.add(rank, name + gate_up, np.concatenate((gate[rows], up[rows])), "bf16")
        packs.add(rank, name + down, projection[:, rows], "bf16")


def pack_experts(reader, packs, m, name):
    planes = {rank: {key: [] for key in ("w1", "w1s", "w3", "w3s", "w2", "w2s")} for rank in packs.ranks}
    for expert in range(EXPERTS):
        base = f"{m}experts.{expert}."
        loaded = {}
        for key, shape in (("w1", (INTER, LATENT)), ("w3", (INTER, LATENT)), ("w2", (LATENT, INTER))):
            values, scales = quant_pair(reader.st, base + key)
            loaded[key] = reader.u8(values, (shape[0], shape[1] // 2))
            raw = reader.st.u8(scales, (shape[0], shape[1] // GROUP))
            check_scales(scales, raw)
            loaded[key + "s"] = np.frombuffer(raw, dtype=np.uint8).reshape(shape[0], shape[1] // GROUP)
        for rank in packs.ranks:
            rows = window(INTER, rank)
            planes[rank]["w1"].append(loaded["w1"][rows])
            planes[rank]["w1s"].append(loaded["w1s"][rows])
            planes[rank]["w3"].append(loaded["w3"][rows])
            planes[rank]["w3s"].append(loaded["w3s"][rows])
            planes[rank]["w2"].append(loaded["w2"][:, window(INTER // 2, rank)])
            planes[rank]["w2s"].append(loaded["w2s"][:, window(INTER // GROUP, rank)])
    for rank in packs.ranks:
        for target, key in (("moe_w1", "w1"), ("moe_w3", "w3"), ("moe_w2", "w2")):
            packs.add(rank, name + target, np.stack(planes[rank][key]), "mxfp4_e2m1")
            packs.add(rank, name + target + "_scale", np.stack(planes[rank][key + "s"]), "e8m0")


def pack_layer(reader, packs, layer, indexer):
    p = f"{S}layers.{layer}."
    name = f"layers.{layer}."
    a = p + "self_attn."
    packs.replicated(name + "attn_norm", reader.gain(p + "input_layernorm.weight", H), "f32")
    packs.replicated(name + "mlp_norm", reader.gain(p + "post_attention_layernorm.weight", H), "f32")
    packs.replicated(name + "attnres_attn", reader.fold(p + "self_attention_res_norm.weight", p + "self_attention_res_proj.weight"), "bf16")
    packs.replicated(name + "attnres_mlp", reader.fold(p + "mlp_res_norm.weight", p + "mlp_res_proj.weight"), "bf16")
    if not mla_layer(layer):
        pack_kda(reader, packs, a, name)
    else:
        pack_mla(reader, packs, a, name)
        if indexer:
            pack_indexer(reader, packs, a, name)
    m = p + "block_sparse_moe."
    if m + "gate.weight" not in reader.names:
        pack_gated(reader, packs, p + "mlp.", name, "ffn_gate_up", "ffn_down", DENSE)
        return
    packs.replicated(name + "router", reader.u16(m + "gate.weight", (EXPERTS, H)), "bf16")
    dtype, _, raw = reader.st.tensor(m + "gate.e_score_correction_bias")
    bias = np.frombuffer(raw, dtype=np.float32) if dtype == "F32" else (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    packs.replicated(name + "router_bias", bias.astype(np.float32), "f32")
    down = reader.u16(m + "routed_expert_down_proj.weight", (LATENT, H))
    up = reader.u16(m + "routed_expert_up_proj.weight", (H, LATENT))
    packs.replicated(name + "routed_norm", reader.gain(m + "routed_expert_norm.weight", LATENT), "f32")
    for rank in packs.ranks:
        packs.add(rank, name + "routed_down", down[window(LATENT, rank)], "bf16")
        packs.add(rank, name + "routed_up", up[:, window(LATENT, rank)], "bf16")
    pack_experts(reader, packs, m, name)
    pack_gated(reader, packs, m + "shared_experts.", name, "shared_gate_up", "shared_down", SHARED)


def finish(packs, revision, contract_sha256, indexer):
    vocab_rows = VOCAB // TP
    for rank in packs.ranks:
        packs.writers[rank].close()
        layout = {"model": MODELS[indexer][0], "tp_degree": TP, "tp_rank": rank, "hidden": H, "layers": LAYERS, "vocab": VOCAB, "vocab_first": rank * vocab_rows,
                  "vocab_rows": vocab_rows, "ffn_rows": DENSE // TP, "experts": EXPERTS, "local_experts": EXPERTS, "expert_first": 0, "top_k": TOP_K,
                  "shared_rows": SHARED // TP, "moe_rows": INTER // TP, "latent_rows": LATENT // TP}
        if indexer:
            layout.update({"index_topk": INDEX_TOPK, "index_heads": INDEX_HEADS, "index_dim": INDEX_DIM})
        fmt = {"container": "spark_named_pack", "version": NAMED_PACK_VERSION, "alignment": NAMED_PACK_ALIGN,
               "linear_codec": "bf16", "expert_codec": "mxfp4", "revision": revision, "contract_sha256": contract_sha256}
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tp-degree", type=int, choices=(1, 2, 4, 8, 16), default=16)
    parser.add_argument("--ranks")
    args = parser.parse_args()
    global TP, PREFIX
    TP = args.tp_degree
    if args.ranks is None:
        args.ranks = ",".join(str(rank) for rank in range(TP))
    ranks = sorted({int(rank) for rank in args.ranks.split(",")})
    if not ranks or ranks[0] < 0 or ranks[-1] >= TP:
        raise PackFailure(f"--ranks must name ranks in 0..{TP - 1}")
    if (INTER // TP) % GROUP or (LATENT // TP) % 8:
        raise PackFailure(f"TP{TP} splits an expert inside an MXFP4 scale group")
    config = json.loads((args.checkpoint / "config.json").read_text())
    config = config.get("text_config", config)
    if config.get("num_hidden_layers") != LAYERS or config.get("hidden_size") != H or config.get("vocab_size") != VOCAB:
        raise PackFailure("the checkpoint is not the Kimi K3 geometry this packer pins")
    indexer = indexed(config)
    _, PREFIX, revision = MODELS[indexer]
    contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = Reader(args.checkpoint)
    check_indexer_census(reader.names, indexer)
    packs = RankPacks(args.output_dir, args.resume, ranks, source_identity(args.checkpoint, PREFIX, files=(args.contract,)))
    embed = reader.u16(S + "embed_tokens.weight", (VOCAB, H))
    head = reader.u16("language_model.lm_head.weight", (VOCAB, H))
    for rank in ranks:
        rows = window(VOCAB, rank)
        packs.add(rank, "embed", embed[rows], "bf16")
        packs.add(rank, "head", head[rows], "bf16")
    packs.replicated("head_norm", reader.gain(S + "norm.weight", H), "f32")
    packs.replicated("attnres_out", reader.fold(S + "output_attn_res_norm.weight", S + "output_attn_res_proj.weight"), "bf16")
    for layer in range(LAYERS):
        pack_layer(reader, packs, layer, indexer)
        print(f"layer {layer} packed", flush=True)
    finish(packs, args.revision or revision, contract_sha256, indexer)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackFailure as error:
        print(f"k3_tp16_pack FAIL: {error}", file=sys.stderr)
        sys.exit(1)
