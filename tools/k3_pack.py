#!/usr/bin/env python3
"""Pack a Kimi K3 checkpoint into the sparkpipe resident layout, format V2.

The engines serve the checkpoint as shipped: MXFP4 expert payloads and their
E8M0 scales are moved, never recomputed, and everything on the ignore list
stays BF16. What the packer OWES beyond moving bytes is what the layer's
weight table cannot get from any checkpoint directly. V2 adds three layout
transforms, each removing a defect the V1 format baked in:

  fused KDA     the six projections that read one normed KDA input shipped as
  projections   six tensors, so six GEMM launches read the same activation
                (layer.cuh K3-PERF-003). One GEMM needs one base pointer, and
                the TP shard table splits the six into two classes, so V2
                emits TWO fused tensors per KDA layer, each carrying exactly
                one shard class:
                  kda_qkv_beta_weight       q|k|v|beta rows, one
                                            [sum_out, hidden] BF16 tensor,
                                            OUTPUT_DIM_HEADS
                  kda_decay_gate_down_weight decay_down|gate_down rows,
                                            REPLICATED
                The manifest's per-tensor section table gives each section's
                row offset and rows-per-head, so bind is pointer arithmetic.

  interleaved   V1 shipped expert payload and E8M0 scales as two planes two
  weight+scale  pointers apart; the GEMM staged the payload by TMA and the
                scales by LDG from the far plane. V2 blocks each expert into
                k-tiles of 128 elements and, per k-tile, interleaves each
                16-neuron cell's sixteen 64-byte payload rows with ONE 64-byte
                scale row (16 neurons x 4 group scales - exactly full). The
                ratio is exact, so the tensor is zero padding: payload+scales
                = cells * k_tiles * 17 * 64 bytes per expert. One TMA box of
                17 * (TILE_N/16) rows then fetches a stage's payload AND its
                scales in a single transaction off one descriptor, the 64-byte
                swizzle span intact. The addressing math is
                interleave_geometry / interleave_byte_offset below and the
                consumption contract is docs/K3_PACK_FORMAT_V2.md.

  alignment     every tensor starts 128-byte aligned (TMA base and L2 line),
                tensors are emitted layer by layer in consumption order so a
                sequential prefetcher never re-seeks, and the manifest header
                block carries per-tensor kind/shape/sections/interleave so a
                loader binds with adds, not parsing.

  q-fold        kv_b's k_nope half absorbed into q_b, per head:
                A[h] = kv_b_k[h]^T @ q_b_nope[h], then the unrotated rows
                byte for byte - heads * (kv_lora + rope) rows over q_lora
  kv_b split    the value half of kv_b, byte for byte, as its own
                [heads * v_head, kv_lora] per-head table (the gate lives in
                v-space and does not commute, so o_proj stays as shipped)
  gamma folds   every attention-residual score projection multiplied
                elementwise by its RMSNorm gamma - exact, the kernel norms
                without a weight
  conv flatten  [channels, 1, kernel] to [channels, kernel] (a shape
                assertion; the row-major bytes are identical)

NUMPY IS OPTIONAL. Every layout transform (fusion, interleave, alignment,
manifest) is pure stdlib and bit-exact without numpy; numpy, when importable,
only accelerates the two heavy paths (the q-fold matmul and the expert
interleave relay). The pure-python q-fold fallback emulates float32 per
operation - bit-exact f32 semantics, but the accumulation ORDER is
sequential where numpy's may be pairwise, so the fallback can differ in the
last ulp and is for host verification, not for shipping a pack. It is also
O(n^3) in python: fine for the test mini, days for the real checkpoint.

Failure is loud: a missing tensor, an E8M0 0xff, a group size that is not 32,
an expert count that is not the config's, a dimension the interleave grid
does not divide - each is a PackFailure naming what and where.
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_named_pack import PayloadWriter, assemble, source_identity  # noqa: E402
from spark_pack_codecs import (bf16_bytes_to_floats, bf16_to_f32, check_e8m0_scales as check_scales,  # noqa: E402
                               f32_round, f32_to_bf16, floats_to_bf16_bytes, np)
from spark_pack_common import INDEX_NAME, PackFailure, SafetensorsSource  # noqa: E402

MAGIC = 0x4B33504B
VERSION = 2
ALIGN = 128
E8M0_NAN = 0xFF
GROUP = 32
TILE_K = 128
CELL_PAYLOAD_ROWS = 16
A_LOG_SOURCE_HEADS = 128

KIND_BF16 = "bf16"
KIND_F32 = "f32"
KIND_MXFP4_INTERLEAVED = "mxfp4_ws_interleaved_v1"



def kda_fused_qkvb_sections(heads, key_dim, value_dim):
    sections = []
    row = 0
    for name, rows, per_head in (
            ("q", heads * key_dim, key_dim),
            ("k", heads * key_dim, key_dim),
            ("v", heads * value_dim, value_dim),
            ("beta", heads, 1)):
        sections.append({"name": name, "row_offset": row, "rows": rows,
                         "rows_per_head": per_head})
        row += rows
    return sections, row



def interleave_geometry(out_dim, k_dim, experts=1, tile_k=TILE_K, group=GROUP,
                        stored_bits=4):
    tile_payload = tile_k * stored_bits // 8
    tile_scale = tile_k // group
    if tile_payload != CELL_PAYLOAD_ROWS * tile_scale:
        raise PackFailure(
            f"interleave cell does not close: {CELL_PAYLOAD_ROWS} payload "
            f"rows of {tile_payload}B need a {CELL_PAYLOAD_ROWS * tile_scale}B "
            f"scale row, not {tile_payload}B")
    if k_dim % tile_k != 0:
        raise PackFailure(f"K {k_dim} is not a whole number of {tile_k}-element "
                          f"interleave tiles")
    if k_dim % group != 0:
        raise PackFailure(f"K {k_dim} is not whole MXFP4 groups of {group}")
    if out_dim % CELL_PAYLOAD_ROWS != 0:
        raise PackFailure(f"output {out_dim} is not a whole number of "
                          f"{CELL_PAYLOAD_ROWS}-neuron interleave cells")
    k_tiles = k_dim // tile_k
    cells = out_dim // CELL_PAYLOAD_ROWS
    cell_rows = CELL_PAYLOAD_ROWS + 1
    rows_per_expert = k_tiles * cells * cell_rows
    expert_bytes = rows_per_expert * tile_payload
    payload_bytes = out_dim * (k_dim * stored_bits // 8)
    scale_bytes = out_dim * (k_dim // group)
    if expert_bytes != payload_bytes + scale_bytes:
        raise PackFailure("interleave padding is not zero; the grid is wrong")
    return {"kind": KIND_MXFP4_INTERLEAVED, "out_dim": out_dim,
            "k_dim": k_dim, "experts": experts, "tile_k": tile_k,
            "group": group, "stored_bits": stored_bits,
            "row_bytes": tile_payload,
            "scale_bytes_per_neuron_tile": tile_scale,
            "cell_payload_rows": CELL_PAYLOAD_ROWS, "cell_rows": cell_rows,
            "cells": cells, "k_tiles": k_tiles,
            "rows_per_expert": rows_per_expert,
            "expert_bytes": expert_bytes,
            "tensor_bytes": expert_bytes * experts}


def interleave_byte_offset(geom, expert, k_tile, neuron, kind, lane=0):
    cell, sub = divmod(neuron, geom["cell_payload_rows"])
    row = (expert * geom["rows_per_expert"]
           + (k_tile * geom["cells"] + cell) * geom["cell_rows"])
    if kind == "payload":
        return (row + sub) * geom["row_bytes"] + lane
    if kind == "scale":
        return (row + geom["cell_payload_rows"]) * geom["row_bytes"] \
            + sub * geom["scale_bytes_per_neuron_tile"] + lane
    raise PackFailure(f"unknown interleave lane kind {kind!r}")


def interleave_py(payload, scales, geom):
    experts = geom["experts"]
    row_payload = geom["k_dim"] * geom["stored_bits"] // 8
    k_groups = geom["k_dim"] // geom["group"]
    per_expert_payload = geom["out_dim"] * row_payload
    per_expert_scale = geom["out_dim"] * k_groups
    if len(payload) != experts * per_expert_payload or \
            len(scales) != experts * per_expert_scale:
        raise PackFailure("interleave source plane size does not match geometry")
    out = bytearray(geom["tensor_bytes"])
    for e in range(experts):
        pay_base = e * per_expert_payload
        sc_base = e * per_expert_scale
        for t in range(geom["k_tiles"]):
            for c in range(geom["cells"]):
                dst = (e * geom["rows_per_expert"]
                       + (t * geom["cells"] + c) * geom["cell_rows"]) \
                    * geom["row_bytes"]
                for r in range(geom["cell_payload_rows"]):
                    n = c * geom["cell_payload_rows"] + r
                    src = pay_base + n * row_payload + t * geom["row_bytes"]
                    out[dst + r * geom["row_bytes"]:
                        dst + (r + 1) * geom["row_bytes"]] = \
                        payload[src:src + geom["row_bytes"]]
                srow = dst + geom["cell_payload_rows"] * geom["row_bytes"]
                for r in range(geom["cell_payload_rows"]):
                    n = c * geom["cell_payload_rows"] + r
                    src = sc_base + n * k_groups \
                        + t * geom["scale_bytes_per_neuron_tile"]
                    width = geom["scale_bytes_per_neuron_tile"]
                    out[srow + r * width:srow + (r + 1) * width] = \
                        scales[src:src + width]
    return bytes(out)


def interleave_np(payload, scales, geom):
    row_bytes = geom["row_bytes"]
    width = geom["scale_bytes_per_neuron_tile"]
    p = payload.reshape(geom["experts"], geom["out_dim"], geom["k_tiles"],
                        row_bytes).transpose(0, 2, 1, 3) \
        .reshape(geom["experts"], geom["k_tiles"], geom["cells"],
                 geom["cell_payload_rows"], row_bytes)
    s = scales.reshape(geom["experts"], geom["out_dim"], geom["k_tiles"],
                       width).transpose(0, 2, 1, 3) \
        .reshape(geom["experts"], geom["k_tiles"], geom["cells"], 1,
                 geom["cell_payload_rows"] * width)
    return np.concatenate([p, s], axis=3).tobytes()


def interleave(payload, scales, geom):
    if np is not None:
        p = np.frombuffer(payload, dtype=np.uint8) \
            .reshape(geom["experts"], geom["out_dim"], -1)
        s = np.frombuffer(scales, dtype=np.uint8) \
            .reshape(geom["experts"], geom["out_dim"], -1)
        return interleave_np(p, s, geom)
    return interleave_py(payload, scales, geom)



class SafetensorDir(SafetensorsSource):

    def __init__(self, model_dir):
        root = Path(model_dir)
        if not (root / INDEX_NAME).is_file() and not (root / "model.safetensors").is_file():
            raise PackFailure(f"no safetensors index or file in {model_dir}")
        super().__init__(root, config_required=False, single_name="model.safetensors")

    def names(self):
        return set(self.weight_map)

    def tensor(self, name):
        if name not in self.weight_map:
            raise PackFailure(f"missing tensor: {name}")
        shard, entry, offset = self.resolve(name)
        begin, end = entry["data_offsets"]
        with open(self.root / shard, "rb") as handle:
            handle.seek(offset)
            raw = handle.read(end - begin)
        return entry["dtype"], tuple(entry["shape"]), raw

    def expect(self, name, dtype, shape=None):
        got_dtype, got_shape, raw = self.tensor(name)
        if got_dtype != dtype:
            raise PackFailure(f"{name}: expected {dtype}, checkpoint says "
                              f"{got_dtype}")
        if shape is not None and got_shape != tuple(shape):
            raise PackFailure(f"{name}: shape {got_shape}, expected "
                              f"{tuple(shape)}")
        return raw

    def bf16(self, name, shape=None):
        return self.expect(name, "BF16", shape)

    def u8(self, name, shape=None):
        return self.expect(name, "U8", shape)

    def f32(self, name, shape=None):
        return self.expect(name, "F32", shape)


def quant_pair(reader, base):
    names = reader.names()
    if base + ".weight_packed" in names:
        return base + ".weight_packed", base + ".weight_scale"
    if base + ".weight" in names and base + ".weight_scale" in names:
        return base + ".weight", base + ".weight_scale"
    raise PackFailure(f"{base}: no recognised quantised serialisation "
                      f"(.weight_packed or .weight + .weight_scale)")


def gamma_fold_bf16(proj_raw, gamma_raw, count):
    proj = bf16_bytes_to_floats(proj_raw)
    gamma = bf16_bytes_to_floats(gamma_raw)
    if len(proj) != count or len(gamma) != count:
        raise PackFailure("gamma fold operand size does not match hidden")
    return floats_to_bf16_bytes(
        [f32_round(proj[i] * gamma[i]) for i in range(count)])


def q_fold_absorb(q_b_raw, kv_b_raw, heads, nope, rope, v_head, kv_lora,
                  q_lora):
    q_shape = (heads, nope + rope, q_lora)
    kv_shape = (heads, nope + v_head, kv_lora)
    if np is not None:
        q_b = bf16_to_f32(np.frombuffer(q_b_raw, dtype=np.uint16)
                          .reshape(q_shape))
        kv_b = bf16_to_f32(np.frombuffer(kv_b_raw, dtype=np.uint16)
                           .reshape(kv_shape))
        absorbed = np.einsum("hnl,hnq->hlq", kv_b[:, :nope, :], q_b[:, :nope, :])
        q_b_bits = np.frombuffer(q_b_raw, dtype=np.uint16).reshape(q_shape)
        kv_b_bits = np.frombuffer(kv_b_raw, dtype=np.uint16).reshape(kv_shape)
        folded = np.concatenate([f32_to_bf16(absorbed), q_b_bits[:, nope:, :]],
                                axis=1)
        return folded.tobytes(), kv_b_bits[:, nope:, :].tobytes()
    print("ADVISORY numpy not importable: the MLA q-fold is running the "
          "pure-python reference (f32-exact, sequential accumulation; "
          "verification-grade, not ship-grade speed)")
    q_b = bf16_bytes_to_floats(q_b_raw)
    kv_b = bf16_bytes_to_floats(kv_b_raw)
    q_up = bytearray()
    value = bytearray()
    for h in range(heads):
        q_base = h * (nope + rope) * q_lora
        kv_base = h * (nope + v_head) * kv_lora
        absorbed = []
        for l in range(kv_lora):
            for q in range(q_lora):
                acc = 0.0
                for n in range(nope):
                    acc = f32_round(acc + f32_round(
                        kv_b[kv_base + n * kv_lora + l]
                        * q_b[q_base + n * q_lora + q]))
                absorbed.append(acc)
        q_up += floats_to_bf16_bytes(absorbed)
        q_up += q_b_raw[2 * (q_base + nope * q_lora):
                        2 * (q_base + (nope + rope) * q_lora)]
        value += kv_b_raw[2 * (kv_base + nope * kv_lora):
                          2 * (kv_base + (nope + v_head) * kv_lora)]
    return bytes(q_up), bytes(value)



def validate_layout(manifest, config):
    hidden = config["hidden"]
    offset = 0
    for name in sorted(manifest, key=lambda n: manifest[n]["offset"]):
        entry = manifest[name]
        if entry["offset"] % ALIGN != 0:
            raise PackFailure(f"{name}: offset {entry['offset']} is not "
                              f"{ALIGN}-byte aligned")
        if entry["offset"] < offset:
            raise PackFailure(f"{name}: overlaps the previous tensor")
        offset = entry["offset"] + entry["bytes"]
        kind = entry["kind"]
        if kind == KIND_MXFP4_INTERLEAVED:
            geom = interleave_geometry(entry["interleave"]["out_dim"],
                                       entry["interleave"]["k_dim"],
                                       entry["interleave"]["experts"])
            if geom["tensor_bytes"] != entry["bytes"]:
                raise PackFailure(f"{name}: {entry['bytes']} bytes, geometry "
                                  f"prices {geom['tensor_bytes']}")
        elif kind == KIND_BF16 and entry["bytes"] % 2 != 0:
            raise PackFailure(f"{name}: odd BF16 byte count")
        if "sections" in entry:
            rows = 0
            for section in entry["sections"]:
                if section["row_offset"] != rows:
                    raise PackFailure(f"{name}: section {section['name']} does "
                                      f"not tile the fused rows")
                rows += section["rows"]
            if rows != entry["shape"][0]:
                raise PackFailure(f"{name}: sections cover {rows} rows, shape "
                                  f"says {entry['shape'][0]}")
            if entry["shape"][1] != hidden:
                raise PackFailure(f"{name}: fused projection is not over "
                                  f"hidden")
            if (hidden * 2) % ALIGN != 0:
                raise PackFailure(f"{name}: a hidden row is {hidden * 2}B, so "
                                  f"section bases cannot stay {ALIGN}B-aligned")


def pack_model(model_dir, out_path, first_layer=0, layer_count=None,
                expert_tile_k=TILE_K):
    reader = SafetensorDir(model_dir)
    config = json.loads((Path(model_dir) / "config.json").read_text())
    if isinstance(config.get("text_config"), dict):
        config = config["text_config"]
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    if layer_count is None:
        layer_count = layers
    if first_layer + layer_count > layers or first_layer < 0 or layer_count <= 0:
        raise PackFailure(f"invalid slice {first_layer}+{layer_count} of {layers}")
    experts = config["num_experts"]
    top_k = config.get("num_experts_per_tok")
    if top_k is None:
        top_k = config.get("num_experts_per_token")
    if top_k is None:
        raise PackFailure("config carries neither num_experts_per_tok nor "
                          "num_experts_per_token")
    latent = config["routed_expert_hidden_size"]
    inter = config["moe_intermediate_size"]
    shared = config.get("num_shared_experts", 1) * inter
    q_lora = config["q_lora_rank"]
    kv_lora = config["kv_lora_rank"]
    rope = config["qk_rope_head_dim"]
    nope = config["qk_nope_head_dim"]
    v_head = config["v_head_dim"]
    heads = config["num_attention_heads"]
    dsa = config.get("dsa_config")
    if dsa is not None:
        if dsa.get("index_topk_freq") != 1 or dsa.get("index_rope_dim") != 0 or dsa.get("index_k_norm_type") != "layernorm":
            raise PackFailure(f"dsa_config {dsa}: only the per-layer NoPE layer-normed indexer (index_topk_freq 1, index_rope_dim 0, "
                              f"index_k_norm_type layernorm) is packed")
        index_heads, index_dim = dsa["index_num_heads"], dsa["index_head_dim"]
    kda_heads = config["linear_attn_config"]["num_heads"] \
        if "linear_attn_config" in config else config["num_attention_heads"]
    kda_head = config["linear_attn_config"]["head_dim"] \
        if "linear_attn_config" in config else config["head_dim"]
    kda_dim = kda_heads * kda_head
    kernel = config["linear_attn_config"]["short_conv_kernel_size"] \
        if "linear_attn_config" in config else 4
    if "layer_types" in config:
        types = config["layer_types"]
    else:
        lac = config.get("linear_attn_config", {})
        types = ["full_attention"] * layers
        for i in lac.get("kda_layers", []):
            if 1 <= i <= layers:
                types[i - 1] = "linear_attention"
    if len(types) != layers:
        raise PackFailure("layer_types does not cover num_hidden_layers")
    w1_geom = interleave_geometry(2 * inter, latent, experts,
                                  tile_k=expert_tile_k)
    w2_geom = interleave_geometry(latent, inter, experts,
                                  tile_k=expert_tile_k)

    payload_path = Path(str(out_path) + ".payload")
    pack = PayloadWriter(payload_path, source_identity(model_dir, first_layer, layer_count), resume=payload_path.exists(), align=ALIGN)
    L = "model.layers.{}."
    SL = "language_model.model.layers.{}."

    def bf(dst, src, shape=None):
        if dst in pack.manifest:
            return
        pack.add(dst, reader.bf16(src, shape), KIND_BF16,
                 shape if shape is not None else [0])

    if first_layer == 0:
        bf("model.embed_tokens.weight", "language_model.model.embed_tokens.weight",
           (config["vocab_size"], hidden))

    for layer in range(first_layer, first_layer + layer_count):
        p = L.format(layer)
        sp = SL.format(layer)
        linear = types[layer] == "linear_attention"
        bf(p + "attn_norm_weight", sp + "input_layernorm.weight", (hidden,))
        g = reader.bf16(sp + "self_attention_res_norm.weight", (hidden,))
        w = reader.bf16(sp + "self_attention_res_proj.weight", (1, hidden))
        pack.add(p + "attnres_attn_weight", gamma_fold_bf16(w, g, hidden),
                 KIND_BF16, [1, hidden])
        if linear:
            a = sp + "self_attn."
            if p + "kda_qkv_beta_weight" in pack.manifest:
                fused = b""
                sections, rows = [], 0
            else:
                sections, rows = kda_fused_qkvb_sections(kda_heads, kda_head,
                                                         kda_head)
                fused = b"".join((
                reader.bf16(a + "q_proj.weight", (kda_dim, hidden)),
                reader.bf16(a + "k_proj.weight", (kda_dim, hidden)),
                reader.bf16(a + "v_proj.weight", (kda_dim, hidden)),
                reader.bf16(a + "b_proj.weight", (kda_heads, hidden))))
            pack.add(p + "kda_qkv_beta_weight", fused, KIND_BF16,
                     [rows, hidden], {"sections": sections,
                                      "shard_class": "output_dim_heads"})
            for conv in "qkv":
                raw = reader.f32(a + f"{conv}_conv1d.weight",
                                 (kda_dim, 1, kernel))
                pack.add(p + f"kda_{conv}_conv_weight", raw, KIND_F32,
                         [kda_dim, kernel])
            pack.add(p + "kda_decay_down_weight",
                     reader.bf16(a + "f_a_proj.weight", (kda_head, hidden)),
                     KIND_BF16, [kda_head, hidden],
                     {"shard_class": "replicated"})
            bf(p + "kda_decay_up_weight", a + "f_b_proj.weight",
               (kda_dim, kda_head))
            pack.add(p + "kda_decay_bias", reader.f32(a + "dt_bias", (kda_dim,)),
                     KIND_F32, [kda_dim])
            head_log_scale = reader.f32(a + "A_log", (A_LOG_SOURCE_HEADS,))
            pack.add(p + "kda_head_log_scale", head_log_scale[:kda_heads * 4],
                     KIND_F32, [kda_heads])
            pack.add(p + "kda_gate_weight",
                     reader.bf16(a + "g_proj.weight", (kda_dim, hidden)),
                     KIND_BF16, [kda_dim, hidden],
                     {"shard_class": "output_dim_heads"})
            pack.add(p + "kda_out_norm_weight",
                     reader.f32(a + "o_norm.weight", (kda_head,)),
                     KIND_F32, [kda_head])
            bf(p + "kda_out_weight", a + "o_proj.weight", (hidden, kda_dim))
        else:
            a = sp + "self_attn."
            bf(p + "mla_q_down_weight", a + "q_a_proj.weight", (q_lora, hidden))
            bf(p + "mla_q_norm_weight", a + "q_a_layernorm.weight", (q_lora,))
            bf(p + "mla_kv_a_weight", a + "kv_a_proj_with_mqa.weight",
               (kv_lora + rope, hidden))
            bf(p + "mla_kv_a_norm_weight", a + "kv_a_layernorm.weight", (kv_lora,))
            q_up, value = q_fold_absorb(
                reader.bf16(a + "q_b_proj.weight",
                            (heads * (nope + rope), q_lora)),
                reader.bf16(a + "kv_b_proj.weight",
                            (heads * (nope + v_head), kv_lora)),
                heads, nope, rope, v_head, kv_lora, q_lora)
            pack.add(p + "mla_q_up_weight", q_up, KIND_BF16,
                     [heads * (kv_lora + rope), q_lora])
            pack.add(p + "mla_kv_b_value_weight", value, KIND_BF16,
                     [heads * v_head, kv_lora])
            pack.add(p + "mla_gate_weight",
                     reader.bf16(a + "g_proj.weight", (heads * v_head, hidden)),
                     KIND_BF16, [heads * v_head, hidden],
                     {"shard_class": "output_dim_heads"})
            bf(p + "mla_out_weight", a + "o_proj.weight", (hidden, heads * v_head))
            if dsa is not None:
                i = a + "indexer."
                bf(p + "mla_index_query_weight", i + "wq_b.weight", (index_heads * index_dim, q_lora))
                bf(p + "mla_index_key_weight", i + "wk.weight", (index_dim, hidden))
                bf(p + "mla_index_head_weight", i + "weights_proj.weight", (index_heads, hidden))
                bf(p + "mla_index_norm_weight", i + "k_norm.weight", (index_dim,))
                bf(p + "mla_index_norm_bias", i + "k_norm.bias", (index_dim,))
        bf(p + "mlp_norm_weight", sp + "post_attention_layernorm.weight", (hidden,))
        g = reader.bf16(sp + "mlp_res_norm.weight", (hidden,))
        w = reader.bf16(sp + "mlp_res_proj.weight", (1, hidden))
        pack.add(p + "attnres_mlp_weight", gamma_fold_bf16(w, g, hidden),
                 KIND_BF16, [1, hidden])
        dense_m = sp + "mlp."
        m = sp + "block_sparse_moe."
        if m + "gate.weight" not in reader.names():
            w1 = reader.bf16(dense_m + "gate_proj.weight")
            w3 = reader.bf16(dense_m + "up_proj.weight")
            dense_inter = len(w1) // (hidden * 2)
            pack.add(p + "dense_gate_up_weight", w1 + w3, KIND_BF16,
                     [2 * dense_inter, hidden])
            down = reader.bf16(dense_m + "down_proj.weight")
            pack.add(p + "dense_down_weight", down, KIND_BF16,
                     [hidden, len(down) // (hidden * 2)])
            continue
        bf(p + "router_weight", m + "gate.weight", (experts, hidden))
        dtype, shape, raw = reader.tensor(m + "gate.e_score_correction_bias")
        if dtype != "F32" and dtype != "BF16":
            raise PackFailure(f"{m}gate.e_score_correction_bias: unexpected "
                              f"{dtype}")
        pack.add(p + "router_bias", raw,
                 KIND_F32 if dtype == "F32" else KIND_BF16, [experts])
        w1_pay, w1_sc, w2_pay, w2_sc = [], [], [], []
        experts_done = p + "expert_w1_weight" in pack.manifest
        for e in range(0 if experts_done else experts):
            base = m + f"experts.{e}."
            g_name, g_scale = quant_pair(reader, base + "w1")
            u_name, u_scale = quant_pair(reader, base + "w3")
            d_name, d_scale = quant_pair(reader, base + "w2")
            g = reader.u8(g_name, (inter, latent // 2))
            u = reader.u8(u_name, (inter, latent // 2))
            d = reader.u8(d_name, (latent, inter // 2))
            gs = reader.u8(g_scale, (inter, latent // GROUP))
            us = reader.u8(u_scale, (inter, latent // GROUP))
            ds = reader.u8(d_scale, (latent, inter // GROUP))
            for name, sc in ((g_scale, gs), (u_scale, us), (d_scale, ds)):
                check_scales(name, sc)
            e1_geom = interleave_geometry(2 * inter, latent, experts=1,
                                          tile_k=w1_geom["tile_k"])
            w1_pay.append(interleave(g + u, gs + us, e1_geom))
            e2_geom = interleave_geometry(latent, inter, experts=1,
                                          tile_k=w2_geom["tile_k"])
            w2_pay.append(interleave(d, ds, e2_geom))
        if p + "expert_w1_weight" not in pack.manifest:
            pack.add(p + "expert_w1_weight", b"".join(w1_pay),
                     KIND_MXFP4_INTERLEAVED, [experts, 2 * inter, latent],
                     {"interleave": w1_geom, "shard_class": "concat_output"})
        if p + "expert_w2_weight" not in pack.manifest:
            pack.add(p + "expert_w2_weight", b"".join(w2_pay),
                     KIND_MXFP4_INTERLEAVED, [experts, latent, inter],
                     {"interleave": w2_geom, "shard_class": "input_dim"})
        s1 = reader.bf16(m + "shared_experts.gate_proj.weight", (shared, hidden))
        s3 = reader.bf16(m + "shared_experts.up_proj.weight", (shared, hidden))
        pack.add(p + "shared_w1_weight", s1 + s3, KIND_BF16, [2 * shared, hidden])
        bf(p + "shared_w2_weight", m + "shared_experts.down_proj.weight",
           (hidden, shared))
        bf(p + "routed_down_weight", m + "routed_expert_down_proj.weight",
           (latent, hidden))
        bf(p + "routed_up_weight", m + "routed_expert_up_proj.weight",
           (hidden, latent))
        bf(p + "routed_norm_weight", m + "routed_expert_norm.weight", (latent,))

    if first_layer + layer_count == layers:
        bf("model.norm.weight", "language_model.model.norm.weight", (hidden,))
        gamma = reader.bf16("language_model.model.output_attn_res_norm.weight", (hidden,))
        proj = reader.bf16("language_model.model.output_attn_res_proj.weight", (1, hidden))
        pack.add("model.attnres_out_weight", gamma_fold_bf16(proj, gamma, hidden),
                 KIND_BF16, [1, hidden])
        bf("lm_head.weight", "language_model.lm_head.weight", (config["vocab_size"], hidden))

    validate_layout(pack.manifest, {"hidden": hidden})

    pack.close()
    echo = {"hidden": hidden, "layers": layer_count, "first_layer": first_layer,
            "total_layers": layers, "experts": experts,
            "top_k": top_k, "latent": latent, "intermediate": inter,
            "group": GROUP, "vocab": config["vocab_size"],
            "kda_heads": kda_heads, "kda_head": kda_head, "heads": heads,
            "kv_lora": kv_lora, "rope": rope, "v_head": v_head,
            "nope": nope, "shared": shared, "q_lora": q_lora,
            "index_topk": dsa["index_topk"] if dsa is not None else 0,
            "index_heads": index_heads if dsa is not None else 0,
            "index_dim": index_dim if dsa is not None else 0}
    fmt = {"version": VERSION, "alignment": ALIGN,
           "mxfp4_interleave": {"tile_k": TILE_K, "group": GROUP,
                                "stored_bits": 4,
                                "cell_payload_rows": CELL_PAYLOAD_ROWS,
                                "cell_rows": CELL_PAYLOAD_ROWS + 1,
                                "row_bytes": TILE_K * 4 // 8,
                                "scale_bytes_per_neuron_tile": TILE_K // GROUP},
           "kda_fused": {"qkvb_sections": ["q", "k", "v", "beta"],
                         "decay_gate_down_sections": ["decay_down",
                                                      "gate_down"]}}
    assemble(out_path, payload_path, MAGIC, VERSION, ALIGN, fmt, echo, pack.manifest)
    return echo, pack.manifest


def main():
    if len(sys.argv) not in (3, 5, 6):
        print("usage: k3_pack.py <checkpoint_dir> <out.pack> "
              "[first_layer layer_count [expert_tile_k]]")
        return 2
    try:
        if len(sys.argv) == 6:
            echo, manifest = pack_model(sys.argv[1], sys.argv[2],
                                        int(sys.argv[3]), int(sys.argv[4]),
                                        int(sys.argv[5]))
        elif len(sys.argv) == 5:
            echo, manifest = pack_model(sys.argv[1], sys.argv[2],
                                        int(sys.argv[3]), int(sys.argv[4]))
        else:
            echo, manifest = pack_model(sys.argv[1], sys.argv[2])
    except PackFailure as failure:
        print(f"PACK FAILURE: {failure}")
        return 1
    print(f"packed {len(manifest)} tensors, "
          f"{sum(t['bytes'] for t in manifest.values())} payload bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
