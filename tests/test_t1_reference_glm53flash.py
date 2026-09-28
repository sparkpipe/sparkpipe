import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from t1_reference_common import (  # noqa: E402
    _E2M1_LUT, _E4M3_LUT, bf16_to_f32, f32_to_bf16_u16, read_fixture)

DEFINES = """#pragma once

#define SPARK_LLM_FAMILY_TAG                    glm53flash
#define SPARK_LLM_HIDDEN_DIMENSION              16u
#define SPARK_LLM_LAYER_COUNT                   2u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            32u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        64u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-05f
#define SPARK_LLM_SWIGLU_LIMIT                  10.0f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          31u
#define SPARK_LLM_MLA_HEAD_COUNT                2u
#define SPARK_LLM_MLA_QUERY_A_DIMENSION         8u
#define SPARK_LLM_MLA_LATENT_DIMENSION          8u
#define SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION    4u
#define SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION    0u
#define SPARK_LLM_MLA_V_HEAD_DIMENSION          8u
#define SPARK_LLM_MOE_EXPERT_COUNT              4u
#define SPARK_LLM_MOE_TOP_K                     2u
#define SPARK_LLM_MOE_SHARED_EXPERT_COUNT       1u
#define SPARK_LLM_MOE_INTERMEDIATE_DIMENSION    16u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     2.5f
#define SPARK_LLM_MOE_NORM_TOPK_PROB            1u
#define SPARK_LLM_FIRST_ROUTED_LAYER            1u
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  16u
#define SPARK_LLM_KDA_HEAD_COUNT                2u
#define SPARK_LLM_KDA_HEAD_KEY_DIMENSION        4u
#define SPARK_LLM_KDA_CONV_KERNEL               4u
#define SPARK_LLM_KDA_GATE_LOWER_BOUND          -5.0f
"""

E2M1_GRID = _E2M1_LUT
E4M3_CODES = np.array([c for c in range(256)
                       if not np.isnan(_E4M3_LUT[c])], dtype=np.uint8)


def e4m3_codes(x):
    flat = x.reshape(-1)
    return E4M3_CODES[np.argmin(np.abs(flat[:, None]
                                       - _E4M3_LUT[E4M3_CODES][None, :]),
                                 axis=1)].reshape(x.shape)


def e4m3_quantize(x):
    return _E4M3_LUT[e4m3_codes(x)].astype(np.float32)


def fp8_payload(x):
    codes = e4m3_codes(x)
    payload = np.empty(codes.shape, dtype=[("f8", "u1")])
    payload["f8"] = codes
    return payload


def e2m1_codes(x):
    flat = x.reshape(-1)
    return np.array([int(np.argmin(np.abs(_E2M1_LUT - v))) for v in flat],
                    dtype=np.uint8)


def pack_e2m1(x):
    codes = e2m1_codes(x)
    packed = np.zeros(codes.size // 2, dtype=np.uint8)
    packed[0::1] = codes[0::2] | (codes[1::2] << 4)
    return packed.reshape(x.shape[0], x.shape[1] // 2).astype(np.uint8)


def config_document():
    return {"text_config": {
        "hidden_size": 16, "num_hidden_layers": 2, "vocab_size": 32,
        "rms_norm_eps": 1e-5, "hc_mult": 2, "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 3, "num_attention_heads": 2,
        "kv_lora_rank": 8, "qk_nope_head_dim": 4, "qk_rope_head_dim": 0,
        "v_head_dim": 4, "q_lora_rank": 8, "index_topk": 8,
        "index_n_heads": 2, "index_head_dim": 4, "index_kpool": 2,
        "max_position_embeddings": 64, "swiglu_limit": 10.0,
        "intermediate_size": 16, "moe_intermediate_size": 16,
        "n_routed_experts": 4, "n_shared_experts": 1,
        "num_experts_per_tok": 2, "routed_scaling_factor": 2.5,
        "norm_topk_prob": True, "first_k_dense_replace": 1,
        "eos_token_id": [31], "scoring_func": "sigmoid",
        "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        "mlp_layer_types": ["dense", "sparse"],
        "linear_attn_config": {
            "head_dim": 4, "num_heads": 2, "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0, "kda_layers": [0],
            "full_attn_layers": [1],
        },
    }}


def grid_weights(shape, rng, grid):
    raw = rng.standard_normal(shape).astype(np.float32).reshape(-1) * 0.5
    picked = grid[np.argmin(np.abs(raw[:, None] - grid[None, :]), axis=1)]
    return picked.reshape(shape).astype(np.float32)


def build_tensors(quant):
    rng = np.random.default_rng(11)
    bf16 = {}
    t = {}

    def store(name, array):
        t[name] = array

    def bf16_name(name, array):
        store(name, f32_to_bf16_u16(array))

    embed = f32_to_bf16_u16(rng.standard_normal((32, 16)).astype(np.float32)
                            * 0.05)
    store("model.language_model.embed_tokens.weight", embed)
    store("model.language_model.norm.weight",
          f32_to_bf16_u16(rng.standard_normal(16).astype(np.float32) * 0.05))
    head = f32_to_bf16_u16(rng.standard_normal((32, 16)).astype(np.float32)
                           * 0.5)
    head[31] = np.zeros(16, dtype=np.uint16)
    store("lm_head.weight", head)
    e2m1 = _E2M1_LUT
    dense_grid = e2m1
    expert_grid = e2m1

    def maybe_fp8(name, values):
        if quant == "fp8":
            store(name, fp8_payload(values))
            store(name + "_scale_inv",
                  np.ones((1, 1), dtype=np.float32))
        else:
            bf16_name(name, values)

    def maybe_nvfp4(name, values):
        if quant == "nvfp4":
            store(name, pack_e2m1(values))
            store(name + "_scale",
                  np.full((values.shape[0],
                           values.shape[1] // 16), 0x38, dtype=np.uint8))
            store(name + "_scale_2", np.float32(1.0))
        else:
            bf16_name(name, values)

    for layer in range(2):
        p = f"model.language_model.layers.{layer}."
        bf16_name(p + "input_layernorm.weight",
                  rng.standard_normal(16).astype(np.float32) * 0.05)
        bf16_name(p + "post_attention_layernorm.weight",
                  rng.standard_normal(16).astype(np.float32) * 0.05)
        bf16_name(p + "hc_attn_fn",
                  rng.standard_normal((8, 32)).astype(np.float32) * 0.05)
        store(p + "hc_attn_base",
              rng.standard_normal(8).astype(np.float32) * 0.05)
        store(p + "hc_attn_scale",
              rng.standard_normal(3).astype(np.float32) + 1.0)
        bf16_name(p + "hc_ffn_fn",
                  rng.standard_normal((8, 32)).astype(np.float32) * 0.05)
        store(p + "hc_ffn_base",
              rng.standard_normal(8).astype(np.float32) * 0.05)
        store(p + "hc_ffn_scale",
              rng.standard_normal(3).astype(np.float32) + 1.0)
        if layer == 0:
            for tail in ("q_proj", "k_proj", "v_proj"):
                bf16_name(p + f"self_attn.{tail}.weight",
                          grid_weights((8, 16), rng, _E4M3_LUT[E4M3_CODES]))
            bf16_name(p + "self_attn.b_proj.weight",
                      grid_weights((2, 16), rng, _E4M3_LUT[E4M3_CODES]))
            for tail in ("q_conv1d", "k_conv1d", "v_conv1d"):
                if quant == "nvfp4":
                    store(p + f"self_attn.{tail}.weight",
                          grid_weights((8, 1, 4), rng,
                                       _E4M3_LUT[E4M3_CODES]).astype(
                                           np.float32))
                else:
                    bf16_name(p + f"self_attn.{tail}.weight",
                              grid_weights((8, 1, 4), rng,
                                           _E4M3_LUT[E4M3_CODES]))
            store(p + "self_attn.A_log",
                  rng.standard_normal(2).astype(np.float32) * 0.1)
            store(p + "self_attn.dt_bias",
                  rng.standard_normal(8).astype(np.float32) * 0.1)
            bf16_name(p + "self_attn.o_norm.weight",
                      rng.standard_normal(4).astype(np.float32) * 0.1)
            bf16_name(p + "self_attn.f_a_proj.weight",
                      grid_weights((8, 16), rng, _E4M3_LUT[E4M3_CODES]))
            bf16_name(p + "self_attn.f_b_proj.weight",
                      grid_weights((8, 8), rng, _E4M3_LUT[E4M3_CODES]))
            bf16_name(p + "self_attn.g_a_proj.weight",
                      grid_weights((8, 16), rng, _E4M3_LUT[E4M3_CODES]))
            bf16_name(p + "self_attn.g_b_proj.weight",
                      grid_weights((8, 8), rng, _E4M3_LUT[E4M3_CODES]))
            bf16_name(p + "self_attn.o_proj.weight",
                      grid_weights((16, 8), rng, _E4M3_LUT[E4M3_CODES]))
            for tail in ("gate_proj", "up_proj"):
                maybe_nvfp4(p + f"mlp.{tail}.weight",
                            grid_weights((16, 16), rng, dense_grid))
            maybe_nvfp4(p + "mlp.down_proj.weight",
                        grid_weights((16, 16), rng, dense_grid))
        else:
            maybe_fp8(p + "self_attn.q_a_proj.weight",
                      grid_weights((8, 16), rng, _E4M3_LUT[E4M3_CODES]))
            bf16_name(p + "self_attn.q_a_layernorm.weight",
                      rng.standard_normal(8).astype(np.float32) * 0.05)
            maybe_fp8(p + "self_attn.q_b_proj.weight",
                      grid_weights((8, 8), rng, _E4M3_LUT[E4M3_CODES]))
            maybe_fp8(p + "self_attn.kv_a_proj_with_mqa.weight",
                      grid_weights((8, 16), rng, _E4M3_LUT[E4M3_CODES]))
            bf16_name(p + "self_attn.kv_a_layernorm.weight",
                      rng.standard_normal(8).astype(np.float32) * 0.05)
            bf16_name(p + "self_attn.kv_b_proj.weight",
                      grid_weights((16, 8), rng, _E4M3_LUT[E4M3_CODES]))
            bf16_name(p + "self_attn.o_proj.weight",
                      grid_weights((16, 8), rng, _E4M3_LUT[E4M3_CODES]))
            bf16_name(p + "mlp.gate.weight",
                      rng.standard_normal((4, 16)).astype(np.float32) * 0.5)
            store(p + "mlp.gate.e_score_correction_bias",
                  rng.standard_normal(4).astype(np.float32) * 0.1)
            for e in range(4):
                ep = p + f"mlp.experts.{e}."
                for tail in ("gate_proj", "up_proj"):
                    maybe_nvfp4(ep + tail + ".weight",
                                grid_weights((16, 16), rng, expert_grid))
                maybe_nvfp4(ep + "down_proj.weight",
                            grid_weights((16, 16), rng, expert_grid))
            sp = p + "mlp.shared_experts."
            for tail in ("gate_proj", "up_proj"):
                maybe_fp8(sp + tail + ".weight",
                          grid_weights((16, 16), rng, _E4M3_LUT[E4M3_CODES]))
            maybe_fp8(sp + "down_proj.weight",
                      grid_weights((16, 16), rng, _E4M3_LUT[E4M3_CODES]))
    del bf16
    return t


def write_safetensors(path, tensors):
    header = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        array = tensors[name]
        if array.dtype == np.uint16:
            dtype, itemsize = "BF16", 2
        elif array.dtype == np.float32:
            dtype, itemsize = "F32", 4
        elif array.dtype == np.uint8:
            dtype, itemsize = "U8", 1
        elif array.dtype.names:
            dtype, itemsize = "F8_E4M3", 1
        else:
            raise AssertionError(f"unhandled dtype for {name}")
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for raw in blobs:
            fh.write(raw)


def write_checkpoint(directory, quant):
    os.makedirs(directory, exist_ok=True)
    write_safetensors(os.path.join(directory, "model.safetensors"),
                      build_tensors(quant))
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(config_document(), fh)


def write_header(path, text=DEFINES):
    with open(path, "w") as fh:
        fh.write(text)


def write_prompts(path):
    document = {"prompts": [{
        "name": "synth_a",
        "prompt_token_ids": [3, 7, 11, 5],
        "new_tokens": 3,
        "capture_layers": [0, 1],
    }]}
    with open(path, "w") as fh:
        json.dump(document, fh)


def run_generator(checkpoint, header, prompts, output):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
         "--family", "glm53flash", "--checkpoint", checkpoint,
         "--header", header, "--prompts", prompts, "--output", output],
        capture_output=True, text=True)


def compare(reference, candidate):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_compare.py"),
         "compare", "--reference", reference, "--candidate", candidate],
        capture_output=True, text=True)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def hand_computed_layer0(engine, token_id):
    prefix = "model.language_model.layers.0."
    embedding = bf16_to_f32(engine.st.raw_rows(
        "model.language_model.embed_tokens.weight", token_id, 1)[0])
    streams = np.tile(embedding, (2, 1))
    flat = streams.reshape(-1)
    inv = 1.0 / np.sqrt((flat * flat).sum() / (2 * 16) + 1e-5)
    fn = engine.tensor(prefix + "hc_attn_fn")
    mixes = (fn @ flat) * inv
    base = engine.raw(prefix + "hc_attn_base").astype(np.float32)
    scale = engine.raw(prefix + "hc_attn_scale").astype(np.float32)
    pre = 1.0 / (1.0 + np.exp(-(mixes[0:2] * scale[0] + base[0:2]))) + 1e-6
    post = 2.0 / (1.0 + np.exp(-(mixes[2:4] * scale[1] + base[2:4])))
    rows = mixes[4:8].reshape(2, 2) * scale[2] \
        + base[4:8].reshape(2, 2)
    comb = np.exp(rows - rows.max())
    comb = comb / comb.sum(axis=1, keepdims=True) + 1e-6
    for it in range(3):
        if it != 0:
            comb = comb / (comb.sum(axis=1, keepdims=True) + 1e-6)
        comb = comb / (comb.sum(axis=0, keepdims=True) + 1e-6)
    from t1_reference_common import bf16_round_f32, rmsnorm
    collapsed = bf16_round_f32(np.sum(pre[:, None] * streams, axis=0))
    x = bf16_round_f32(rmsnorm(
        collapsed, engine.tensor(prefix + "input_layernorm.weight"), 1e-5))
    heads, kd, conv = 2, 4, 4
    q_raw = bf16_round_f32(engine.tensor(
        prefix + "self_attn.q_proj.weight") @ x)
    k_raw = bf16_round_f32(engine.tensor(
        prefix + "self_attn.k_proj.weight") @ x)
    v_raw = bf16_round_f32(engine.tensor(
        prefix + "self_attn.v_proj.weight") @ x)
    b_raw = bf16_round_f32(engine.tensor(
        prefix + "self_attn.b_proj.weight") @ x)

    def conv_step(raw, weights_name, taps):
        weights = bf16_round_f32(engine.tensor(weights_name).reshape(
            heads * kd, conv))
        window = np.concatenate([taps[:, 1:],
                                 f32_to_bf16_u16(raw).reshape(-1, 1)],
                                axis=1)
        acc = (bf16_to_f32(window) * weights).sum(axis=1)
        return bf16_round_f32(acc / (1.0 + np.exp(-acc))), window

    q_conv, wq = conv_step(q_raw, prefix + "self_attn.q_conv1d.weight",
                           np.zeros((heads * kd, conv), dtype=np.uint16))
    k_conv, wk = conv_step(k_raw, prefix + "self_attn.k_conv1d.weight",
                           np.zeros((heads * kd, conv), dtype=np.uint16))
    v_conv, wv = conv_step(v_raw, prefix + "self_attn.v_conv1d.weight",
                           np.zeros((heads * kd, conv), dtype=np.uint16))

    def l2(m):
        m = m.reshape(heads, kd)
        return m / np.sqrt((m * m).sum(axis=1, keepdims=True) + 1e-6)

    latent = bf16_round_f32(engine.tensor(
        prefix + "self_attn.f_a_proj.weight") @ x)
    dl = bf16_round_f32(engine.tensor(
        prefix + "self_attn.f_b_proj.weight").T @ latent)
    a_log = engine.raw(prefix + "self_attn.A_log").astype(np.float32)
    dt_bias = engine.raw(prefix + "self_attn.dt_bias").astype(np.float32)
    retention = np.exp(-5.0 * (1.0 / (1.0 + np.exp(
        -(np.exp(a_log).reshape(heads, 1)
          * (dl.reshape(heads, kd) + dt_bias.reshape(heads, kd)))))))
    beta = 1.0 / (1.0 + np.exp(-b_raw.reshape(heads)))
    q2 = l2(q_conv) / np.sqrt(np.float32(kd))
    k2 = l2(k_conv)
    v2 = v_conv.reshape(heads, kd)
    state = np.zeros((heads, kd, kd), dtype=np.float32)
    out = np.empty((heads, kd), dtype=np.float32)
    for h in range(heads):
        pred = (state[h] * (k2[h] * retention[h])[:, None]).sum(axis=0)
        state[h] = (retention[h][:, None] * state[h]
                    + beta[h] * (v2[h] - pred)[None, :] * k2[h][:, None])
        out[h] = (state[h] * q2[h][:, None]).sum(axis=0)
    o32 = out
    rms = np.sqrt((o32 * o32).sum(axis=1) / kd + 1e-5)
    o_norm = bf16_to_f32(engine.raw(
        prefix + "self_attn.o_norm.weight").reshape(-1))
    gated_o = o32 / rms[:, None] * o_norm[None, :]
    gate = bf16_round_f32(bf16_round_f32(engine.tensor(
        prefix + "self_attn.g_a_proj.weight") @ x)
        @ engine.tensor(prefix + "self_attn.g_b_proj.weight").T)
    gs = 1.0 / (1.0 + np.exp(-gate.reshape(heads, kd)))
    attention = bf16_round_f32(engine.tensor(
        prefix + "self_attn.o_proj.weight") @ (gated_o * gs).reshape(-1))
    streams = bf16_round_f32(
        bf16_round_f32(bf16_round_f32(post)[:, None] * attention)
        + bf16_round_f32(bf16_round_f32(comb).T @ streams))
    flat = streams.reshape(-1)
    inv = 1.0 / np.sqrt((flat * flat).sum() / (2 * 16) + 1e-5)
    fn = engine.tensor(prefix + "hc_ffn_fn")
    mixes = (fn @ flat) * inv
    base = engine.raw(prefix + "hc_ffn_base").astype(np.float32)
    scale = engine.raw(prefix + "hc_ffn_scale").astype(np.float32)
    pre = 1.0 / (1.0 + np.exp(-(mixes[0:2] * scale[0] + base[0:2]))) + 1e-6
    post = 2.0 / (1.0 + np.exp(-(mixes[2:4] * scale[1] + base[2:4])))
    rows = mixes[4:8].reshape(2, 2) * scale[2] \
        + base[4:8].reshape(2, 2)
    comb = np.exp(rows - rows.max())
    comb = comb / comb.sum(axis=1, keepdims=True) + 1e-6
    for it in range(3):
        if it != 0:
            comb = comb / (comb.sum(axis=1, keepdims=True) + 1e-6)
        comb = comb / (comb.sum(axis=0, keepdims=True) + 1e-6)
    collapsed = bf16_round_f32(np.sum(pre[:, None] * streams, axis=0))
    x = bf16_round_f32(rmsnorm(
        collapsed, engine.tensor(prefix + "post_attention_layernorm.weight"),
        1e-5))
    gate_v = np.minimum(bf16_round_f32(engine.tensor(
        prefix + "mlp.gate_proj.weight") @ x), 10.0)
    up_v = np.clip(bf16_round_f32(engine.tensor(
        prefix + "mlp.up_proj.weight") @ x), -10.0, 10.0)
    activated = bf16_round_f32(bf16_round_f32(
        gate_v / (1.0 + np.exp(-gate_v))) * up_v)
    mlp = bf16_round_f32(engine.tensor(
        prefix + "mlp.down_proj.weight") @ activated)
    return bf16_round_f32(
        bf16_round_f32(bf16_round_f32(post)[:, None] * mlp)
        + bf16_round_f32(bf16_round_f32(comb).T @ streams))


OCP_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


class Nvfp4Store:
    def __init__(self, arrays):
        self.arrays = arrays

    def pread(self, name):
        return self.arrays[name]

    def entry(self, name):
        return {"dtype": "U8" if self.arrays[name].dtype == np.uint8
                else "F32"}


def nvfp4_semantics_arrays(name, global_scale):
    codes = np.array([list(range(16)), list(range(15, -1, -1))],
                     dtype=np.uint8)
    payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    scales = np.array([[0x38], [0x40]], dtype=np.uint8)
    want = np.array([[(-1.0 if c & 8 else 1.0) * OCP_E2M1[c & 7]
                      for c in row] for row in codes], dtype=np.float32)
    want *= np.array([[1.0], [2.0]], dtype=np.float32) * np.float32(
        global_scale)
    arrays = {name: payload, name + "_scale": scales,
              name + "_scale_2": np.array(global_scale, dtype=np.float32)}
    return arrays, want


def nvfp4_scale_semantics(module):
    name = "w.weight"
    arrays, want = nvfp4_semantics_arrays(name, 0.75)
    engine = object.__new__(module.ENGINE_CLASS)
    engine.st = Nvfp4Store(arrays)
    engine.raw = engine.st.pread
    spine = engine.tensor(name)
    expect(np.array_equal(spine, want),
           f"nvfp4 spine decode is not e2m1 x e4m3 x scale_2: {spine} vs {want}")
    expert = engine.expert_weight(name)
    expect(np.array_equal(expert, want),
           f"nvfp4 expert decode is not e2m1 x e4m3 x scale_2: {expert} vs {want}")


def main():
    workspace = tempfile.mkdtemp(prefix="t1ref-flash-test-")
    try:
        header = os.path.join(workspace, "llm_defines.h")
        write_header(header)
        prompts = os.path.join(workspace, "prompts.json")
        write_prompts(prompts)
        fixtures = {}
        for quant in ("bf16", "fp8", "nvfp4"):
            checkpoint = os.path.join(workspace, f"checkpoint-{quant}")
            write_checkpoint(checkpoint, quant)
            out = os.path.join(workspace, f"run-{quant}")
            result = run_generator(checkpoint, header, prompts, out)
            expect(result.returncode == 0,
                   f"{quant} generator failed: {result.stderr}")
            fixtures[quant] = os.path.join(out, "glm53flash", "synth_a.t1r")
        expect(open(fixtures["bf16"], "rb").read()
               == open(fixtures["fp8"], "rb").read(),
               "fp8 source fixture differs from the bf16 anchor")
        expect(open(fixtures["bf16"], "rb").read()
               == open(fixtures["nvfp4"], "rb").read(),
               "nvfp4 source fixture differs from the bf16 anchor")
        out_again = os.path.join(workspace, "run-bf16-again")
        checkpoint = os.path.join(workspace, "checkpoint-bf16")
        result = run_generator(checkpoint, header, prompts, out_again)
        expect(result.returncode == 0, f"rerun failed: {result.stderr}")
        fixture_again = os.path.join(out_again, "glm53flash", "synth_a.t1r")
        expect(open(fixtures["bf16"], "rb").read()
               == open(fixture_again, "rb").read(),
               "generator is not byte-deterministic")
        _, arrays = read_fixture(fixtures["bf16"])
        expect("pos0000_layer0000_streams" in arrays,
               "anchor capture missing")
        expect("pos0006_layer0001_route_ids" in arrays,
               "route capture missing")
        expect(int(arrays["generated_token_ids"][0]) >= 0,
               "no generated tokens")
        manifest = json.load(open(os.path.join(workspace, "run-bf16",
                                               "glm53flash",
                                               "MANIFEST.json")))
        expect(manifest["fixtures"]["synth_a.t1r"]["sha256"]
               == hashlib.sha256(
                   open(fixtures["bf16"], "rb").read()).hexdigest(),
               "manifest sha mismatch")
        expect(len(manifest["defines_config_mismatches"]) == 1
               and manifest["defines_config_mismatches"][0]["define"]
               == "SPARK_LLM_MLA_V_HEAD_DIMENSION",
               "the recorded v_head_dim mismatch must be the only one")
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import t1_reference_glm53flash as module
        from t1_reference_common import parse_llm_defines
        defines = parse_llm_defines(header)
        config = config_document()["text_config"]
        nvfp4_scale_semantics(module)
        engine = module.ENGINE_CLASS(checkpoint, defines, config)
        hand = f32_to_bf16_u16(hand_computed_layer0(engine, 3).reshape(-1))
        engine_out = arrays["pos0000_layer0000_streams"]
        expect(np.array_equal(hand, engine_out),
               f"hand-computed layer 0 diverges: "
               f"{np.abs(hand.astype(np.int32) - engine_out.astype(np.int32)).max()}")
        passed = compare(fixtures["bf16"], fixture_again)
        expect(passed.returncode == 0,
               f"identical fixtures must PASS: {passed.stdout} {passed.stderr}")
        corrupted = os.path.join(workspace, "synth_a_corrupt.t1r")
        corrupt = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "tools", "t1_reference_compare.py"),
             "corrupt-fixture", "--source", fixtures["bf16"],
             "--target", corrupted,
             "--array", "pos0001_layer0000_streams", "--offset", "9"],
            capture_output=True, text=True)
        expect(corrupt.returncode == 0,
               f"corrupt-fixture failed: {corrupt.stderr}")
        diverged = compare(fixtures["bf16"], corrupted)
        expect(diverged.returncode == 1, "corrupted fixture must FAIL")
        expect("pos0001_layer0000_streams" in diverged.stdout,
               f"FAIL must name the corrupted array: {diverged.stdout}")
        bad_header = os.path.join(workspace, "llm_defines_bad.h")
        write_header(bad_header, DEFINES.replace(
            "SPARK_LLM_MLA_LATENT_DIMENSION          8u",
            "SPARK_LLM_MLA_LATENT_DIMENSION          9u"))
        mismatch = run_generator(checkpoint, bad_header, prompts,
                                 os.path.join(workspace, "run-bad"))
        expect(mismatch.returncode != 0,
               "defines/config disagreement must fail loud")
        expect("MLA_LATENT_DIMENSION" in mismatch.stderr,
               f"failure must name the mismatched define: {mismatch.stderr}")
        bad_tensors = build_tensors("bf16")
        bad_tensors["model.language_model.layers.1.self_attn.kv_b_proj.weight"] = \
            bad_tensors[
                "model.language_model.layers.1.self_attn.kv_b_proj.weight"]
        del bad_tensors[
            "model.language_model.layers.1.self_attn.kv_b_proj.weight"]
        bad_tensors[
            "model.language_model.layers.1.self_attn.kv_b_proj.weight"] = \
            np.zeros((16, 8), dtype=np.float32)
        bad_dir = os.path.join(workspace, "checkpoint-bad")
        os.makedirs(bad_dir, exist_ok=True)
        write_safetensors(os.path.join(bad_dir, "model.safetensors"),
                          bad_tensors)
        with open(os.path.join(bad_dir, "config.json"), "w") as fh:
            json.dump(config_document(), fh)
        try:
            module.ENGINE_CLASS(bad_dir, defines, config)
            raise AssertionError("f32 kv_b_proj must fail closed")
        except ValueError as error:
            expect("kv_b_proj" in str(error),
                   f"failure must name kv_b_proj: {error}")
        shutil.rmtree(workspace, ignore_errors=True)
        print("PASS t1_reference_glm53flash synthetic proof: determinism, "
              "bf16/fp8/nvfp4 codec invariance, nvfp4 OCP scale semantics, "
              "hand-computed layer 0 "
              "bit-identical, manifest sha, negative control, "
              "defines/config fail-closed, dtype fail-closed")
        return 0
    except AssertionError:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
