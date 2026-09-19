import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from t1_reference_common import (bf16_round_f32, bf16_to_f32,  # noqa: E402
                                 f32_to_bf16_u16, read_fixture, sigmoid)
from t1_reference_k3 import kda_delta_step, l2_per_head  # noqa: E402

BLOCK = 12


def defines_text(layers, experts, topk, shared, hidden=16, vocab=32,
                 kda_heads=2, kd=4, q_lora=8, kv_lora=8, nope=4, rope=4,
                 v_head=4, inter=32, routed_hidden=32, period=4, phase=3):
    kda = layers - len({i for i in range(layers)
                        if i % period == phase or i == layers - 1})
    return f"""#pragma once

#define SPARK_LLM_FAMILY_TAG                    k3
#define SPARK_LLM_HIDDEN_DIMENSION              {hidden}u
#define SPARK_LLM_LAYER_COUNT                   {layers}u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            {vocab}u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-05f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          {vocab - 1}u
#define SPARK_LLM_ATTENTION_PERIOD              {period}u
#define SPARK_LLM_GLOBAL_ATTENTION_PHASE        {phase}u
#define SPARK_LLM_KDA_LAYER_COUNT               {kda}u
#define SPARK_LLM_KDA_HEAD_COUNT                {kda_heads}u
#define SPARK_LLM_KDA_HEAD_KEY_DIMENSION        {kd}u
#define SPARK_LLM_KDA_HEAD_VALUE_DIMENSION      {kd}u
#define SPARK_LLM_KDA_CONV_KERNEL               4u
#define SPARK_LLM_KDA_GATE_LOWER_BOUND          -5.0f
#define SPARK_LLM_MLA_HEAD_COUNT                2u
#define SPARK_LLM_MLA_QUERY_A_DIMENSION         {q_lora}u
#define SPARK_LLM_MLA_LATENT_DIMENSION          {kv_lora}u
#define SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION    {nope}u
#define SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION    {rope}u
#define SPARK_LLM_MLA_V_HEAD_DIMENSION          {v_head}u
#define SPARK_LLM_MOE_EXPERT_COUNT              {experts}u
#define SPARK_LLM_MOE_TOP_K                     {topk}u
#define SPARK_LLM_MOE_SHARED_EXPERT_COUNT       {shared}u
#define SPARK_LLM_MOE_INTERMEDIATE_DIMENSION    {inter}u
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  {inter}u
#define SPARK_LLM_MOE_ROUTED_EXPERT_HIDDEN_DIMENSION {routed_hidden}u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     1.0f
#define SPARK_LLM_MOE_NORM_TOPK_PROB            1u
#define SPARK_LLM_FIRST_ROUTED_LAYER            1u
"""


def config_document(layers, experts, topk, shared, hidden=16, vocab=32,
                    kda_heads=2, kd=4, q_lora=8, kv_lora=8, nope=4, rope=4,
                    v_head=4, inter=32, routed_hidden=32, period=4, phase=3):
    full = sorted(i + 1 for i in range(layers)
                  if i % period == phase or i == layers - 1)
    kda = sorted(set(range(1, layers + 1)) - set(full))
    return {"text_config": {
        "hidden_size": hidden, "num_hidden_layers": layers,
        "vocab_size": vocab, "rms_norm_eps": 1e-5, "eos_token_id": vocab - 1,
        "first_k_dense_replace": 1, "num_experts": experts,
        "num_experts_per_token": topk, "num_shared_experts": shared,
        "moe_intermediate_size": inter, "routed_expert_hidden_size":
            routed_hidden,
        "routed_scaling_factor": 1.0, "num_attention_heads": 2,
        "intermediate_size": inter,
        "q_lora_rank": q_lora, "kv_lora_rank": kv_lora,
        "qk_nope_head_dim": nope, "qk_rope_head_dim": rope,
        "v_head_dim": v_head, "hidden_act": "situ",
        "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": BLOCK, "latent_moe_use_norm": True,
        "moe_renormalize": True, "moe_router_activation_func": "sigmoid",
        "num_expert_group": 1, "num_key_value_heads": 2,
        "mla_use_nope": True, "mla_use_output_gate": True,
        "attn_res_block_size": BLOCK,
        "linear_attn_config": {
            "num_heads": kda_heads, "head_dim": kd,
            "short_conv_kernel_size": 4, "gate_lower_bound": -5.0,
            "use_full_rank_gate": True, "full_attn_layers": full,
            "kda_layers": kda,
        },
    }}


def bf16(name, shape, rng, scale=0.05):
    del name
    return f32_to_bf16_u16(rng.standard_normal(shape).astype(np.float32) * scale)


def f32(name, shape, rng, scale=0.05):
    del name
    return rng.standard_normal(shape).astype(np.float32) * scale


def mxfp4_expert(rng, rows, in_dim, payload_scale=1.0):
    nibbles = rng.integers(0, 16, (rows, in_dim), dtype=np.uint8)
    payload = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4))
    scale = rng.integers(118, 137, (rows, in_dim // 32), dtype=np.uint8)
    del payload_scale
    return payload, scale


def build_tensors(layers, seed=7, experts=4, inter=32, routed_hidden=32,
                  topk=2, shared=1):
    rng = np.random.default_rng(seed)
    t = {}
    vocab, hidden = 32, 16
    t["language_model.model.embed_tokens.weight"] = bf16("e", (vocab, hidden),
                                                         rng)
    t["language_model.model.norm.weight"] = bf16("n", (hidden,), rng)
    t["language_model.model.output_attn_res_norm.weight"] = bf16(
        "on", (hidden,), rng, scale=0.5)
    t["language_model.model.output_attn_res_proj.weight"] = bf16(
        "op", (1, hidden), rng, scale=0.5)
    head = bf16("lm", (vocab, hidden), rng, scale=0.5)
    head[vocab - 1] = np.zeros(hidden, dtype=np.uint16)
    head[5] = f32_to_bf16_u16(np.full(hidden, 10.0, dtype=np.float32))
    t["language_model.lm_head.weight"] = head
    full = {i for i in range(layers)
            if i % 4 == 3 or i == layers - 1}
    for layer in range(layers):
        p = f"language_model.model.layers.{layer}."
        t[p + "input_layernorm.weight"] = bf16("il", (hidden,), rng)
        t[p + "post_attention_layernorm.weight"] = bf16("pl", (hidden,), rng)
        t[p + "self_attention_res_norm.weight"] = bf16("ar", (hidden,), rng,
                                                       scale=0.5)
        t[p + "self_attention_res_proj.weight"] = bf16("ap", (1, hidden), rng,
                                                       scale=0.5)
        t[p + "mlp_res_norm.weight"] = bf16("mr", (hidden,), rng, scale=0.5)
        t[p + "mlp_res_proj.weight"] = bf16("mp", (1, hidden), rng, scale=0.5)
        if layer in full:
            t[p + "self_attn.q_a_proj.weight"] = bf16("qa", (8, hidden), rng)
            t[p + "self_attn.q_a_layernorm.weight"] = bf16("qn", (8,), rng)
            t[p + "self_attn.q_b_proj.weight"] = bf16("qb", (16, 8), rng)
            t[p + "self_attn.kv_a_proj_with_mqa.weight"] = bf16("ka",
                                                                (12, hidden),
                                                                rng)
            t[p + "self_attn.kv_a_layernorm.weight"] = bf16("kn", (8,), rng)
            t[p + "self_attn.kv_b_proj.weight"] = bf16("kb", (16, 8), rng)
            t[p + "self_attn.g_proj.weight"] = bf16("gp", (8, hidden), rng)
            t[p + "self_attn.o_proj.weight"] = bf16("opj", (hidden, 8), rng)
        else:
            t[p + "self_attn.q_proj.weight"] = bf16("q", (8, hidden), rng)
            t[p + "self_attn.k_proj.weight"] = bf16("k", (8, hidden), rng)
            t[p + "self_attn.v_proj.weight"] = bf16("v", (8, hidden), rng)
            t[p + "self_attn.b_proj.weight"] = bf16("b", (2, hidden), rng)
            t[p + "self_attn.f_a_proj.weight"] = bf16("fa", (4, hidden), rng)
            t[p + "self_attn.f_b_proj.weight"] = bf16("fb", (8, 4), rng)
            t[p + "self_attn.g_proj.weight"] = bf16("gp", (8, hidden), rng)
            t[p + "self_attn.o_proj.weight"] = bf16("opj", (hidden, 8), rng)
            for c in "qkv":
                t[p + f"self_attn.{c}_conv1d.weight"] = f32(
                    c, (8, 1, 4), rng)
            t[p + "self_attn.A_log"] = f32("al", (128,), rng, scale=0.3)
            t[p + "self_attn.dt_bias"] = f32("db", (8,), rng)
            t[p + "self_attn.o_norm.weight"] = f32("on", (4,), rng, scale=0.5)
        if layer == 0:
            t[p + "mlp.gate_proj.weight"] = bf16("g", (32, hidden), rng)
            t[p + "mlp.up_proj.weight"] = bf16("u", (32, hidden), rng)
            t[p + "mlp.down_proj.weight"] = bf16("dn", (hidden, 32), rng)
        else:
            t[p + "block_sparse_moe.gate.weight"] = bf16("gw", (experts,
                                                                hidden), rng,
                                                         scale=0.5)
            t[p + "block_sparse_moe.gate.e_score_correction_bias"] = \
                f32("gb", (experts,), rng, scale=0.2)
            t[p + "block_sparse_moe.routed_expert_down_proj.weight"] = \
                bf16("rd", (routed_hidden, hidden), rng)
            t[p + "block_sparse_moe.routed_expert_up_proj.weight"] = \
                bf16("ru", (hidden, routed_hidden), rng)
            t[p + "block_sparse_moe.routed_expert_norm.weight"] = \
                bf16("rn", (routed_hidden,), rng)
            t[p + "block_sparse_moe.shared_experts.gate_proj.weight"] = \
                bf16("sg", (inter * shared, hidden), rng)
            t[p + "block_sparse_moe.shared_experts.up_proj.weight"] = \
                bf16("su", (inter * shared, hidden), rng)
            t[p + "block_sparse_moe.shared_experts.down_proj.weight"] = \
                bf16("sd", (hidden, inter * shared), rng)
            for e in range(experts):
                pay1, sc1 = mxfp4_expert(rng, inter, routed_hidden)
                pay3, sc3 = mxfp4_expert(rng, inter, routed_hidden)
                pay2, sc2 = mxfp4_expert(rng, routed_hidden, inter)
                base = p + f"block_sparse_moe.experts.{e}."
                t[base + "w1.weight_packed"] = pay1
                t[base + "w1.weight_scale"] = sc1
                t[base + "w3.weight_packed"] = pay3
                t[base + "w3.weight_scale"] = sc3
                t[base + "w2.weight_packed"] = pay2
                t[base + "w2.weight_scale"] = sc2
    return t


def write_safetensors(path, tensors):
    header = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        array = tensors[name]
        if array.dtype == np.uint16:
            dtype, itemsize = "BF16", 2
        elif array.dtype == np.uint8:
            dtype, itemsize = "U8", 1
        else:
            dtype, itemsize = "F32", 4
        raw = array.tobytes()
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


def write_checkpoint(directory, tensors, config):
    os.makedirs(directory, exist_ok=True)
    write_safetensors(os.path.join(directory, "model.safetensors"), tensors)
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(config, fh)


def write_prompts(path, new_tokens=2, ids=(3, 7, 11)):
    document = {"prompts": [{
        "name": "synth_k3",
        "prompt_token_ids": list(ids),
        "new_tokens": new_tokens,
        "capture_layers": [0, 3, 6, 12],
    }]}
    with open(path, "w") as fh:
        json.dump(document, fh)


def run_generator(checkpoint, header, prompts, output):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
         "--family", "k3", "--checkpoint", checkpoint, "--header", header,
         "--prompts", prompts, "--output", output],
        capture_output=True, text=True)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def kda_oracle_cross_check():
    """Cross-check the engine's delta step against the glm5-next KDA host
    oracle recurrence (tests/test_kda_host.py reference form), fp32, scalar
    transcription, heads=3 kd=vd=8 five steps."""
    rng = np.random.default_rng(23)
    heads, kd, vd, steps = 3, 8, 8, 5
    state = np.zeros((heads, kd, vd), dtype=np.float32)
    oracle = [[[0.0] * vd for _ in range(kd)] for _ in range(heads)]
    worst = 0.0
    for _ in range(steps):
        q = rng.standard_normal((heads, kd)).astype(np.float32)
        k = rng.standard_normal((heads, kd)).astype(np.float32)
        v = rng.standard_normal((heads, vd)).astype(np.float32)
        z = rng.standard_normal((heads, kd)).astype(np.float32)
        beta = rng.random(heads).astype(np.float32)
        for h in range(heads):
            kk = float((k[h] * k[h]).sum())
            qq = float((q[h] * q[h]).sum())
            kn = (k[h] / math.sqrt(kk + 1e-6)).astype(np.float32)
            qn = (q[h] / math.sqrt(qq + 1e-6)
                  / math.sqrt(kd)).astype(np.float32)
            engine_out = kda_delta_step(state[h], qn, kn, v[h],
                                        np.ones(kd, dtype=np.float32),
                                        np.float32(beta[h]))
            predicted = [sum(oracle[h][c][e] * kn[c] for c in range(kd))
                         for e in range(vd)]
            for c in range(kd):
                for e in range(vd):
                    oracle[h][c][e] = (oracle[h][c][e]
                                       + beta[h] * (v[h][e] - predicted[e])
                                       * kn[c])
            for e in range(vd):
                want = sum(oracle[h][c][e] * qn[c] for c in range(kd))
                worst = max(worst, abs(want - float(engine_out[e])))
            state[h] = state[h]
    print(f"[a] kda oracle cross-check vs host-oracle recurrence: worst "
          f"|diff| = {worst:.3e} over {steps} steps x {heads} heads")
    expect(worst < 1e-5, f"kda oracle cross-check diverged: {worst}")
    decay_worst = 0.0
    for _ in range(200):
        logit, bias, scale = rng.standard_normal(3).astype(np.float32) * 2
        del scale
        a_log = abs(float(rng.standard_normal())) * 0.5
        got = float(np.exp(-5.0 * sigmoid(
            np.exp(np.float32(a_log)) * (np.float32(logit)
                                         + np.float32(bias)))))
        want = math.exp(-5.0 / (1.0 + math.exp(
            -math.exp(a_log) * (logit + bias))))
        decay_worst = max(decay_worst, abs(got - want))
    print(f"[a] bounded-decay mapping vs test_kda_decay reference: worst "
          f"|diff| = {decay_worst:.3e}")
    expect(decay_worst < 1e-6, f"decay mapping diverged: {decay_worst}")


def hand_computed_layer0(workspace):
    """Layer 0 recomputed longhand from crafted weights.

    The weights pick single hidden components so every intermediate is a
    named scalar; the arithmetic is written out with math.* and the shared
    bf16 rounding helpers only.
    """
    rng = np.random.default_rng(11)
    hidden, vocab, kd, kda_dim = 4, 8, 2, 4
    embedding = np.zeros((vocab, hidden), dtype=np.uint16)
    row = np.array([0.5, -1.0, 2.0, -0.25], dtype=np.float32)
    embedding[3] = f32_to_bf16_u16(row)
    other = rng.standard_normal((vocab, hidden)).astype(np.float32) * 0.05
    for i in range(vocab):
        if i != 3:
            embedding[i] = f32_to_bf16_u16(other[i])
    norm = f32_to_bf16_u16(np.ones(hidden, dtype=np.float32))
    t = {"language_model.model.embed_tokens.weight": embedding,
         "language_model.model.norm.weight": norm,
         "language_model.model.output_attn_res_norm.weight":
             f32_to_bf16_u16(np.ones(hidden, dtype=np.float32)),
         "language_model.model.output_attn_res_proj.weight":
             f32_to_bf16_u16(np.array([[1.0, 0, 0, 0]], dtype=np.float32)),
         "language_model.lm_head.weight": embedding.copy()}
    unit = f32_to_bf16_u16(np.ones(hidden, dtype=np.float32))
    pick = np.eye(hidden, dtype=np.float32)
    p = "language_model.model.layers.0."
    t[p + "input_layernorm.weight"] = unit
    t[p + "post_attention_layernorm.weight"] = unit
    t[p + "self_attention_res_norm.weight"] = unit
    t[p + "self_attention_res_proj.weight"] = \
        f32_to_bf16_u16(np.array([[1.0, 0, 0, 0]], dtype=np.float32))
    t[p + "mlp_res_norm.weight"] = unit
    t[p + "mlp_res_proj.weight"] = \
        f32_to_bf16_u16(np.array([[0.0, 1.0, 0, 0]], dtype=np.float32))
    q_rows = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                       [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    v_rows = np.array([[1, 0, 0, 0], [1, 0, 0, 0],
                       [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    b_rows = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    t[p + "self_attn.q_proj.weight"] = f32_to_bf16_u16(q_rows)
    t[p + "self_attn.k_proj.weight"] = f32_to_bf16_u16(q_rows)
    t[p + "self_attn.v_proj.weight"] = f32_to_bf16_u16(v_rows)
    t[p + "self_attn.b_proj.weight"] = f32_to_bf16_u16(b_rows)
    conv = np.zeros((kda_dim, 1, 4), dtype=np.float32)
    conv[:, 0, 3] = 1.0
    for c in "qkv":
        t[p + f"self_attn.{c}_conv1d.weight"] = conv
    t[p + "self_attn.A_log"] = np.zeros(128, dtype=np.float32)
    t[p + "self_attn.dt_bias"] = np.zeros(kda_dim, dtype=np.float32)
    t[p + "self_attn.o_norm.weight"] = np.ones(kd, dtype=np.float32)
    t[p + "self_attn.f_a_proj.weight"] = f32_to_bf16_u16(b_rows)
    t[p + "self_attn.f_b_proj.weight"] = f32_to_bf16_u16(
        np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32))
    t[p + "self_attn.g_proj.weight"] = f32_to_bf16_u16(q_rows)
    t[p + "self_attn.o_proj.weight"] = f32_to_bf16_u16(pick)
    for name, shape in [("mlp.gate_proj.weight", (2, hidden)),
                        ("mlp.up_proj.weight", (2, hidden)),
                        ("mlp.down_proj.weight", (hidden, 2))]:
        t[p + name] = bf16(name, shape, rng)
    t[p + "mlp.gate_proj.weight"] = f32_to_bf16_u16(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    t[p + "mlp.up_proj.weight"] = f32_to_bf16_u16(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    t[p + "mlp.down_proj.weight"] = f32_to_bf16_u16(
        np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32))
    mlp_res_mix_weight = f32_to_bf16_u16(
        np.array([[0.0, 1.0, 0, 0]], dtype=np.float32))
    attn_res_mix_weight = f32_to_bf16_u16(
        np.array([[1.0, 0, 0, 0]], dtype=np.float32))
    full_attn = {3}
    for layer in range(1, 4):
        q = f"language_model.model.layers.{layer}."
        t[q + "input_layernorm.weight"] = unit
        t[q + "post_attention_layernorm.weight"] = unit
        t[q + "self_attention_res_norm.weight"] = unit
        t[q + "self_attention_res_proj.weight"] = attn_res_mix_weight \
            if layer in full_attn else f32_to_bf16_u16(
                np.array([[0, 0, 1.0, 0]], dtype=np.float32))
        t[q + "mlp_res_norm.weight"] = unit
        t[q + "mlp_res_proj.weight"] = mlp_res_mix_weight
        if layer in full_attn:
            t[q + "self_attn.q_a_proj.weight"] = bf16("qa", (4, hidden), rng)
            t[q + "self_attn.q_a_layernorm.weight"] = bf16("qn", (4,), rng)
            t[q + "self_attn.q_b_proj.weight"] = bf16("qb", (8, 4), rng)
            t[q + "self_attn.kv_a_proj_with_mqa.weight"] = bf16("ka",
                                                                (6, hidden),
                                                                rng)
            t[q + "self_attn.kv_a_layernorm.weight"] = bf16("kn", (4,), rng)
            t[q + "self_attn.kv_b_proj.weight"] = bf16("kb", (8, 4), rng)
            t[q + "self_attn.g_proj.weight"] = bf16("gp", (4, hidden), rng)
            t[q + "self_attn.o_proj.weight"] = bf16("opj", (hidden, 4), rng)
        else:
            t[q + "self_attn.q_proj.weight"] = bf16("q", (4, hidden), rng)
            t[q + "self_attn.k_proj.weight"] = bf16("k", (4, hidden), rng)
            t[q + "self_attn.v_proj.weight"] = bf16("v", (4, hidden), rng)
            t[q + "self_attn.b_proj.weight"] = bf16("b", (2, hidden), rng)
            t[q + "self_attn.f_a_proj.weight"] = bf16("fa", (2, hidden), rng)
            t[q + "self_attn.f_b_proj.weight"] = bf16("fb", (4, 2), rng)
            t[q + "self_attn.g_proj.weight"] = bf16("gp", (4, hidden), rng)
            t[q + "self_attn.o_proj.weight"] = bf16("opj", (hidden, 4), rng)
            for c in "qkv":
                t[q + f"self_attn.{c}_conv1d.weight"] = conv
            t[q + "self_attn.A_log"] = np.zeros(128, dtype=np.float32)
            t[q + "self_attn.dt_bias"] = np.zeros(4, dtype=np.float32)
            t[q + "self_attn.o_norm.weight"] = np.ones(2, dtype=np.float32)
        t[q + "block_sparse_moe.gate.weight"] = bf16("gw", (2, hidden), rng)
        t[q + "block_sparse_moe.gate.e_score_correction_bias"] = \
            f32("gb", (2,), rng, scale=0.2)
        t[q + "block_sparse_moe.routed_expert_down_proj.weight"] = \
            bf16("rd", (32, hidden), rng)
        t[q + "block_sparse_moe.routed_expert_up_proj.weight"] = \
            bf16("ru", (hidden, 32), rng)
        t[q + "block_sparse_moe.routed_expert_norm.weight"] = \
            bf16("rn", (32,), rng)
        t[q + "block_sparse_moe.shared_experts.gate_proj.weight"] = \
            bf16("sg", (32, hidden), rng)
        t[q + "block_sparse_moe.shared_experts.up_proj.weight"] = \
            bf16("su", (32, hidden), rng)
        t[q + "block_sparse_moe.shared_experts.down_proj.weight"] = \
            bf16("sd", (hidden, 32), rng)
        for e in range(2):
            nibbles = rng.integers(0, 16, (32, 32), dtype=np.uint8)
            payload = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4))
            scale = rng.integers(118, 137, (32, 1), dtype=np.uint8)
            base = q + f"block_sparse_moe.experts.{e}."
            t[base + "w1.weight_packed"] = payload
            t[base + "w1.weight_scale"] = scale
            t[base + "w3.weight_packed"] = payload.copy()
            t[base + "w3.weight_scale"] = scale.copy()
            t[base + "w2.weight_packed"] = payload.copy()
            t[base + "w2.weight_scale"] = scale.copy()
    config = config_document(4, 2, 1, 1, hidden=hidden, vocab=vocab,
                             kda_heads=2, kd=kd, q_lora=4, kv_lora=4,
                             nope=2, rope=2, v_head=2, inter=32,
                             routed_hidden=32)
    checkpoint = os.path.join(workspace, "hand_checkpoint")
    write_checkpoint(checkpoint, t, config)
    header = os.path.join(workspace, "hand_defines.h")
    with open(header, "w") as fh:
        fh.write(defines_text(4, 2, 1, 1, hidden=hidden, vocab=vocab,
                              kda_heads=2, kd=kd, q_lora=4, kv_lora=4,
                              nope=2, rope=2, v_head=2, inter=32,
                              routed_hidden=32))
    prompts = os.path.join(workspace, "hand_prompts.json")
    write_prompts(prompts, new_tokens=0, ids=(3, 5, 2))
    result = run_generator(checkpoint, header, prompts,
                           os.path.join(workspace, "hand_out"))
    expect(result.returncode == 0,
           f"hand-check generator failed: {result.stderr}")
    _, arrays = read_fixture(os.path.join(workspace, "hand_out", "k3",
                                          "synth_k3.t1r"))
    engine_partial = bf16_to_f32(
        arrays["pos0000_layer0000_streams"].reshape(-1))

    x = bf16_round_f32(bf16_to_f32(embedding[3]))
    inv = 1.0 / math.sqrt(sum(float(v) * float(v) for v in x) / hidden + 1e-5)
    n = bf16_round_f32(x * inv)
    q_raw = bf16_round_f32(np.array([n[0], n[1], n[0], n[1]], dtype=np.float32))
    v_raw = bf16_round_f32(np.array([n[0], n[0], n[1], n[1]], dtype=np.float32))
    b_raw = bf16_round_f32(np.array([n[0], n[1]], dtype=np.float32))

    def conv_first(raw):
        outs = []
        for value in raw:
            acc = float(value)
            outs.append(acc)
        return bf16_round_f32(
            np.array(outs, dtype=np.float32)
            * sigmoid(np.array(outs, dtype=np.float32)))

    q_conv = conv_first(q_raw)
    k_conv = conv_first(q_raw)
    v_conv = conv_first(v_raw)
    q1 = bf16_round_f32(l2_per_head(q_conv, 2, kd, 1e-5).reshape(-1))
    k1 = bf16_round_f32(l2_per_head(k_conv, 2, kd, 1e-5).reshape(-1))
    q2 = l2_per_head(q1, 2, kd, 1e-6) / math.sqrt(kd)
    k2 = l2_per_head(k1, 2, kd, 1e-6)
    latent = bf16_round_f32(np.array([n[0], n[1]], dtype=np.float32))
    decay_logit = bf16_round_f32(np.array(
        [latent[0], latent[1], latent[0], latent[1]], dtype=np.float32))
    retention = np.exp(-5.0 * sigmoid(decay_logit))
    beta = sigmoid(b_raw)
    v2 = v_conv.reshape(2, kd)
    state = np.zeros((2, kd, kd), dtype=np.float32)
    out = np.empty((2, kd), dtype=np.float32)
    for h in range(2):
        out[h] = kda_delta_step(state[h], q2[h], k2[h], v2[h],
                                retention.reshape(2, kd)[h], beta[h])
    o32 = bf16_round_f32(out.reshape(-1)).reshape(2, kd)
    rms = np.sqrt((o32 * o32).sum(axis=1) / kd + 1e-5)
    normed_o = bf16_round_f32(o32 / rms[:, None])
    gate = bf16_round_f32(np.array([n[0], n[1], n[0], n[1]],
                                   dtype=np.float32))
    gated = bf16_round_f32(normed_o.reshape(-1) * sigmoid(gate))
    attention = bf16_round_f32(gated)
    partial = attention
    bank0 = bf16_round_f32(x)
    mlp_weight = bf16_to_f32(mlp_res_mix_weight.reshape(-1))
    scores = []
    for values in (bank0, partial):
        vinv = 1.0 / math.sqrt(float((values * values).mean()) + 1e-5)
        scores.append(float(values[1]) * vinv * float(mlp_weight[1]))
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    weights = [w / total for w in weights]
    mixed = bf16_round_f32(weights[0] * bank0 + weights[1] * partial)
    inv2 = 1.0 / math.sqrt(float((mixed * mixed).mean()) + 1e-5)
    n2 = bf16_round_f32(mixed * inv2)
    g = bf16_round_f32(np.array([n2[0], n2[1]], dtype=np.float32))
    u = bf16_round_f32(np.array([n2[0], n2[1]], dtype=np.float32))
    act = 4.0 * np.tanh(g / 4.0) * sigmoid(g)
    lin = 25.0 * np.tanh(u / 25.0)
    inter = bf16_round_f32(act * lin)
    dense = bf16_round_f32(np.array(
        [inter[0], inter[1], inter[0], inter[1]], dtype=np.float32))
    final_partial = bf16_round_f32(partial + dense)
    worst = float(np.abs(engine_partial - final_partial).max())
    print(f"[b] hand-computed layer 0: engine partial {engine_partial.tolist()}")
    print(f"[b] hand partial             {final_partial.tolist()}, "
          f"worst |diff| = {worst:.3e}")
    expect(worst < 2e-2,
           f"hand-computed layer 0 disagrees with the engine: {worst}")


def main():
    workspace = tempfile.mkdtemp(prefix="t1ref-k3-")
    try:
        kda_oracle_cross_check()
        hand_computed_layer0(workspace)
        layers = 13
        tensors = build_tensors(layers)
        config = config_document(layers, 4, 2, 1)
        checkpoint = os.path.join(workspace, "checkpoint")
        write_checkpoint(checkpoint, tensors, config)
        header = os.path.join(workspace, "llm_defines.h")
        with open(header, "w") as fh:
            fh.write(defines_text(layers, 4, 2, 1))
        prompts = os.path.join(workspace, "prompts.json")
        write_prompts(prompts)
        out_a = os.path.join(workspace, "run_a")
        out_b = os.path.join(workspace, "run_b")
        engine_probe = None
        run_a = run_generator(checkpoint, header, prompts, out_a)
        expect(run_a.returncode == 0,
               f"generator failed: {run_a.stderr}")
        run_generator(checkpoint, header, prompts, out_b)
        fixture_a = os.path.join(out_a, "k3", "synth_k3.t1r")
        fixture_b = os.path.join(out_b, "k3", "synth_k3.t1r")
        expect(os.path.exists(fixture_a), "fixture missing after generation")
        _, arrays = read_fixture(fixture_a)
        expect("pos0000_layer0000_streams" in arrays, "anchor capture missing")
        expect("pos0004_layer0003_streams" in arrays,
               "mla-layer anchor capture missing")
        expect("pos0004_layer0001_route_ids" in arrays,
               "route capture missing")
        expect(arrays["pos0004_layer0001_route_ids"].shape[0] == 2,
               "route capture must carry top_k ids")
        expect(int(arrays["generated_token_ids"][0]) not in (0, 31),
               "generated token must be a real non-eot vocabulary entry")
        expect(open(fixture_a, "rb").read() == open(fixture_b, "rb").read(),
               "generator is not byte-deterministic")
        from t1_reference_k3 import K3Engine as _E
        import json as _json
        _cfg = _json.load(open(os.path.join(checkpoint, "config.json")))
        _defines = __import__("t1_reference_common").parse_llm_defines(header)
        _eng = _E(checkpoint, _defines, _cfg["text_config"])
        _pre = "language_model.model.layers.1."
        _g = _eng._scratch((_pre, "gate"), _eng.inter, _eng.routed_hidden)
        _u = _eng._scratch((_pre, "up"), _eng.inter, _eng.routed_hidden)
        expect(_g is not _u, "gate and up dequant planes must not alias")
        print("[c] scratch-plane independence verified (gate/up distinct)")
        manifest = json.load(open(os.path.join(out_a, "k3", "MANIFEST.json")))
        import hashlib
        expect(manifest["fixtures"]["synth_k3.t1r"]["sha256"]
               == hashlib.sha256(open(fixture_a, "rb").read()).hexdigest(),
               "manifest sha mismatch")
        print("[c] determinism: two generator runs byte-identical; manifest "
              "sha matches; anchors and route captures present")
        perturbed = dict(build_tensors(layers))
        key = "language_model.model.embed_tokens.weight"
        value = perturbed[key].copy()
        value[3, 0] = np.uint16(value[3, 0] ^ 0x100)
        perturbed[key] = value
        perturbed_config = os.path.join(workspace, "perturbed")
        write_checkpoint(perturbed_config, perturbed, config)
        out_c = os.path.join(workspace, "run_c")
        run_generator(perturbed_config, header, prompts, out_c)
        fixture_c = os.path.join(out_c, "k3", "synth_k3.t1r")
        expect(os.path.exists(fixture_c), "perturbed run produced no fixture")
        expect(open(fixture_a, "rb").read() != open(fixture_c, "rb").read(),
               "perturbed weight did not change the output")
        _, arrays_c = read_fixture(fixture_c)
        expect(not np.array_equal(arrays["pos0000_layer0000_streams"],
                                  arrays_c["pos0000_layer0000_streams"]),
               "embedding perturbation must change the layer streams")
        print("[c] negative control: one flipped bf16 bit in the prompt "
              "token's embedding row changed every stream and the fixture "
              "bytes")
        bad_header = os.path.join(workspace, "llm_defines_bad.h")
        with open(header) as fh:
            text = fh.read()
        with open(bad_header, "w") as fh:
            fh.write(text.replace("SPARK_LLM_MLA_LATENT_DIMENSION          8u",
                                  "SPARK_LLM_MLA_LATENT_DIMENSION          9u"))
        mismatch = run_generator(checkpoint, bad_header, prompts,
                                 os.path.join(workspace, "run_d"))
        expect(mismatch.returncode != 0,
               "defines/config disagreement must fail loud")
        expect("MLA_LATENT_DIMENSION" in mismatch.stderr,
               f"failure must name the mismatched define: {mismatch.stderr}")
        print("[c] negative control: defines/config mismatch fails closed "
              "naming SPARK_LLM_MLA_LATENT_DIMENSION")
        shutil.rmtree(workspace, ignore_errors=True)
        print("PASS t1_reference_k3: kda oracle cross-check, hand-computed "
              "layer 0, determinism, manifest sha, negative controls, "
              "defines/config fail-closed")
        return 0
    except AssertionError:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
