import hashlib
import importlib.util
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

from t1_reference_common import read_fixture  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "donor_t1_test", os.path.join(ROOT, "tests", "test_t1_reference_decoder.py"))
_donor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_donor)
bf16, f32, write_safetensors = (_donor.bf16, _donor.f32,
                                _donor.write_safetensors)


def write_header(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def write_json(path, document):
    with open(path, "w") as fh:
        json.dump(document, fh)


def run_generator(family, checkpoint, header, prompts, output):
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
         "--family", family, "--checkpoint", checkpoint,
         "--header", header, "--prompts", prompts, "--output", output],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"{family} generator failed: {result.stderr}")
    return result


def compare(reference, candidate):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_compare.py"),
         "compare", "--reference", reference, "--candidate", candidate],
        capture_output=True, text=True)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


LING_DEFINES = """#pragma once

#define SPARK_LLM_FAMILY_TAG                    ling
#define SPARK_LLM_HIDDEN_DIMENSION              16u
#define SPARK_LLM_LAYER_COUNT                   4u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            32u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          31u
#define SPARK_LLM_ATTENTION_PERIOD              4u
#define SPARK_LLM_GLOBAL_ATTENTION_PHASE        3u
#define SPARK_LLM_MLA_LAYER_COUNT               1u
#define SPARK_LLM_KDA_LAYER_COUNT               3u
#define SPARK_LLM_MLA_HEAD_COUNT                2u
#define SPARK_LLM_MLA_LATENT_DIMENSION          8u
#define SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION    4u
#define SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION    2u
#define SPARK_LLM_MLA_VALUE_HEAD_DIMENSION      4u
#define SPARK_LLM_MLA_ROPE_THETA                10000.0f
#define SPARK_LLM_MLA_QK_SCALE                  0.40824830532073975f
#define SPARK_LLM_KDA_CONV_KERNEL               4u
#define SPARK_LLM_KDA_GATE_LOWER_BOUND          -5.0f
#define SPARK_LLM_MOE_EXPERT_COUNT              4u
#define SPARK_LLM_MOE_TOP_K                     2u
#define SPARK_LLM_MOE_ROUTER_GROUP_COUNT        2u
#define SPARK_LLM_MOE_ROUTER_TOP_GROUPS         1u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     2.5f
#define SPARK_LLM_MOE_INTERMEDIATE_DIMENSION    4u
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  8u
#define SPARK_LLM_FIRST_ROUTED_LAYER            2u
"""

GEMMA4_DEFINES = """#pragma once

#define SPARK_LLM_FAMILY_TAG                    gemma4
#define SPARK_LLM_HIDDEN_DIMENSION              16u
#define SPARK_LLM_LAYER_COUNT                   6u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            32u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          31u
#define SPARK_LLM_EMBED_SCALE                   1.0f
#define SPARK_LLM_SLIDING_QUERY_HEAD_COUNT      2u
#define SPARK_LLM_SLIDING_KV_HEAD_COUNT         2u
#define SPARK_LLM_SLIDING_HEAD_DIMENSION        4u
#define SPARK_LLM_SLIDING_WINDOW_TOKENS         1024u
#define SPARK_LLM_SLIDING_ROPE_THETA            10000.0f
#define SPARK_LLM_FULL_QUERY_HEAD_COUNT         2u
#define SPARK_LLM_FULL_KV_HEAD_COUNT            1u
#define SPARK_LLM_FULL_HEAD_DIMENSION           8u
#define SPARK_LLM_FULL_LAYER_PERIOD             6u
#define SPARK_LLM_FULL_LAYER_PHASE              5u
#define SPARK_LLM_FULL_ROPE_BASE                1000000.0f
#define SPARK_LLM_FULL_ROTATED_PAIR_COUNT       2u
#define SPARK_LLM_QK_SCALE                      1.0f
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  8u
"""

GEMMA4_MOE_DEFINES = GEMMA4_DEFINES + """#define SPARK_LLM_ROUTED_EXPERT_COUNT           4u
#define SPARK_LLM_EXPERTS_PER_TOKEN             2u
#define SPARK_LLM_EXPERT_INTERMEDIATE_DIMENSION 4u
"""

QWEN38_MAX_DEFINES = """#pragma once

#define SPARK_LLM_FAMILY_TAG                    qwen38_max
#define SPARK_LLM_HIDDEN_DIMENSION              16u
#define SPARK_LLM_LAYER_COUNT                   4u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            32u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          31u
#define SPARK_LLM_ATTN_PERIOD                   4u
#define SPARK_LLM_FULL_ATTENTION_PHASE          3u
#define SPARK_LLM_ATTN_HEAD_COUNT               2u
#define SPARK_LLM_KV_HEAD_COUNT                 1u
#define SPARK_LLM_HEAD_DIMENSION                8u
#define SPARK_LLM_ROPE_DIMENSION                2u
#define SPARK_LLM_ROPE_THETA                    10000.0f
#define SPARK_LLM_GDN_KEY_HEAD_COUNT            2u
#define SPARK_LLM_GDN_VALUE_HEAD_COUNT          4u
#define SPARK_LLM_GDN_CONV_KERNEL               4u
#define SPARK_LLM_KDA_HEAD_KEY_DIMENSION        4u
#define SPARK_LLM_KDA_HEAD_VALUE_DIMENSION      4u
#define SPARK_LLM_ROUTED_EXPERT_COUNT           4u
#define SPARK_LLM_EXPERTS_PER_TOKEN             2u
#define SPARK_LLM_EXPERT_INTERMEDIATE_DIMENSION 16u
#define SPARK_LLM_SHARED_EXPERT_COUNT           1u
"""

LAGUNA_DEFINES = """#pragma once

#define SPARK_LLM_FAMILY_TAG                    laguna
#define SPARK_LLM_HIDDEN_DIMENSION              16u
#define SPARK_LLM_LAYER_COUNT                   4u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            32u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          31u
#define SPARK_LLM_ATTENTION_HEAD_DIMENSION      4u
#define SPARK_LLM_ATTENTION_KV_HEAD_COUNT       2u
#define SPARK_LLM_Q_HEAD_COUNT_FULL             2u
#define SPARK_LLM_Q_HEAD_COUNT_SLIDING          4u
#define SPARK_LLM_SLIDING_WINDOW                2u
#define SPARK_LLM_ATTENTION_SCALE               0.5f
#define SPARK_LLM_ROPE_FULL_THETA               5e5f
#define SPARK_LLM_ROPE_FULL_FACTOR              128.0f
#define SPARK_LLM_ROPE_FULL_ORIGINAL_POSITIONS  8192.0f
#define SPARK_LLM_ROPE_FULL_BETA_FAST           32.0f
#define SPARK_LLM_ROPE_FULL_BETA_SLOW           1.0f
#define SPARK_LLM_ROPE_FULL_ATTENTION_FACTOR    1.4852030263919618f
#define SPARK_LLM_ROPE_FULL_ROTARY_DIMENSION    4u
#define SPARK_LLM_ROPE_SLIDING_THETA            1e4f
#define SPARK_LLM_ROPE_SLIDING_ROTARY_DIMENSION 4u
#define SPARK_LLM_MOE_EXPERT_COUNT              4u
#define SPARK_LLM_MOE_TOP_K                     2u
#define SPARK_LLM_MOE_INTERMEDIATE_DIMENSION    4u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     2.5f
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  8u
"""


def ling_config():
    return {
        "hidden_size": 16, "num_hidden_layers": 4, "vocab_size": 32,
        "rms_norm_eps": 1e-6, "eos_token_id": 31, "layer_group_size": 4,
        "num_attention_heads": 2, "qk_nope_head_dim": 4, "qk_rope_head_dim": 2,
        "v_head_dim": 4, "kv_lora_rank": 8, "rope_theta": 10000.0,
        "head_dim": 4, "short_conv_kernel_size": 4, "kda_lower_bound": -5.0,
        "num_experts": 4, "num_experts_per_tok": 2, "n_group": 2,
        "topk_group": 1, "routed_scaling_factor": 2.5,
        "moe_intermediate_size": 4, "intermediate_size": 8,
        "first_k_dense_replace": 2,
    }


def ling_tensors():
    rng = np.random.default_rng(11)
    t = {}
    t["model.embed_tokens.weight"] = bf16("e", (32, 16), rng)
    t["model.norm.weight"] = bf16("n", (16,), rng)
    t["lm_head.weight"] = bf16("lm", (32, 16), rng, scale=0.5)
    for layer in range(4):
        p = f"model.layers.{layer}."
        t[p + "input_layernorm.weight"] = bf16("il", (16,), rng)
        t[p + "post_attention_layernorm.weight"] = bf16("pl", (16,), rng)
        if layer != 3:
            for kind in ("q", "k", "v"):
                t[p + f"attention.{kind}_proj.weight"] = bf16(kind, (8, 16), rng)
                t[p + f"attention.{kind}_conv1d.weight"] = \
                    bf16(kind, (8, 4), rng, scale=0.5)
            t[p + "attention.b_proj.weight"] = bf16("b", (2, 16), rng)
            t[p + "attention.A_log"] = f32("a", (2,), rng)
            t[p + "attention.dt_bias"] = f32("d", (8,), rng)
            t[p + "attention.f_proj.weight"] = bf16("f", (8, 16), rng)
            t[p + "attention.g_proj.weight"] = bf16("g", (8, 16), rng)
            t[p + "attention.o_norm.weight"] = bf16("on", (4,), rng)
            t[p + "attention.o_proj.weight"] = bf16("op", (16, 8), rng)
        else:
            t[p + "attention.q_proj.weight"] = bf16("q", (12, 16), rng)
            t[p + "attention.kv_a_proj_with_mqa.weight"] = bf16("ka", (10, 16), rng)
            t[p + "attention.kv_a_layernorm.weight"] = bf16("kn", (8,), rng)
            t[p + "attention.kv_b_proj.weight"] = bf16("kb", (16, 8), rng)
            t[p + "attention.g_proj.weight"] = bf16("g", (2, 16), rng)
            t[p + "attention.dense.weight"] = bf16("dn", (16, 8), rng)
        if layer < 2:
            t[p + "mlp.gate_proj.weight"] = bf16("g", (8, 16), rng)
            t[p + "mlp.up_proj.weight"] = bf16("u", (8, 16), rng)
            t[p + "mlp.down_proj.weight"] = bf16("dn", (16, 8), rng)
        else:
            t[p + "mlp.gate.weight"] = bf16("gw", (4, 16), rng, scale=0.5)
            t[p + "mlp.gate.expert_bias"] = bf16("eb", (4,), rng)
            for e in range(4):
                ep = p + f"mlp.experts.{e}."
                t[ep + "gate_proj.weight"] = bf16("g", (4, 16), rng)
                t[ep + "up_proj.weight"] = bf16("u", (4, 16), rng)
                t[ep + "down_proj.weight"] = bf16("dn", (16, 4), rng)
            sp = p + "mlp.shared_experts."
            t[sp + "gate_proj.weight"] = bf16("g", (4, 16), rng)
            t[sp + "up_proj.weight"] = bf16("u", (4, 16), rng)
            t[sp + "down_proj.weight"] = bf16("dn", (16, 4), rng)
    return t


def gemma4_tensors():
    rng = np.random.default_rng(13)
    t = {}
    t["model.language_model.embed_tokens.weight"] = bf16("e", (32, 16), rng)
    t["model.language_model.norm.weight"] = bf16("n", (16,), rng)
    for layer in range(6):
        p = f"model.language_model.layers.{layer}."
        full = layer == 5
        dim = 8 if full else 4
        t[p + "input_layernorm.weight"] = bf16("il", (16,), rng)
        t[p + "post_attention_layernorm.weight"] = bf16("pl", (16,), rng)
        t[p + "pre_feedforward_layernorm.weight"] = bf16("pf", (16,), rng)
        t[p + "post_feedforward_layernorm.weight"] = bf16("df", (16,), rng)
        t[p + "layer_scalar"] = f32("s", (1,), rng, scale=1.0)
        t[p + "self_attn.q_proj.weight"] = bf16("q", (2 * dim, 16), rng)
        kv_rows = 2 * dim if not full else dim
        t[p + "self_attn.k_proj.weight"] = bf16("k", (kv_rows, 16), rng)
        t[p + "self_attn.v_proj.weight"] = bf16("v", (kv_rows, 16), rng)
        t[p + "self_attn.q_norm.weight"] = bf16("qn", (dim,), rng)
        t[p + "self_attn.k_norm.weight"] = bf16("kn", (dim,), rng)
        t[p + "self_attn.o_proj.weight"] = bf16("op", (16, 2 * dim), rng)
        t[p + "mlp.gate_proj.weight"] = bf16("g", (8, 16), rng)
        t[p + "mlp.up_proj.weight"] = bf16("u", (8, 16), rng)
        t[p + "mlp.down_proj.weight"] = bf16("dn", (16, 8), rng)
    return t


def gemma4_config():
    return {
        "hidden_size": 16, "num_hidden_layers": 6, "vocab_size": 32,
        "rms_norm_eps": 1e-6, "eos_token_id": 31, "num_key_value_heads": 2,
        "head_dim": 4, "intermediate_size": 8,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    }


def gemma4_moe_config():
    config = gemma4_config()
    config.update({
        "enable_moe_block": True, "num_experts": 4, "top_k_experts": 2,
        "moe_intermediate_size": 4,
    })
    return config


def gemma4_moe_tensors():
    rng = np.random.default_rng(23)
    t = gemma4_tensors()
    fresh = np.random.default_rng(29)
    for layer in range(6):
        p = f"model.language_model.layers.{layer}."
        t[p + "experts.gate_up_proj"] = bf16("egu", (4, 8, 16), fresh)
        t[p + "experts.down_proj"] = bf16("edn", (4, 16, 4), fresh)
        t[p + "router.proj.weight"] = bf16("rp", (4, 16), fresh)
        t[p + "router.scale"] = bf16("rs", (16,), fresh)
        t[p + "router.per_expert_scale"] = bf16("pes", (4,), fresh)
        t[p + "post_feedforward_layernorm_1.weight"] = bf16("df1", (16,), fresh)
        t[p + "pre_feedforward_layernorm_2.weight"] = bf16("pf2", (16,), fresh)
        t[p + "post_feedforward_layernorm_2.weight"] = bf16("df2", (16,), fresh)
    return t


def laguna_tensors():
    rng = np.random.default_rng(17)
    t = {}
    t["model.embed_tokens.weight"] = bf16("e", (32, 16), rng)
    t["model.norm.weight"] = bf16("n", (16,), rng)
    t["lm_head.weight"] = bf16("lm", (32, 16), rng, scale=0.5)
    for layer in range(4):
        p = f"model.layers.{layer}."
        full = layer == 3
        q_dim = 2 * 4 if full else 4 * 4
        t[p + "input_layernorm.weight"] = bf16("il", (16,), rng)
        t[p + "post_attention_layernorm.weight"] = bf16("pl", (16,), rng)
        t[p + "self_attn.q_proj.weight"] = bf16("q", (q_dim, 16), rng)
        t[p + "self_attn.k_proj.weight"] = bf16("k", (8, 16), rng)
        t[p + "self_attn.v_proj.weight"] = bf16("v", (8, 16), rng)
        t[p + "self_attn.q_norm.weight"] = bf16("qn", (4,), rng)
        t[p + "self_attn.k_norm.weight"] = bf16("kn", (4,), rng)
        t[p + "self_attn.g_proj.weight"] = bf16("g", (2 if full else 4, 16), rng)
        t[p + "self_attn.o_proj.weight"] = bf16("op", (16, q_dim), rng)
        if layer == 0:
            t[p + "mlp.gate_proj.weight"] = bf16("g", (8, 16), rng)
            t[p + "mlp.up_proj.weight"] = bf16("u", (8, 16), rng)
            t[p + "mlp.down_proj.weight"] = bf16("dn", (16, 8), rng)
        else:
            t[p + "mlp.gate.weight"] = bf16("gw", (4, 16), rng, scale=0.5)
            for e in range(4):
                ep = p + f"mlp.experts.{e}."
                t[ep + "gate_proj.weight"] = bf16("g", (4, 16), rng)
                t[ep + "up_proj.weight"] = bf16("u", (4, 16), rng)
                t[ep + "down_proj.weight"] = bf16("dn", (16, 4), rng)
            sp = p + "mlp.shared_expert."
            t[sp + "gate_proj.weight"] = bf16("g", (4, 16), rng)
            t[sp + "up_proj.weight"] = bf16("u", (4, 16), rng)
            t[sp + "down_proj.weight"] = bf16("dn", (16, 4), rng)
    return t


def laguna_config():
    return {
        "hidden_size": 16, "num_hidden_layers": 4, "vocab_size": 32,
        "rms_norm_eps": 1e-6, "eos_token_id": [31], "head_dim": 4,
        "num_key_value_heads": 2, "sliding_window": 2, "num_experts": 4,
        "num_experts_per_tok": 2, "moe_intermediate_size": 4,
        "moe_routed_scaling_factor": 2.5, "intermediate_size": 8,
        "mlp_only_layers": [0],
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
        "rope_parameters": {"full_attention": {"rope_theta": 500000.0,
                                               "factor": 128.0,
                                               "original_max_position_embeddings": 8192.0,
                                               "beta_fast": 32.0, "beta_slow": 1.0,
                                               "attention_factor": 1.4852030263919618}},
    }


def qwen38_max_config():
    return {
        "hidden_size": 16, "num_hidden_layers": 4, "vocab_size": 32,
        "rms_norm_eps": 1e-6, "eos_token_id": 31,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_theta": 10000.0},
        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "linear_key_head_dim": 4, "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2, "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    }


def typed_u8(array, dtype):
    return (array, dtype)


def qwen38_max_tensors():
    rng = np.random.default_rng(31)
    t = {}
    t["model.embed_tokens.weight"] = bf16("e", (32, 16), rng)
    t["model.norm.weight"] = bf16("n", (16,), rng)
    head = bf16("lm", (32, 16), rng, scale=0.5)
    head[31] = np.zeros(16, dtype=np.uint16)
    t["lm_head.weight"] = head
    for layer in range(4):
        p = f"model.layers.{layer}."
        t[p + "input_layernorm.weight"] = bf16("il", (16,), rng)
        t[p + "post_attention_layernorm.weight"] = bf16("pl", (16,), rng)
        t[p + "mlp.gate.weight"] = bf16("gw", (4, 16), rng, scale=0.5)
        t[p + "mlp.shared_expert_gate.weight"] = bf16("sg", (1, 16), rng)
        t[p + "mlp.shared_expert.gate_proj.weight"] = bf16("g", (16, 16), rng)
        t[p + "mlp.shared_expert.up_proj.weight"] = bf16("u", (16, 16), rng)
        t[p + "mlp.shared_expert.down_proj.weight"] = bf16("dn", (16, 16), rng)
        for e in range(4):
            ep = p + f"mlp.experts.{e}."
            for kind in ("gate_proj", "up_proj", "down_proj"):
                t[ep + f"{kind}.weight"] = typed_u8(
                    rng.integers(0, 256, (16, 8), dtype=np.uint8), "U8")
                t[ep + f"{kind}.weight_scale"] = typed_u8(
                    rng.integers(0, 64, (16, 1), dtype=np.uint8), "F8_E4M3")
                t[ep + f"{kind}.weight_scale_2"] = (
                    np.array(0.02, dtype=np.float32), "F32")
        if layer < 3:
            t[p + "linear_attn.in_proj_qkv.weight"] = bf16("qkv", (32, 16), rng)
            t[p + "linear_attn.in_proj_a.weight"] = bf16("ia", (4, 16), rng)
            t[p + "linear_attn.in_proj_b.weight"] = bf16("ib", (4, 16), rng)
            t[p + "linear_attn.in_proj_z.weight"] = bf16("z", (16, 16), rng)
            t[p + "linear_attn.out_proj.weight"] = bf16("op", (16, 16), rng)
            t[p + "linear_attn.conv1d.weight"] = bf16("cv", (32, 1, 4), rng)
            t[p + "linear_attn.A_log"] = bf16("al", (4,), rng, scale=0.01)
            t[p + "linear_attn.dt_bias"] = bf16("db", (4,), rng, scale=0.01)
            t[p + "linear_attn.norm.weight"] = bf16("nw", (4,), rng)
        else:
            t[p + "self_attn.q_proj.weight"] = bf16("q", (32, 16), rng)
            t[p + "self_attn.k_proj.weight"] = bf16("k", (8, 16), rng)
            t[p + "self_attn.v_proj.weight"] = bf16("v", (8, 16), rng)
            t[p + "self_attn.q_norm.weight"] = bf16("qn", (8,), rng)
            t[p + "self_attn.k_norm.weight"] = bf16("kn", (8,), rng)
            t[p + "self_attn.o_proj.weight"] = bf16("op", (16, 16), rng)
    return t


def write_safetensors_typed(path, tensors):
    header = {}
    offset = 0
    blobs = []
    tags = {"BF16": ("BF16", 2), "F32": ("F32", 4), "U8": ("U8", 1),
            "F8_E4M3": ("F8_E4M3", 1)}
    for name in sorted(tensors):
        value = tensors[name]
        array, tag = value if isinstance(value, tuple) else (value, None)
        if tag is None:
            tag = "BF16" if array.dtype == np.uint16 else "F32"
        dtype, itemsize = tags[tag]
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


FAMILIES = [
    ("ling", ling_tensors, ling_config(), LING_DEFINES, "ling"),
    ("gemma4", gemma4_tensors, gemma4_config(), GEMMA4_DEFINES, "gemma4"),
    ("gemma4_moe", gemma4_moe_tensors, gemma4_moe_config(), GEMMA4_MOE_DEFINES,
     "gemma4"),
    ("laguna", laguna_tensors, laguna_config(), LAGUNA_DEFINES, "laguna"),
    ("qwen38_max", qwen38_max_tensors, qwen38_max_config(), QWEN38_MAX_DEFINES,
     "qwen38_max"),
]

TYPED_WRITERS = {"qwen38_max": write_safetensors_typed}


def check_family(workspace, family, tensors, config, defines, engine):
    checkpoint = os.path.join(workspace, f"{family}_checkpoint")
    os.makedirs(checkpoint, exist_ok=True)
    (TYPED_WRITERS.get(family, write_safetensors))(
        os.path.join(checkpoint, "model.safetensors"), tensors)
    write_json(os.path.join(checkpoint, "config.json"), config)
    header = os.path.join(workspace, f"{family}_defines.h")
    write_header(header, defines)
    prompts = os.path.join(workspace, f"{family}_prompts.json")
    write_json(prompts, {"prompts": [{
        "name": "synth",
        "prompt_token_ids": {"ling": [3, 7, 11],
                             "gemma4": [5, 9, 13],
                             "gemma4_moe": [5, 9, 13],
                             "laguna": [5, 9, 13],
                             "qwen38_max": [5, 9, 13]}[family],
        "new_tokens": 2,
        "capture_layers": [0, config["num_hidden_layers"] - 1],
    }]})
    out_a = os.path.join(workspace, f"{family}_a")
    out_b = os.path.join(workspace, f"{family}_b")
    run_generator(engine, checkpoint, header, prompts, out_a)
    run_generator(engine, checkpoint, header, prompts, out_b)
    fixture_a = os.path.join(out_a, engine, "synth.t1r")
    fixture_b = os.path.join(out_b, engine, "synth.t1r")
    expect(os.path.exists(fixture_a), f"{family}: fixture missing")
    _, arrays = read_fixture(fixture_a)
    expect("pos0000_layer0000_streams" in arrays, f"{family}: anchor missing")
    expect(any(name.endswith("_route_ids") for name in arrays) or
           family in ("gemma4", "gemma4_moe"), f"{family}: route capture missing")
    expect(open(fixture_a, "rb").read() == open(fixture_b, "rb").read(),
           f"{family}: generator is not byte-deterministic")
    manifest = json.load(open(os.path.join(out_a, engine, "MANIFEST.json")))
    digest = hashlib.sha256(open(fixture_a, "rb").read()).hexdigest()
    expect(manifest["fixtures"]["synth.t1r"]["sha256"] == digest,
           f"{family}: manifest sha mismatch")
    identical = compare(fixture_a, fixture_b)
    expect(identical.returncode == 0,
           f"{family}: identical fixtures must PASS: {identical.stdout}")
    corrupted = os.path.join(workspace, f"{family}_corrupt.t1r")
    corrupt = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_compare.py"),
         "corrupt-fixture", "--source", fixture_a, "--target", corrupted,
         "--array", "pos0000_layer0000_streams", "--offset", "5"],
        capture_output=True, text=True)
    expect(corrupt.returncode == 0, f"{family}: corrupt-fixture failed")
    diverged = compare(fixture_a, corrupted)
    expect(diverged.returncode == 1, f"{family}: corrupted fixture must FAIL")
    expect("pos0000_layer0000_streams" in diverged.stdout,
           f"{family}: FAIL must name the corrupted array")
    with open(header) as fh:
        text = fh.read()
    bad = os.path.join(workspace, f"{family}_bad.h")
    broken = text.replace("SPARK_LLM_HIDDEN_DIMENSION              16u",
                          "SPARK_LLM_HIDDEN_DIMENSION              17u")
    expect(broken != text, f"{family}: hidden define not found in header")
    write_header(bad, broken)
    mismatch = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
         "--family", engine, "--checkpoint", checkpoint, "--header", bad,
         "--prompts", prompts, "--output", os.path.join(workspace, f"{family}_c")],
        capture_output=True, text=True)
    expect(mismatch.returncode != 0,
           f"{family}: defines/config disagreement must fail loud")
    expect("HIDDEN_DIMENSION" in mismatch.stderr,
           f"{family}: failure must name the mismatched define")
    if family == "gemma4_moe":
        with open(header) as fh:
            moe_text = fh.read()
        bad_moe = os.path.join(workspace, f"{family}_bad_moe.h")
        broken_moe = moe_text.replace(
            "SPARK_LLM_ROUTED_EXPERT_COUNT           4u",
            "SPARK_LLM_ROUTED_EXPERT_COUNT           5u")
        expect(broken_moe != moe_text,
               f"{family}: routed expert define not found in header")
        write_header(bad_moe, broken_moe)
        moe_mismatch = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
             "--family", engine, "--checkpoint", checkpoint,
             "--header", bad_moe, "--prompts", prompts,
             "--output", os.path.join(workspace, f"{family}_d")],
            capture_output=True, text=True)
        expect(moe_mismatch.returncode != 0,
               f"{family}: MoE defines/config disagreement must fail loud")
        expect("ROUTED_EXPERT_COUNT" in moe_mismatch.stderr,
               f"{family}: failure must name the mismatched MoE define")


def main():
    workspace = tempfile.mkdtemp(prefix="t1ref-engines-")
    try:
        for family, tensor_fn, config, defines, engine in FAMILIES:
            check_family(workspace, family, tensor_fn(), config, defines,
                         engine)
            print(f"PASS {family}: determinism, manifest sha, negative control, "
                  "defines/config fail-closed")
        shutil.rmtree(workspace, ignore_errors=True)
        return 0
    except AssertionError:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
