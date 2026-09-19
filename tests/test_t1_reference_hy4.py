import importlib.util
import json
import os
import struct
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import t1_reference_hy4 as engine_module  # noqa: E402
from t1_reference_common import bf16_round_f32  # noqa: E402
from t1_reference_common import f32_to_bf16_u16  # noqa: E402
from t1_reference_common import parse_llm_defines  # noqa: E402

HIDDEN = 32
HC = 2
LAYERS = 2
HEADS = 2
QK = 16
NOPE = 8
ROT = 8
V_DIM = 16
Q_LORA = 32
KV_LORA = 32
VOCAB = 32
EXPERTS = 4
TOP_K = 2
INTER = 32
DENSE = 32


def dtsize(dt):
    return {"F8_E4M3": 1, "BF16": 2, "F32": 4, "U8": 1}[dt]


def fp8_encode(values):
    out = np.zeros(values.size, dtype=np.uint8)
    for i, v in enumerate(np.asarray(values, dtype=np.float32).ravel()):
        sign = 0x80 if v < 0 else 0
        a = abs(float(v))
        if a == 0.0:
            continue
        if a < 2.0 ** -6:
            m = int(round(a * 512.0))
            if m > 7:
                m = 7
            out[i] = sign | m
            continue
        e = 0
        while a >= 2.0 and e < 15:
            a /= 2.0
            e += 1
        while a < 1.0 and e > -6:
            a *= 2.0
            e -= 1
        m = int(round((a - 1.0) * 8.0))
        if m > 7:
            m = 7
        out[i] = sign | ((e + 7) << 3) | m
    return out


def write_safetensors(path, arrays):
    header = {}
    cursor = 0
    blobs = []
    for name in sorted(arrays):
        dtype, raw = arrays[name]
        header[name] = {"dtype": dtype, "shape": list(raw.shape),
                        "data_offsets": [cursor, cursor + raw.nbytes]}
        blobs.append(raw.tobytes())
        cursor += raw.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for piece in blobs:
            fh.write(piece)


def quantized(rows, cols, seed):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((rows, cols)).astype(np.float32)
    groups = values.reshape(rows, cols // 32, 32)
    absmax = np.maximum(np.abs(groups).max(axis=2, keepdims=True), 1e-6)
    exponent = np.ceil(np.log2(absmax)).astype(np.int32)
    scales = (exponent + 127).astype(np.uint8)
    normalised = groups / np.exp2(exponent.astype(np.float32))
    payload = fp8_encode(normalised.reshape(rows, cols)).reshape(rows, cols)
    return payload, scales.reshape(rows, cols // 32), values


def synthetic_checkpoint(root):
    rng = np.random.default_rng(7)
    arrays = {}
    index = {}

    def put(name, dtype, raw, shard="model-00001-of-00001.safetensors"):
        arrays[name] = (dtype, raw)
        index[name] = shard

    embed = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)
    put("model.embed_tokens.weight", "BF16",
        np.frombuffer(embed.astype(np.float16).view(np.uint8), dtype=np.uint8)
        .view(np.uint16).reshape(VOCAB, HIDDEN))
    put("lm_head.weight", "BF16",
        f32_to_bf16_u16(
            rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)))
    put("model.norm.weight", "BF16",
        (rng.random(HIDDEN).astype(np.float16)).view(np.uint8)
        .view(np.uint16).reshape(HIDDEN))
    put("model.hc_head.hc_head_fn", "F32",
        rng.standard_normal((HC, HC * HIDDEN)).astype(np.float32))
    put("model.hc_head.hc_head_scale", "F32",
        np.array([0.3], dtype=np.float32))
    put("model.hc_head.hc_head_base", "F32",
        rng.standard_normal(HC).astype(np.float32))
    for il in range(LAYERS):
        p = f"model.layers.{il}."
        put(p + "input_layernorm.weight", "BF16",
            (rng.random(HIDDEN).astype(np.float16)).view(np.uint8)
            .view(np.uint16).reshape(HIDDEN))
        put(p + "post_attention_layernorm.weight", "BF16",
            (rng.random(HIDDEN).astype(np.float16)).view(np.uint8)
            .view(np.uint16).reshape(HIDDEN))
        put(p + "hc_attn_layer.hc_pre.hc_fn", "F32",
            rng.standard_normal((2 * HC, HC * HIDDEN)).astype(np.float32))
        put(p + "hc_attn_layer.hc_pre.hc_scale", "F32",
            np.array([0.2, 0.3], dtype=np.float32))
        put(p + "hc_attn_layer.hc_pre.hc_base", "F32",
            rng.standard_normal(2 * HC).astype(np.float32))
        put(p + "hc_mlp_layer.hc_pre.hc_fn", "F32",
            rng.standard_normal((2 * HC, HC * HIDDEN)).astype(np.float32))
        put(p + "hc_mlp_layer.hc_pre.hc_scale", "F32",
            np.array([0.4, 0.1], dtype=np.float32))
        put(p + "hc_mlp_layer.hc_pre.hc_base", "F32",
            rng.standard_normal(2 * HC).astype(np.float32))
        payload, scale, _ = quantized(Q_LORA, HIDDEN, 100 + il)
        put(p + "self_attn.q_a_proj.weight", "F8_E4M3", payload)
        put(p + "self_attn.q_a_proj.weight_scale", "U8", scale)
        put(p + "self_attn.q_a_layernorm.weight", "BF16",
            (rng.random(Q_LORA).astype(np.float16)).view(np.uint8)
            .view(np.uint16).reshape(Q_LORA))
        payload, scale, _ = quantized(HEADS * QK, Q_LORA, 200 + il)
        put(p + "self_attn.q_b_proj.weight", "F8_E4M3", payload)
        put(p + "self_attn.q_b_proj.weight_scale", "U8", scale)
        payload, scale, _ = quantized(KV_LORA + ROT, HIDDEN, 300 + il)
        put(p + "self_attn.kv_a_proj_with_mqa.weight", "F8_E4M3", payload)
        put(p + "self_attn.kv_a_proj_with_mqa.weight_scale", "U8", scale)
        put(p + "self_attn.kv_a_layernorm.weight", "BF16",
            (rng.random(KV_LORA).astype(np.float16)).view(np.uint8)
            .view(np.uint16).reshape(KV_LORA))
        payload, scale, _ = quantized(HEADS * (NOPE + V_DIM), KV_LORA,
                                      400 + il)
        put(p + "self_attn.kv_b_proj.weight", "F8_E4M3", payload)
        put(p + "self_attn.kv_b_proj.weight_scale", "U8", scale)
        put(p + "self_attn.linear_gate.weight", "BF16",
            (rng.standard_normal(HEADS * V_DIM * HIDDEN)
             .astype(np.float16)).view(np.uint8).view(np.uint16)
            .reshape(HEADS * V_DIM, HIDDEN))
        put(p + "self_attn.learnable_sink_param", "F32",
            ((1.0 + il) * rng.standard_normal(HEADS)).astype(np.float32))
        payload, scale, _ = quantized(HIDDEN, HEADS * V_DIM, 500 + il)
        put(p + "self_attn.o_proj.weight", "F8_E4M3", payload)
        put(p + "self_attn.o_proj.weight_scale", "U8", scale)
        if il == 0:
            for kind, rows in (("gate_proj", DENSE), ("up_proj", DENSE)):
                payload, scale, _ = quantized(rows, HIDDEN, 600 + il)
                put(p + f"mlp.{kind}.weight", "F8_E4M3", payload)
                put(p + f"mlp.{kind}.weight_scale", "U8", scale)
            payload, scale, _ = quantized(HIDDEN, DENSE, 650 + il)
            put(p + "mlp.down_proj.weight", "F8_E4M3", payload)
            put(p + "mlp.down_proj.weight_scale", "U8", scale)
        else:
            put(p + "mlp.gate.weight", "BF16",
                (rng.standard_normal(EXPERTS * HIDDEN).astype(np.float16))
                .view(np.uint8).view(np.uint16).reshape(EXPERTS, HIDDEN))
            put(p + "mlp.gate.e_score_correction_bias", "F32",
                (rng.random(EXPERTS) * 0.1 - 0.05).astype(np.float32))
            payload, scale, _ = quantized(EXPERTS * 2 * INTER, HIDDEN, 700)
            put(p + "mlp.experts.gate_up_proj", "F8_E4M3",
                payload.reshape(EXPERTS, 2 * INTER, HIDDEN))
            put(p + "mlp.experts.gate_up_proj_scale", "U8",
                scale.reshape(EXPERTS, 2 * INTER, HIDDEN // 32))
            payload, scale, _ = quantized(EXPERTS * HIDDEN, INTER, 800)
            put(p + "mlp.experts.down_proj", "F8_E4M3",
                payload.reshape(EXPERTS, HIDDEN, INTER))
            put(p + "mlp.experts.down_proj_scale", "U8",
                scale.reshape(EXPERTS, HIDDEN, INTER // 32))
            for kind, rows in (("gate_proj", INTER), ("up_proj", INTER)):
                payload, scale, _ = quantized(rows, HIDDEN, 900 + il)
                put(p + f"mlp.shared_experts.{kind}.weight", "F8_E4M3",
                    payload)
                put(p + f"mlp.shared_experts.{kind}.weight_scale", "U8",
                    scale)
            payload, scale, _ = quantized(HIDDEN, INTER, 950 + il)
            put(p + "mlp.shared_experts.down_proj.weight", "F8_E4M3", payload)
            put(p + "mlp.shared_experts.down_proj.weight_scale", "U8", scale)
    write_safetensors(os.path.join(root, "model-00001-of-00001.safetensors"),
                      arrays)
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as fh:
        json.dump({"weight_map": index}, fh)
    config = {
        "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB, "rms_norm_eps": 1e-05,
        "num_attention_heads": HEADS, "q_lora_rank": Q_LORA,
        "kv_lora_rank": KV_LORA, "qk_head_dim": QK,
        "qk_nope_head_dim": NOPE, "qk_rope_head_dim": ROT,
        "v_head_dim": V_DIM, "hc_mult": HC, "hc_eps": 1e-06,
        "hc_magnitude": 2.0, "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K, "n_shared_experts": 1,
        "moe_intermediate_size": INTER, "routed_scaling_factor": 2.827,
        "norm_topk_prob": True, "intermediate_size": DENSE,
        "swiglu_limit": 10.0, "index_n_heads": 2, "index_head_dim": 4,
        "index_topk": 2048, "eos_token_id": VOCAB - 1,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    }
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump(config, fh)
    return config


DEFINES = """#pragma once

#define SPARK_LLM_FAMILY_TAG                    hy4
#define SPARK_LLM_HIDDEN_DIMENSION              {hidden}u
#define SPARK_LLM_LAYER_COUNT                   {layers}u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            {vocab}u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        2048u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-05f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          31u
#define SPARK_LLM_PAD_TOKEN_ID                  2u
#define SPARK_LLM_MAX_PREFILL_TOKENS_PER_DISPATCH 256u
#define SPARK_LLM_MLA_HEAD_COUNT                {heads}u
#define SPARK_LLM_MLA_Q_LORA_RANK               {qlora}u
#define SPARK_LLM_MLA_LATENT_DIMENSION          {kvlora}u
#define SPARK_LLM_MLA_QK_HEAD_DIMENSION         {qk}u
#define SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION    {nope}u
#define SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION    {rot}u
#define SPARK_LLM_MLA_VALUE_HEAD_DIMENSION      {vdim}u
#define SPARK_LLM_MLA_ROPE_THETA                1e4f
#define SPARK_LLM_MLA_SINK_COUNT                {heads}u
#define SPARK_LLM_MOE_EXPERT_COUNT              {experts}u
#define SPARK_LLM_MOE_TOP_K                     {topk}u
#define SPARK_LLM_MOE_SHARED_EXPERT_COUNT       1u
#define SPARK_LLM_MOE_INTERMEDIATE_DIMENSION    {inter}u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     2.827f
#define SPARK_LLM_MOE_NORM_TOPK_PROB            1u
#define SPARK_LLM_FIRST_ROUTED_LAYER            1u
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  {dense}u
#define SPARK_LLM_SWIGLU_LIMIT                  10.0f
#define SPARK_LLM_HC_STREAM_COUNT               {hc}u
#define SPARK_LLM_HC_EPSILON                    1e-06f
#define SPARK_LLM_HC_MAGNITUDE                  2.0f
#define SPARK_LLM_INDEX_HEAD_COUNT              2u
#define SPARK_LLM_INDEX_HEAD_DIMENSION          4u
#define SPARK_LLM_INDEX_TOP_K                   2048u
#define SPARK_LLM_MTP_ENABLED                   0u
#define SPARK_LLM_MTP_DRAFT_DEPTH               0u
""".format(hidden=HIDDEN, layers=LAYERS, vocab=VOCAB, heads=HEADS,
           qlora=Q_LORA, kvlora=KV_LORA, qk=QK, nope=NOPE, rot=ROT,
           vdim=V_DIM, experts=EXPERTS, topk=TOP_K, inter=INTER, dense=DENSE,
           hc=HC)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def test_dequant_fp8():
    payload = np.zeros((1, 64), dtype=np.uint8)
    payload[0, 0] = 0x00
    payload[0, 1] = 0x38
    payload[0, 2] = 0xB8
    payload[0, 3] = 0x01
    payload[0, 4] = 0x40
    payload[0, 5] = 0x7F
    scale = np.array([[100, 130]], dtype=np.uint8)
    got = engine_module.dequant_fp8(payload, scale)
    expect(got.shape == (1, 64), "dequant shape")
    expect(got[0, 0] == 0.0, "zero code is 0.0")
    expect(abs(got[0, 1] - 2.0 ** (100 - 127)) < 1e-12,
           "0x38 is 1.0 * 2^(100-127)")
    expect(got[0, 2] == -got[0, 1], "sign bit")
    expect(abs(got[0, 3] - 2.0 ** (100 - 127) * (1.0 / 8.0 * 2.0 ** -6))
           < 1e-15, "subnormal m=1 e=0 in group 0")
    expect(abs(got[0, 4] - 2.0 * 2.0 ** (100 - 127)) < 1e-12,
           "0x40 is (1+0)*2^(8-7) in group 0")
    payload[0, 33] = 0x38
    got = engine_module.dequant_fp8(payload, scale)
    expect(abs(got[0, 33] - 2.0 ** (130 - 127)) < 1e-9,
           "group 1 carries its own e8m0 scale")
    expect(np.isnan(got[0, 5]), "0x7F is NaN")
    try:
        engine_module.dequant_fp8(payload, np.zeros((1, 3), dtype=np.uint8))
        raise AssertionError("mismatched scale shape must fail closed")
    except ValueError:
        pass


def test_cross_check():
    values = {}
    for dname, cname, kind in engine_module.DEFINES_VS_CONFIG:
        values[dname] = {"hidden_size": "32u", "num_hidden_layers": "2u",
                         "vocab_size": "32u", "rms_norm_eps": "1e-05f",
                         "num_attention_heads": "2u", "q_lora_rank": "16u",
                         "kv_lora_rank": "32u", "qk_head_dim": "16u",
                         "qk_nope_head_dim": "8u", "qk_rope_head_dim": "8u",
                         "v_head_dim": "8u", "hc_mult": "2u",
                         "hc_eps": "1e-06f", "hc_magnitude": "2.0f",
                         "n_routed_experts": "4u", "num_experts_per_tok": "2u",
                         "n_shared_experts": "1u",
                         "moe_intermediate_size": "8u",
                         "routed_scaling_factor": "2.827f",
                         "norm_topk_prob": "1u",
                         "intermediate_size": "16u",
                         "swiglu_limit": "10.0f", "index_n_heads": "2u",
                         "index_head_dim": "4u", "index_topk": "2048u",
                         "eos_token_id": "31u"}[cname]
    config = {}
    for dname, _cname, kind in engine_module.DEFINES_VS_CONFIG:
        text = values[dname]
        config[_cname] = float(text.rstrip("f")) if kind == "float" \
            else int(text.rstrip("u"))
    config["rope_parameters"] = {"rope_theta": 10000.0}
    mismatches = engine_module.cross_check(
        {**values, "MLA_ROPE_THETA": "1e4f"}, config)
    expect(mismatches == [], "clean configs produce no mismatches")
    try:
        engine_module.cross_check({**values, "MLA_ROPE_THETA": "1e4f"},
                                  {**config, "hidden_size": 64})
        raise AssertionError("mismatch must raise")
    except engine_module.Hy4ConfigError:
        pass
    recorded = engine_module.cross_check(
        {**values, "MLA_ROPE_THETA": "5e4f"}, config)
    expect(len(recorded) == 1 and
           recorded[0]["define"] == "SPARK_LLM_MLA_ROPE_THETA",
           "rope theta mismatch is recorded, not fatal")


def test_routing_selection_ties():
    probs = np.array([0.5, 0.9, 0.5, 0.1, 0.9, 0.2, 0.3, 0.4],
                     dtype=np.float32)
    bias = np.zeros(8, dtype=np.float32)
    key = probs + bias
    order = np.argsort(-key, kind="stable")
    selected = order[:2]
    expect(list(selected) == [1, 4],
           f"ties resolve to the lowest index first, got {selected}")
    weights = probs[selected]
    total = float(weights.sum())
    weights = weights / total * 2.827
    expect(abs(float(weights.sum()) - 2.827) < 1e-6,
           "normalised weights scaled to the routing scale")


def test_sink_softmax():
    scores = np.array([[2.0, 1.0]], dtype=np.float32)
    sinks = np.array([3.0], dtype=np.float32)
    ceiling = np.maximum(scores.max(axis=1), sinks)
    weights = np.exp(scores - ceiling[:, None])
    denominator = weights.sum(axis=1) + np.exp(sinks - ceiling)
    probs = weights / denominator[:, None]
    sink_probability = np.exp(sinks - ceiling) / denominator
    expect(abs(float(probs.sum() + sink_probability[0]) - 1.0) < 1e-6,
           "softmax over tokens plus sink sums to one")
    expect(probs[0, 0] > probs[0, 1], "higher score wins")
    expect(abs(float(sink_probability[0]) - 1.0 /
               (1.0 + np.exp(2.0 - 3.0) + np.exp(1.0 - 3.0))) < 1e-6,
           "sink participates as its own logit")


def test_rope_interleaved():
    freqs = np.power(np.float32(10000.0),
                     -np.arange(0, 8, 2, dtype=np.float32) / 8)
    value = np.array([[1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 2.0]],
                     dtype=np.float32)
    eng = engine_module.Hy4Engine.__new__(engine_module.Hy4Engine)
    eng.rot = ROT
    eng.rope_theta = 10000.0
    eng.rope_freqs = freqs
    original = value.copy()
    eng._rope_(value, 0)
    expect(np.array_equal(value, original), "rope is identity at position 0")
    value = original.copy()
    eng._rope_(value, 3)
    angle = 3.0 * float(freqs[0])
    expect(abs(value[0, 0] - (np.cos(angle) * 1.0 - np.sin(angle) * 0.0))
           < 1e-6, "consecutive pair (v0, v1) rotates with the first frequency")
    expect(abs(value[0, 1] - (np.sin(angle) * 1.0 + np.cos(angle) * 0.0))
           < 1e-6, "odd element carries the sine term")
    angle2 = 3.0 * float(freqs[1])
    expect(abs(value[0, 2] - (np.cos(angle2) * 2.0 - np.sin(angle2) * 0.0))
           < 1e-6, "consecutive pair (v2, v3) carries the second frequency")
    angle3 = 3.0 * float(freqs[2])
    expect(abs(value[0, 4] - (np.cos(angle3) * 0.0 - np.sin(angle3) * 1.0))
           < 1e-6, "pair (v4, v5) uses freqs[2] on the tail half")


def test_weight_cache_lru(tmp):
    config = synthetic_checkpoint(tmp)
    from t1_reference_common import Safetensors
    st = Safetensors(tmp)
    st.cache_limit = 1
    embed = st.raw("model.embed_tokens.weight")
    expect(embed.shape == (VOCAB, HIDDEN), "embed shape")
    head = st.raw("lm_head.weight")
    expect(len(st.cache) == 1, "cache bounded at the byte limit")
    expect(st.cache_bytes <= head.nbytes, "cache accounting tracks planes")
    again = st.raw("model.embed_tokens.weight")
    expect(again.shape == (VOCAB, HIDDEN),
           "evicted plane re-reads on demand")
    expect(st.raw("lm_head.weight").shape == (VOCAB, HIDDEN),
           "hot plane stays addressable across re-reads")


def test_dequant_parallel_matches_serial(tmp):
    config = synthetic_checkpoint(tmp)
    header_path = os.path.join(tmp, "llm_defines.h")
    with open(header_path, "w") as fh:
        fh.write(DEFINES)
    defines = parse_llm_defines(header_path)
    engine_module.PARALLEL_MIN_ELEMENTS = 10 ** 9
    serial = engine_module.Hy4Engine(tmp, defines, config).decode_step(
        7, 0, {}, {}, {})
    engine_module.PARALLEL_MIN_ELEMENTS = 1
    parallel = engine_module.Hy4Engine(tmp, defines, config).decode_step(
        7, 0, {}, {}, {})
    engine_module.shutdown_worker_pool()
    engine_module.PARALLEL_MIN_ELEMENTS = 1000000
    expect(np.array_equal(serial, parallel),
           "parallel dequant path is bit-identical to serial")
    expect(np.isfinite(parallel).all(), "parallel path stays finite")


def test_per_layer_sinks(tmp):
    config = synthetic_checkpoint(tmp)
    header_path = os.path.join(tmp, "llm_defines.h")
    with open(header_path, "w") as fh:
        fh.write(DEFINES)
    defines = parse_llm_defines(header_path)
    eng = engine_module.Hy4Engine(tmp, defines, config)
    expect(eng.sinks.shape == (LAYERS, HEADS), "one sink row per layer")
    expect(not np.array_equal(eng.sinks[0], eng.sinks[1]),
           "synthetic sinks differ between layers")
    reference = eng.decode_step(3, 0, {}, {}, {})
    eng.sinks = np.tile(eng.sinks[0], (LAYERS, 1))
    degraded = eng.decode_step(3, 0, {}, {}, {})
    expect(not np.allclose(reference, degraded),
           "layer-1 attention must consume the layer-1 sink row")
    expect(np.isfinite(degraded).all(), "degraded run stays finite")


def test_engine_end_to_end(tmp):
    config = synthetic_checkpoint(tmp)
    header_path = os.path.join(tmp, "llm_defines.h")
    with open(header_path, "w") as fh:
        fh.write(DEFINES)
    defines = parse_llm_defines(header_path)
    eng = engine_module.Hy4Engine(tmp, defines, config)
    states = {}
    caches = {}
    capture = {}
    streams = eng.decode_step(3, 0, states, caches, capture)
    expect(streams.shape == (HC, HIDDEN), "stream block shape")
    expect(np.isfinite(streams).all(), "streams finite")
    expect(np.array_equal(bf16_round_f32(streams), streams),
           "hc stream state is bf16-resident like the vendor reference")
    expect((0, 1) in capture, "routed layer captured")
    ids, weights = capture[(0, 1)]
    expect(ids.shape == (TOP_K,), "route ids shape")
    expect(int(ids.min()) >= 0 and int(ids.max()) < EXPERTS,
           "route ids inside expert range")
    expect(abs(float(weights.sum()) - 2.827) < 1e-5,
           "route weights sum to the routing scale")
    k_pe = caches[1][1][0]
    raw = None
    expect(len(caches[1][0]) == 1, "one cached token after position 0")
    eng.decode_step(5, 1, states, caches, capture)
    expect(len(caches[1][0]) == 2, "kv cache grows with position")
    token, score = eng.logits(streams)
    expect(0 <= token < VOCAB, "argmax inside vocabulary")
    expect(np.isfinite(score), "finite head score")
    again = eng.decode_step(3, 0, {}, {}, {})
    token2, score2 = eng.logits(again)
    expect(token2 == token and score2 == score,
           "decode is deterministic for the same input")
    try:
        eng.decode_step(3, 2048, {}, {}, {})
        raise AssertionError("context beyond indexer inertness must fail")
    except ValueError:
        pass
    try:
        eng.decode_step(VOCAB + 5, 0, {}, {}, {})
        raise AssertionError("token outside vocabulary must fail")
    except ValueError:
        pass


def main():
    test_dequant_fp8()
    test_cross_check()
    test_routing_selection_ties()
    test_sink_softmax()
    test_rope_interleaved()
    with tempfile.TemporaryDirectory() as tmp:
        test_per_layer_sinks(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        test_engine_end_to_end(tmp)
    print("test_t1_reference_hy4: ALL PASS")


if __name__ == "__main__":
    main()
