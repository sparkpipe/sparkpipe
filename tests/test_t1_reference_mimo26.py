import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from t1_reference_common import (_E2M1_LUT, _E4M3_LUT, f32_to_bf16_u16,  # noqa: E402
                                 parse_llm_defines, read_fixture)
import t1_reference_mimo26 as mimo  # noqa: E402

HIDDEN = 128
HEADS = 4
HEAD_DIM = 12
V_DIM = 8
ROPE = 4
KV = {0: 2, 1: 4}
RANKS = 2
WINDOW = 2
EXPERTS = 4
TOP_K = 2
EXPERT_INTER = 64
DENSE_INTER = 128
VOCAB = 48
EOT = 47
EPS = 1e-6
THETA = {0: 1000000.0, 1: 10000.0}
VALUE_SCALE = 0.707
PATTERN = [0, 1]

DEFINES = f"""#pragma once
#define SPARK_LLM_FAMILY_TAG                    mimo26
#define SPARK_LLM_HIDDEN_DIMENSION              {HIDDEN}u
#define SPARK_LLM_LAYER_COUNT                   2u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            {VOCAB}u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        64u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          {EOT}u
#define SPARK_LLM_ATTENTION_HEAD_COUNT          {HEADS}u
#define SPARK_LLM_SWA_ATTENTION_HEAD_COUNT      {HEADS}u
#define SPARK_LLM_HEAD_DIMENSION                {HEAD_DIM}u
#define SPARK_LLM_SWA_HEAD_DIMENSION            {HEAD_DIM}u
#define SPARK_LLM_VALUE_HEAD_DIMENSION          {V_DIM}u
#define SPARK_LLM_SWA_VALUE_HEAD_DIMENSION      {V_DIM}u
#define SPARK_LLM_FULL_KV_HEAD_COUNT            {KV[0]}u
#define SPARK_LLM_SWA_KV_HEAD_COUNT             {KV[1]}u
#define SPARK_LLM_ROPE_DIMENSION                {ROPE}u
#define SPARK_LLM_FULL_ROPE_THETA               {THETA[0]}f
#define SPARK_LLM_SWA_ROPE_THETA                {THETA[1]}f
#define SPARK_LLM_SLIDING_WINDOW_TOKENS         {WINDOW}u
#define SPARK_LLM_ATTENTION_VALUE_SCALE         {VALUE_SCALE}f
#define SPARK_LLM_FULL_SINK_BIAS                0u
#define SPARK_LLM_SWA_SINK_BIAS                 1u
#define SPARK_LLM_FULL_ATTENTION_LAYER_MASK     0x1
#define SPARK_LLM_QKV_SOURCE_INTERLEAVE_RANKS   {RANKS}u
#define SPARK_LLM_FIRST_ROUTED_LAYER            1u
#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  {DENSE_INTER}u
#define SPARK_LLM_EXPERT_INTERMEDIATE_DIMENSION {EXPERT_INTER}u
#define SPARK_LLM_ROUTED_EXPERT_COUNT           {EXPERTS}u
#define SPARK_LLM_EXPERTS_PER_TOKEN             {TOP_K}u
#define SPARK_LLM_ROUTER_GROUP_COUNT            1u
#define SPARK_LLM_ROUTER_TOP_GROUP_COUNT        1u
#define SPARK_LLM_ROUTER_NORM_TOPK              1u
#define SPARK_LLM_ROUTER_NORM_EPSILON           1e-20f
#define SPARK_LLM_ROUTED_SCALING_FACTOR         1.0f
"""


def config_document():
    return {
        "hidden_size": HIDDEN, "num_hidden_layers": 2, "vocab_size": VOCAB,
        "max_position_embeddings": 64, "layernorm_epsilon": EPS, "eos_token_id": EOT,
        "num_attention_heads": HEADS, "swa_num_attention_heads": HEADS,
        "head_dim": HEAD_DIM, "swa_head_dim": HEAD_DIM, "v_head_dim": V_DIM,
        "swa_v_head_dim": V_DIM, "num_key_value_heads": KV[0],
        "swa_num_key_value_heads": KV[1], "partial_rotary_factor": 0.334,
        "rope_theta": THETA[0], "swa_rope_theta": THETA[1], "sliding_window": WINDOW,
        "attention_value_scale": VALUE_SCALE, "add_full_attention_sink_bias": False,
        "add_swa_attention_sink_bias": True, "intermediate_size": DENSE_INTER,
        "moe_intermediate_size": EXPERT_INTER, "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K, "n_group": 1, "topk_group": 1,
        "norm_topk_prob": True, "routed_scaling_factor": None,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "hidden_act": "silu",
        "attention_projection_layout": "fused_qkv", "tie_word_embeddings": False,
        "attention_bias": False, "hybrid_layer_pattern": PATTERN, "moe_layer_freq": [0, 1],
        "quantization_config": {"fmt": "e4m3", "weight_block_size": [128, 128],
                                "mxfp4_block_size": 32},
    }


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def spec_e2m1(code):
    sign = -1.0 if code & 8 else 1.0
    return sign * [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][code & 7]


def e4m3_encode(values):
    finite = np.where(np.isfinite(_E4M3_LUT), _E4M3_LUT, np.inf)
    return np.abs(values[..., None] - finite[None, :]).argmin(axis=-1).astype(np.uint8)


def fp8_quantize(weight, block_rows):
    payload = np.zeros(weight.shape, dtype=np.uint8)
    grid = np.zeros((len(block_rows), weight.shape[1] // 128), dtype=np.float32)
    for block, rows in enumerate(block_rows):
        for col in range(grid.shape[1]):
            tile = weight[rows, col * 128:(col + 1) * 128]
            scale = np.float32(np.abs(tile).max() / 448.0)
            grid[block, col] = scale
            payload[rows, col * 128:(col + 1) * 128] = e4m3_encode(tile / scale)
    return payload, grid


def fp8_dequantize(payload, grid, block_rows):
    out = np.zeros(payload.shape, dtype=np.float64)
    for block, rows in enumerate(block_rows):
        for col in range(grid.shape[1]):
            out[rows, col * 128:(col + 1) * 128] = _E4M3_LUT[payload[rows, col * 128:(col + 1) * 128]] * float(grid[block, col])
    return out


def mxfp4_dequantize(payload, scales):
    rows = payload.shape[0]
    out = np.zeros((rows, payload.shape[1] * 2), dtype=np.float64)
    for r in range(rows):
        for p in range(payload.shape[1]):
            scale = 2.0 ** (int(scales[r, (2 * p) // 32]) - 127)
            out[r, 2 * p] = spec_e2m1(int(payload[r, p]) & 15) * scale
            out[r, 2 * p + 1] = spec_e2m1(int(payload[r, p]) >> 4) * scale
    return out


def rank_segments(kind):
    kv = KV[kind]
    q_local = HEADS // RANKS * HEAD_DIM
    k_local = kv // RANKS * HEAD_DIM
    v_local = kv // RANKS * V_DIM
    return q_local, k_local, v_local, q_local + k_local + v_local


def interleave_qkv(q, k, v, kind):
    q_local, k_local, v_local, per = rank_segments(kind)
    rows = []
    for rank in range(RANKS):
        rows += [q[rank * q_local:(rank + 1) * q_local], k[rank * k_local:(rank + 1) * k_local],
                 v[rank * v_local:(rank + 1) * v_local]]
    fused = np.concatenate(rows)
    blocks = []
    for rank in range(RANKS):
        for first in range(0, per, 128):
            blocks.append(np.arange(rank * per + first, rank * per + min(per, first + 128)))
    return fused, blocks


def bf16(x):
    return f32_to_bf16_u16(np.asarray(x, dtype=np.float32))


def build_checkpoint(rng):
    tensors = {}
    model = {"layers": []}
    embed = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32) * 0.5
    head = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32) * 0.5
    head[EOT] = 0.0
    final_norm = 1.0 + 0.1 * rng.standard_normal(HIDDEN).astype(np.float32)
    tensors[mimo.EMBED_NAME] = bf16(embed)
    tensors[mimo.LM_HEAD_NAME] = bf16(head)
    tensors[mimo.FINAL_NORM_NAME] = bf16(final_norm)
    model.update(embed=mimo.bf16_to_f32(tensors[mimo.EMBED_NAME]), head=mimo.bf16_to_f32(tensors[mimo.LM_HEAD_NAME]),
                 final_norm=mimo.bf16_to_f32(tensors[mimo.FINAL_NORM_NAME]))
    for layer, kind in enumerate(PATTERN):
        prefix = f"model.layers.{layer}."
        entry = {"kind": kind}
        kv = KV[kind]
        q = rng.standard_normal((HEADS * HEAD_DIM, HIDDEN)).astype(np.float32) * 0.2
        k = rng.standard_normal((kv * HEAD_DIM, HIDDEN)).astype(np.float32) * 0.2
        v = rng.standard_normal((kv * V_DIM, HIDDEN)).astype(np.float32) * 0.2
        fused, blocks = interleave_qkv(q, k, v, kind)
        payload, grid = fp8_quantize(fused, blocks)
        tensors[prefix + "self_attn.qkv_proj.weight"] = payload
        tensors[prefix + "self_attn.qkv_proj.weight_scale_inv"] = grid
        deq = fp8_dequantize(payload, grid, blocks)
        q_local, k_local, v_local, per = rank_segments(kind)
        entry["q"] = np.concatenate([deq[r * per:r * per + q_local] for r in range(RANKS)])
        entry["k"] = np.concatenate([deq[r * per + q_local:r * per + q_local + k_local] for r in range(RANKS)])
        entry["v"] = np.concatenate([deq[r * per + q_local + k_local:(r + 1) * per] for r in range(RANKS)])
        o = rng.standard_normal((HIDDEN, HEADS * V_DIM)).astype(np.float32) * 0.2
        tensors[prefix + "self_attn.o_proj.weight"] = bf16(o)
        entry["o"] = mimo.bf16_to_f32(tensors[prefix + "self_attn.o_proj.weight"])
        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensors[prefix + norm + ".weight"] = bf16(1.0 + 0.1 * rng.standard_normal(HIDDEN))
            entry[norm] = mimo.bf16_to_f32(tensors[prefix + norm + ".weight"])
        if kind == 1:
            tensors[prefix + "self_attn.attention_sink_bias"] = bf16(rng.standard_normal(HEADS) * 2.0)
            entry["sink"] = mimo.bf16_to_f32(tensors[prefix + "self_attn.attention_sink_bias"])
        if layer == 0:
            for proj, rows, cols in (("gate_proj", DENSE_INTER, HIDDEN), ("up_proj", DENSE_INTER, HIDDEN),
                                     ("down_proj", HIDDEN, DENSE_INTER)):
                weight = rng.standard_normal((rows, cols)).astype(np.float32) * 0.2
                blocks = [np.arange(b * 128, min(rows, b * 128 + 128)) for b in range(rows // 128)]
                payload, grid = fp8_quantize(weight, blocks)
                tensors[prefix + f"mlp.{proj}.weight"] = payload
                tensors[prefix + f"mlp.{proj}.weight_scale_inv"] = grid
                entry[proj] = fp8_dequantize(payload, grid, blocks)
        else:
            router = rng.standard_normal((EXPERTS, HIDDEN)).astype(np.float32) * 0.2
            tensors[prefix + "mlp.gate.weight"] = bf16(router)
            entry["router"] = mimo.bf16_to_f32(tensors[prefix + "mlp.gate.weight"])
            entry["bias"] = (rng.standard_normal(EXPERTS) * 0.1).astype(np.float32)
            tensors[prefix + "mlp.gate.e_score_correction_bias"] = entry["bias"]
            entry["experts"] = []
            for expert in range(EXPERTS):
                weights = {}
                for proj, rows, cols in (("gate_proj", EXPERT_INTER, HIDDEN), ("up_proj", EXPERT_INTER, HIDDEN),
                                         ("down_proj", HIDDEN, EXPERT_INTER)):
                    payload = rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8)
                    scales = rng.integers(122, 126, (rows, cols // 32), dtype=np.uint8)
                    name = prefix + f"mlp.experts.{expert}.{proj}"
                    tensors[name + ".weight"] = payload
                    tensors[name + ".weight_scale"] = scales
                    weights[proj] = mxfp4_dequantize(payload, scales)
                entry["experts"].append(weights)
        model["layers"].append(entry)
    return tensors, model


def write_safetensors(path, tensors, fp8_names):
    header = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        array = tensors[name]
        if array.dtype == np.uint16:
            dtype = "BF16"
        elif array.dtype == np.float32:
            dtype = "F32"
        else:
            dtype = "F8_E4M3" if name in fp8_names else "U8"
        raw = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for raw in blobs:
            fh.write(raw)


def rms(x, w):
    return x / np.sqrt(np.mean(x * x) + EPS) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(rows, position, kind):
    out = rows.copy()
    half = ROPE // 2
    for j in range(half):
        angle = position * THETA[kind] ** (-2.0 * j / ROPE)
        a = rows[:, j].copy()
        b = rows[:, j + half].copy()
        out[:, j] = a * np.cos(angle) - b * np.sin(angle)
        out[:, j + half] = b * np.cos(angle) + a * np.sin(angle)
    return out


def naive_decode(model, tokens):
    caches = [[] for _ in PATTERN]
    streams_log = []
    routes = []
    next_tokens = []
    for position, token in enumerate(tokens):
        h = model["embed"][token].astype(np.float64)
        per_layer = []
        for layer, entry in enumerate(model["layers"]):
            kind = entry["kind"]
            kv = KV[kind]
            x = rms(h, entry["input_layernorm"])
            q = rope((entry["q"] @ x).reshape(HEADS, HEAD_DIM), position, kind)
            k = rope((entry["k"] @ x).reshape(kv, HEAD_DIM), position, kind)
            v = (entry["v"] @ x).reshape(kv, V_DIM) * VALUE_SCALE
            caches[layer].append((position, k, v))
            out = np.zeros((HEADS, V_DIM))
            for head in range(HEADS):
                group = head // (HEADS // kv)
                keys = [(p, kk[group], vv[group]) for p, kk, vv in caches[layer] if kind == 0 or p > position - WINDOW]
                logits = np.array([q[head] @ kk for _, kk, _ in keys]) / np.sqrt(HEAD_DIM)
                extra = [entry["sink"][head]] if kind == 1 else []
                full = np.concatenate([logits, extra])
                probs = np.exp(full - full.max())
                probs /= probs.sum()
                out[head] = sum(probs[i] * keys[i][2] for i in range(len(keys)))
            h = h + entry["o"] @ out.reshape(-1)
            x = rms(h, entry["post_attention_layernorm"])
            if layer == 0:
                h = h + entry["down_proj"] @ (silu(entry["gate_proj"] @ x) * (entry["up_proj"] @ x))
            else:
                scores = 1.0 / (1.0 + np.exp(-(entry["router"] @ x)))
                chosen = sorted(range(EXPERTS), key=lambda e: (-(scores[e] + entry["bias"][e]), e))[:TOP_K]
                weights = scores[chosen] / scores[chosen].sum()
                routes.append(chosen)
                for weight, expert in zip(weights, chosen):
                    w = entry["experts"][expert]
                    h = h + weight * (w["down_proj"] @ (silu(w["gate_proj"] @ x) * (w["up_proj"] @ x)))
            per_layer.append(h.copy())
        streams_log.append(per_layer)
        logits = model["head"] @ rms(h, model["final_norm"])
        next_tokens.append(int(np.argmax(logits)))
    return streams_log, routes, next_tokens


def check_lut_and_matvec(rng):
    expect([float(v) for v in _E2M1_LUT] == [spec_e2m1(c) for c in range(16)],
           f"e2m1 table {list(_E2M1_LUT)} is not the OCP MX e2m1 value set")
    payload = rng.integers(0, 256, (16, 64), dtype=np.uint8)
    scales = rng.integers(118, 130, (16, 4), dtype=np.uint8)
    x = rng.standard_normal(128).astype(np.float32)
    want = mxfp4_dequantize(payload, scales) @ x
    got = mimo.mxfp4_matvec(payload, scales, x)
    expect(np.allclose(got, want, rtol=1e-5, atol=1e-5), "mxfp4 matvec disagrees with the element-wise dequantization")
    print("  PASS e2m1 table and mxfp4 matvec")


def check_layout_guard(rng):
    kind = 0
    kv = KV[kind]
    q = rng.standard_normal((HEADS * HEAD_DIM, HIDDEN)).astype(np.float32)
    k = rng.standard_normal((kv * HEAD_DIM, HIDDEN)).astype(np.float32)
    v = rng.standard_normal((kv * V_DIM, HIDDEN)).astype(np.float32)
    fused, blocks = interleave_qkv(q, k, v, kind)
    payload, _ = fp8_quantize(fused, blocks)
    engine = mimo.Mimo26Engine.__new__(mimo.Mimo26Engine)
    engine.heads, engine.head_dim, engine.v_dim, engine.interleave = HEADS, HEAD_DIM, V_DIM, RANKS
    engine.kv_heads = KV
    order, block_of_row, rows, grid_rows = engine.qkv_layout(kind)
    expect(rows == fused.shape[0] and grid_rows == len(blocks), "layout extent disagrees with the interleave")
    expect(np.array_equal(fused[order], np.concatenate([q, k, v])), "layout order does not de-interleave q|k|v")
    mimo.check_block_amax(payload, block_of_row, "interleaved")
    quarters = np.arange(fused.shape[0]) * 4 // fused.shape[0]
    try:
        mimo.check_block_amax(payload, quarters, "quarters")
    except mimo.Mimo26ConfigError:
        pass
    else:
        raise AssertionError("the amax guard accepted a row-to-block map that is not the quantization map")
    print("  PASS qkv rank interleave and fp8 block guard")


def check_end_to_end(rng):
    tensors, model = build_checkpoint(rng)
    fp8_names = {n for n in tensors if n.endswith("qkv_proj.weight") or (".mlp." in n and n.endswith("_proj.weight") and "experts" not in n)}
    prompt = [3, 9, 17, 5]
    budget = 3
    with tempfile.TemporaryDirectory() as work:
        checkpoint = os.path.join(work, "ckpt")
        os.makedirs(checkpoint)
        write_safetensors(os.path.join(checkpoint, "model.safetensors"), tensors, fp8_names)
        json.dump(config_document(), open(os.path.join(checkpoint, "config.json"), "w"))
        header = os.path.join(work, "llm_defines.h")
        open(header, "w").write(DEFINES)
        prompts = os.path.join(work, "prompts.json")
        json.dump({"prompts": [{"name": "mini", "prompt_token_ids": prompt, "new_tokens": budget,
                                "capture_layers": [0, 1]}]}, open(prompts, "w"))
        outputs = []
        for run in range(2):
            out = os.path.join(work, f"out{run}")
            result = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
                                     "--family", "mimo26", "--checkpoint", checkpoint, "--header", header,
                                     "--prompts", prompts, "--output", out], capture_output=True, text=True)
            expect(result.returncode == 0, f"generator failed: {result.stderr}")
            outputs.append(open(os.path.join(out, "mimo26", "mini.t1r"), "rb").read())
        expect(outputs[0] == outputs[1], "two reference runs are not byte-identical")
        _, arrays = read_fixture(os.path.join(work, "out0", "mimo26", "mini.t1r"))
        generated = [int(t) for t in arrays["generated_token_ids"]]
        tokens = prompt + generated
        streams, routes, next_tokens = naive_decode(model, tokens[:-1])
        for position in range(len(tokens) - 1):
            for layer in (0, 1):
                got = mimo.bf16_to_f32(arrays[f"pos{position:04d}_layer{layer:04d}_streams"])
                want = streams[position][layer]
                error = np.linalg.norm(got - want) / np.linalg.norm(want)
                expect(error < 2e-2, f"position {position} layer {layer}: stream error {error:.3e}")
            ids = [int(i) for i in arrays[f"pos{position:04d}_layer0001_route_ids"]]
            expect(ids == routes[position], f"position {position}: routes {ids} != independent {routes[position]}")
        expect(generated == next_tokens[len(prompt) - 1:], f"greedy tokens {generated} != independent {next_tokens[len(prompt) - 1:]}")
        expect(len(tokens) - 1 > WINDOW + 1, "the fixture must run past the sliding window")
        broken = dict(tensors)
        broken["model.layers.0.self_attn.qkv_proj.weight_scale_inv"] = tensors["model.layers.0.self_attn.qkv_proj.weight_scale_inv"][:1]
        write_safetensors(os.path.join(checkpoint, "model.safetensors"), broken, fp8_names)
        result = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
                                 "--family", "mimo26", "--checkpoint", checkpoint, "--header", header,
                                 "--prompts", prompts, "--output", os.path.join(work, "neg")], capture_output=True, text=True)
        expect(result.returncode != 0 and "qkv_proj.weight_scale_inv" in result.stderr,
               "a qkv scale grid that is not per-rank must fail closed")
        mutated = DEFINES.replace("SPARK_LLM_SLIDING_WINDOW_TOKENS         2u", "SPARK_LLM_SLIDING_WINDOW_TOKENS         3u")
        open(header, "w").write(mutated)
        try:
            mimo.cross_check(parse_llm_defines(header), config_document())
        except mimo.Mimo26ConfigError:
            pass
        else:
            raise AssertionError("a window define that disagrees with config must fail closed")
    print(f"  PASS end to end: {len(tokens) - 1} positions match an independent float64 decode, tokens {generated}")


def check_family_headers():
    llm = parse_llm_defines(os.path.join(ROOT, "model-families/mimo26/include/sparkpipe/llm_defines.h"))
    text = open(os.path.join(ROOT, "model-families/mimo26/include/sparkpipe/spark_mimo26_model.h")).read()
    model = dict((name, value.rstrip("uf")) for name, value in
                 re.findall(r"#define SPARK_MIMO26_MODEL_([A-Z0-9_]+) ([0-9.e+-]+[uf]?)\s*$", text, re.M))
    for name, alias in re.findall(r"#define SPARK_MIMO26_MODEL_([A-Z0-9_]+) SPARK_LLM_([A-Z0-9_]+)\s*$", text, re.M):
        model[name] = str(mimo.define_number(llm, alias))
    pairs = {"HIDDEN_DIMENSION": "HIDDEN_DIMENSION", "LAYER_COUNT": "LAYER_COUNT",
             "OUTPUT_VOCAB_COUNT": "VOCAB_COUNT", "MAXIMUM_CONTEXT_TOKENS": "MAX_POSITIONS",
             "RMS_NORM_EPSILON": "RMS_NORM_EPSILON", "ATTENTION_HEAD_COUNT": "ATTN_HEAD_COUNT",
             "HEAD_DIMENSION": "ATTN_HEAD_DIMENSION", "VALUE_HEAD_DIMENSION": "ATTN_VALUE_DIMENSION",
             "ROPE_DIMENSION": "ATTN_ROPE_DIMENSION", "FULL_KV_HEAD_COUNT": "FULL_KV_HEAD_COUNT",
             "SWA_KV_HEAD_COUNT": "SWA_KV_HEAD_COUNT", "SLIDING_WINDOW_TOKENS": "SLIDING_WINDOW_TOKENS",
             "ATTENTION_VALUE_SCALE": "ATTN_VALUE_SCALE", "FULL_ROPE_THETA": "FULL_ROPE_THETA",
             "SWA_ROPE_THETA": "SWA_ROPE_THETA", "FULL_SINK_BIAS": "FULL_SINK_BIAS", "SWA_SINK_BIAS": "SWA_SINK_BIAS",
             "DENSE_INTERMEDIATE_DIMENSION": "DENSE_INTERMEDIATE_DIMENSION",
             "EXPERT_INTERMEDIATE_DIMENSION": "EXPERT_INTERMEDIATE_DIMENSION",
             "ROUTED_EXPERT_COUNT": "ROUTED_EXPERT_COUNT", "EXPERTS_PER_TOKEN": "EXPERTS_PER_TOKEN",
             "ROUTER_NORM_EPSILON": "ROUTER_NORM_EPSILON", "ROUTER_GROUP_COUNT": "ROUTER_GROUP_COUNT",
             "ROUTER_TOP_GROUP_COUNT": "ROUTER_TOP_GROUP_COUNT", "ROUTED_SCALING_FACTOR": "ROUTED_SCALING_FACTOR"}
    for llm_name, model_name in pairs.items():
        expect(float(mimo.define_number(llm, llm_name)) == float(model[model_name]),
               f"llm_defines {llm_name} disagrees with SPARK_MIMO26_MODEL_{model_name}")
    mask = mimo.define_uint(llm, "FULL_ATTENTION_LAYER_MASK")
    kinds = [int(v) for v in re.search(r"LAYER_KIND\[[^]]*\] =\s*\{([^}]*)\}", text).group(1).replace("\n", "").split(",")]
    expect(all(((mask >> layer) & 1) == (kind == 0) for layer, kind in enumerate(kinds)),
           "FULL_ATTENTION_LAYER_MASK disagrees with SPARK_MIMO26_MODEL_LAYER_KIND")
    print("  PASS llm_defines.h agrees with spark_mimo26_model.h")


COMMITTED = {
    "capital": [264, 3283, 315, 29263, 11, 1947, 11, 323, 3840, 13, 1084, 374, 264, 3283, 429, 702],
    "code": [262, 421, 308, 2651, 220, 15, 510, 286, 470, 220, 15, 198, 262, 4409, 308, 621],
    "science": [429, 374, 279, 9315, 518, 892, 279, 37652, 7262, 315, 3015, 16819, 279, 44375, 7262, 13],
}


def check_committed_fixtures():
    directory = os.path.join(ROOT, "qualification", "t1_reference", "mimo26")
    result = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "t1_reference_compare.py"),
                             "verify-manifest", "--fixture-dir", directory], capture_output=True, text=True)
    expect(result.returncode == 0, f"committed mimo26 fixtures fail their manifest: {result.stdout}{result.stderr}")
    manifest = json.load(open(os.path.join(directory, "MANIFEST.json")))
    header = os.path.join(ROOT, "model-families", "mimo26", "include", "sparkpipe", "llm_defines.h")
    expect(manifest["llm_defines_sha256"] == hashlib.sha256(open(header, "rb").read()).hexdigest(),
           "committed fixtures were generated from a different llm_defines.h; regenerate them")
    expect(manifest["checkpoint"]["config_sha256"] == "61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621",
           "committed fixtures are not from the pinned MiMo-V2.6-Flash-RL config")
    for name, tokens in COMMITTED.items():
        _, arrays = read_fixture(os.path.join(directory, name + ".t1r"))
        expect([int(t) for t in arrays["generated_token_ids"]] == tokens, f"{name}: committed greedy tokens changed")
        expect(sum(1 for key in arrays if key.endswith("_route_ids")) == (len(arrays["prompt_token_ids"]) + 16) * 47,
               f"{name}: fixture does not carry a route decision per routed layer and position")
    print("  PASS committed Flash fixtures: manifest, header, config pin, 3 x 16 greedy tokens")


def main():
    rng = np.random.default_rng(26)
    print("mimo26 t1 reference:")
    check_family_headers()
    check_committed_fixtures()
    check_lut_and_matvec(rng)
    check_layout_guard(rng)
    check_end_to_end(rng)
    print("PASS mimo26 reference engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
