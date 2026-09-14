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

from t1_reference_common import f32_to_bf16_u16, read_fixture  # noqa: E402

DEFINES = """#pragma once

#define SPARK_LLM_FAMILY_TAG                    glm5_next
#define SPARK_LLM_HIDDEN_DIMENSION              16u
#define SPARK_LLM_LAYER_COUNT                   2u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            32u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-05f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          31u
#define SPARK_LLM_MLA_HEAD_COUNT                2u
#define SPARK_LLM_MLA_LATENT_DIMENSION          8u
#define SPARK_LLM_MLA_QK_NOPE_HEAD_DIMENSION    4u
#define SPARK_LLM_MLA_V_HEAD_DIMENSION          4u
#define SPARK_LLM_MOE_EXPERT_COUNT              4u
#define SPARK_LLM_MOE_TOP_K                     2u
#define SPARK_LLM_MOE_ROUTED_SCALING_FACTOR     2.5f
#define SPARK_LLM_MOE_NORM_TOPK_PROB            1u
#define SPARK_LLM_FIRST_ROUTED_LAYER            1u
"""


def config_document():
    return {
        "text_config": {
            "hidden_size": 16, "num_hidden_layers": 2, "vocab_size": 32,
            "rms_norm_eps": 1e-5, "hc_mult": 2, "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 3, "num_attention_heads": 2,
            "kv_lora_rank": 8, "qk_nope_head_dim": 4, "v_head_dim": 4,
            "swiglu_limit": 10.0, "n_routed_experts": 4, "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "routed_scaling_factor": 2.5, "norm_topk_prob": True,
            "first_k_dense_replace": 1, "eos_token_id": [31],
            "layer_types": ["linear_attention", "deepseek_sparse_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "linear_attn_config": {
                "head_dim": 4, "num_heads": 2, "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
        }
    }


def bf16(name, shape, rng, scale=0.05):
    del name
    return f32_to_bf16_u16(rng.standard_normal(shape).astype(np.float32) * scale)


def f32(name, shape, rng, scale=0.05):
    del name
    return (rng.standard_normal(shape).astype(np.float32) * scale)


def build_tensors():
    rng = np.random.default_rng(7)
    t = {}
    t["model.language_model.embed_tokens.weight"] = bf16("e", (32, 16), rng)
    t["model.language_model.norm.weight"] = bf16("n", (16,), rng)
    head = bf16("lm", (32, 16), rng, scale=0.5)
    head[31] = np.zeros(16, dtype=np.uint16)
    t["lm_head.weight"] = head
    conv = (8, 4)
    t["model.language_model.layers.0.self_attn.q_conv1d.weight"] = \
        bf16("q", conv, rng)
    t["model.language_model.layers.0.self_attn.k_conv1d.weight"] = \
        bf16("k", conv, rng)
    t["model.language_model.layers.0.self_attn.v_conv1d.weight"] = \
        bf16("v", conv, rng)
    t["model.language_model.layers.0.self_attn.A_log"] = f32("a", (2,), rng)
    t["model.language_model.layers.0.self_attn.dt_bias"] = f32("d", (8,), rng)
    t["model.language_model.layers.0.self_attn.o_norm.weight"] = \
        bf16("o", (4,), rng)
    for layer in range(2):
        p = f"model.language_model.layers.{layer}."
        t[p + "input_layernorm.weight"] = bf16("il", (16,), rng)
        t[p + "post_attention_layernorm.weight"] = bf16("pl", (16,), rng)
        rows = 8
        t[p + "hc_attn_fn"] = bf16("hf", (8, 32), rng)
        t[p + "hc_attn_base"] = f32("hb", (8,), rng)
        t[p + "hc_attn_scale"] = f32("hs", (3,), rng, scale=1.0)
        t[p + "hc_ffn_fn"] = bf16("ff", (8, 32), rng)
        t[p + "hc_ffn_base"] = f32("fb", (8,), rng)
        t[p + "hc_ffn_scale"] = f32("fs", (3,), rng, scale=1.0)
        if layer == 0:
            t[p + "self_attn.q_proj.weight"] = bf16("q", (8, 16), rng)
            t[p + "self_attn.k_proj.weight"] = bf16("k", (8, 16), rng)
            t[p + "self_attn.v_proj.weight"] = bf16("v", (8, 16), rng)
            t[p + "self_attn.b_proj.weight"] = bf16("b", (2, 16), rng)
            t[p + "self_attn.f_a_proj.weight"] = bf16("fa", (8, 16), rng)
            t[p + "self_attn.f_b_proj.weight"] = bf16("fb2", (8, 8), rng)
            t[p + "self_attn.g_a_proj.weight"] = bf16("ga", (8, 16), rng)
            t[p + "self_attn.g_b_proj.weight"] = bf16("gb", (8, 8), rng)
            t[p + "self_attn.o_proj.weight"] = bf16("op", (16, 8), rng)
            t[p + "mlp.gate_proj.weight"] = bf16("g", (8, 16), rng)
            t[p + "mlp.up_proj.weight"] = bf16("u", (8, 16), rng)
            t[p + "mlp.down_proj.weight"] = bf16("dn", (16, 8), rng)
        else:
            t[p + "self_attn.q_a_proj.weight"] = bf16("qa", (8, 16), rng)
            t[p + "self_attn.q_a_layernorm.weight"] = bf16("qn", (8,), rng)
            t[p + "self_attn.q_b_proj.weight"] = bf16("qb", (8, 8), rng)
            t[p + "self_attn.kv_a_proj_with_mqa.weight"] = bf16("ka", (8, 16), rng)
            t[p + "self_attn.kv_a_layernorm.weight"] = bf16("kn", (8,), rng)
            t[p + "self_attn.kv_b_proj.weight"] = bf16("kb", (16, 8), rng)
            t[p + "self_attn.o_proj.weight"] = bf16("op", (16, 8), rng)
            t[p + "mlp.gate.weight"] = bf16("gw", (4, 16), rng, scale=0.5)
            t[p + "mlp.gate.e_score_correction_bias"] = bf16("gb2", (4,), rng)
            for e in range(4):
                ep = p + f"mlp.experts.{e}."
                t[ep + "gate_proj.weight"] = bf16("g", (8, 16), rng)
                t[ep + "up_proj.weight"] = bf16("u", (8, 16), rng)
                t[ep + "down_proj.weight"] = bf16("dn", (16, 8), rng)
            sp = p + "mlp.shared_experts."
            t[sp + "gate_proj.weight"] = bf16("g", (8, 16), rng)
            t[sp + "up_proj.weight"] = bf16("u", (8, 16), rng)
            t[sp + "down_proj.weight"] = bf16("dn", (16, 8), rng)
    return t


def write_safetensors(path, tensors):
    header = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        array = tensors[name]
        dtype = "BF16" if array.dtype == np.uint16 else "F32"
        itemsize = 2 if array.dtype == np.uint16 else 4
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


def write_checkpoint(directory):
    os.makedirs(directory, exist_ok=True)
    write_safetensors(os.path.join(directory, "model.safetensors"),
                      build_tensors())
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(config_document(), fh)


def write_header(path):
    with open(path, "w") as fh:
        fh.write(DEFINES)


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
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
         "--family", "glm5_next", "--checkpoint", checkpoint,
         "--header", header, "--prompts", prompts, "--output", output],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"generator failed: {result.stderr}")
    return result


def compare(reference, candidate):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_compare.py"),
         "compare", "--reference", reference, "--candidate", candidate],
        capture_output=True, text=True)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    workspace = tempfile.mkdtemp(prefix="t1ref-test-")
    try:
        checkpoint = os.path.join(workspace, "checkpoint")
        write_checkpoint(checkpoint)
        header = os.path.join(workspace, "llm_defines.h")
        write_header(header)
        prompts = os.path.join(workspace, "prompts.json")
        write_prompts(prompts)
        out_a = os.path.join(workspace, "run_a")
        out_b = os.path.join(workspace, "run_b")
        run_generator(checkpoint, header, prompts, out_a)
        run_generator(checkpoint, header, prompts, out_b)
        fixture_a = os.path.join(out_a, "glm5_next", "synth_a.t1r")
        fixture_b = os.path.join(out_b, "glm5_next", "synth_a.t1r")
        expect(os.path.exists(fixture_a), "fixture missing after generation")
        _, arrays = read_fixture(fixture_a)
        expect("pos0000_layer0000_streams" in arrays, "anchor capture missing")
        expect("pos0006_layer0001_route_ids" in arrays, "route capture missing")
        expect(int(arrays["generated_token_ids"][0]) >= 0, "no generated tokens")
        expect(open(fixture_a, "rb").read() == open(fixture_b, "rb").read(),
               "generator is not byte-deterministic")
        manifest = json.load(open(os.path.join(out_a, "glm5_next",
                                               "MANIFEST.json")))
        expect(manifest["fixtures"]["synth_a.t1r"]["sha256"] ==
               __import__("hashlib").sha256(open(fixture_a, "rb").read())
               .hexdigest(), "manifest sha mismatch")
        passed = compare(fixture_a, fixture_b)
        expect(passed.returncode == 0, f"identical fixtures must PASS: "
                                       f"{passed.stdout} {passed.stderr}")
        corrupted = os.path.join(workspace, "synth_a_corrupt.t1r")
        corrupt = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "tools", "t1_reference_compare.py"),
             "corrupt-fixture", "--source", fixture_a, "--target", corrupted,
             "--array", "pos0001_layer0001_streams", "--offset", "9"],
            capture_output=True, text=True)
        expect(corrupt.returncode == 0, f"corrupt-fixture failed: {corrupt.stderr}")
        diverged = compare(fixture_a, corrupted)
        expect(diverged.returncode == 1, "corrupted fixture must FAIL")
        expect("pos0001_layer0001_streams" in diverged.stdout,
               f"FAIL must name the corrupted array: {diverged.stdout}")
        bad_header = os.path.join(workspace, "llm_defines_bad.h")
        with open(header) as fh:
            text = fh.read()
        with open(bad_header, "w") as fh:
            fh.write(text.replace("SPARK_LLM_MLA_LATENT_DIMENSION          8u",
                                  "SPARK_LLM_MLA_LATENT_DIMENSION          9u"))
        mismatch = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
             "--family", "glm5_next", "--checkpoint", checkpoint,
             "--header", bad_header, "--prompts", prompts,
             "--output", os.path.join(workspace, "run_c")],
            capture_output=True, text=True)
        expect(mismatch.returncode != 0,
               "defines/config disagreement must fail loud")
        expect("MLA_LATENT_DIMENSION" in mismatch.stderr,
               f"failure must name the mismatched define: {mismatch.stderr}")
        shutil.rmtree(workspace, ignore_errors=True)
        print("PASS t1_reference_decoder synthetic proof: determinism, "
              "manifest sha, negative control, defines/config fail-closed")
        return 0
    except AssertionError:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
