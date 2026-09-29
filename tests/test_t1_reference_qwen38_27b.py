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

from t1_reference_common import (_E4M3_LUT, _E2M1_LUT, f32_to_bf16_u16,
                                 parse_llm_defines, read_fixture)  # noqa: E402
import t1_reference_qwen38_27b as engine_module  # noqa: E402

FAMILY = "qwen38_27b"
VOCAB = 32
HIDDEN = 16
LAYERS = 4
INTERMEDIATE = 48
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
GDN_K_HEADS = 2
GDN_V_HEADS = 4
KD = 4
CONV = 4
QK = GDN_K_HEADS * KD
CHANNELS = 2 * QK + GDN_V_HEADS * KD
EOT = 31

DEFINES = """#pragma once

#define SPARK_LLM_FAMILY_TAG                    qwen38_27b
#define SPARK_LLM_HIDDEN_DIMENSION              16u
#define SPARK_LLM_LAYER_COUNT                   4u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            32u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        64u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-05f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          31u

#define SPARK_LLM_QUERY_HEAD_COUNT              4u
#define SPARK_LLM_KV_HEAD_COUNT                 2u
#define SPARK_LLM_HEAD_DIMENSION                8u
#define SPARK_LLM_ROPE_DIMENSION                2u
#define SPARK_LLM_ROPE_THETA                    10000.0f

#define SPARK_LLM_ATTENTION_PERIOD              4u
#define SPARK_LLM_FULL_ATTENTION_PHASE          3u

#define SPARK_LLM_GDN_KEY_HEAD_COUNT            2u
#define SPARK_LLM_GDN_VALUE_HEAD_COUNT          4u
#define SPARK_LLM_GDN_HEAD_KEY_DIMENSION        4u
#define SPARK_LLM_GDN_HEAD_VALUE_DIMENSION      4u
#define SPARK_LLM_GDN_CONV_KERNEL               4u

#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  48u
#define SPARK_LLM_MTP_LAYER_COUNT               0u
"""

BROKEN_DEFINES = [
    ("#define SPARK_LLM_GDN_VALUE_HEAD_COUNT          4u",
     "#define SPARK_LLM_GDN_VALUE_HEAD_COUNT          5u",
     "GDN_VALUE_HEAD_COUNT"),
    ("#define SPARK_LLM_FULL_ATTENTION_PHASE          3u",
     "#define SPARK_LLM_FULL_ATTENTION_PHASE          0u",
     "FULL_ATTENTION_PHASE"),
    ("#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  48u",
     "#define SPARK_LLM_DENSE_INTERMEDIATE_DIMENSION  999u",
     "DENSE_INTERMEDIATE_DIMENSION"),
    ("#define SPARK_LLM_ROPE_THETA                    10000.0f",
     "#define SPARK_LLM_ROPE_THETA                    999.0f",
     "ROPE_THETA"),
    ("#define SPARK_LLM_ROPE_DIMENSION                2u",
     "#define SPARK_LLM_ROPE_DIMENSION                3u",
     "ROPE_DIMENSION"),
]


def config_document():
    return {
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "eos_token_id": EOT,
        "max_position_embeddings": 64,
        "full_attention_interval": 4,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_theta": 10000.0,
                            "mrope_interleaved": True,
                            "mrope_section": [1, 1, 0],
                            "rope_type": "default"},
        "linear_num_key_heads": GDN_K_HEADS,
        "linear_num_value_heads": GDN_V_HEADS,
        "linear_key_head_dim": KD,
        "linear_value_head_dim": KD,
        "linear_conv_kernel_dim": CONV,
        "intermediate_size": INTERMEDIATE,
        "mtp_num_hidden_layers": 0,
        "hidden_act": "silu",
        "layer_types": ["linear_attention", "linear_attention",
                        "linear_attention", "full_attention"],
    }


def bf16(shape, rng, scale=0.05):
    return f32_to_bf16_u16(rng.standard_normal(shape).astype(np.float32)
                           * scale)


def nearest_codes(lut, values):
    finite = np.isfinite(lut)
    keep = np.flatnonzero(finite)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    distances = np.abs(lut[keep][None, :] - flat[:, None])
    codes = keep[np.argmin(distances, axis=1)]
    return codes.astype(np.uint8).reshape(np.asarray(values).shape)


def fp8_weight(shape, rng, scale=0.05):
    values = rng.standard_normal(shape).astype(np.float32) * scale
    payload = nearest_codes(_E4M3_LUT, values / 0.01)
    scale_inv = f32_to_bf16_u16(np.array([[0.01]], dtype=np.float32))
    return payload, scale_inv


def nvfp4_weight(rows, cols, rng, scale=0.05):
    groups = cols // 16
    payload = np.empty((rows, cols // 2), dtype=np.uint8)
    group_scale = np.float32(0.25)
    for g in range(groups):
        values = rng.standard_normal((rows, 16)).astype(np.float32) * scale
        codes = nearest_codes(_E2M1_LUT, values / group_scale)
        payload[:, g * 8:(g + 1) * 8] = (codes[:, 1::2] << 4) | codes[:, 0::2]
    scales = nearest_codes(_E4M3_LUT,
                           np.full((rows, groups), group_scale,
                                   dtype=np.float32))
    return payload, scales, np.array([1.0], dtype=np.float32)


def build_tensors():
    rng = np.random.default_rng(27)
    t = {}
    t["model.language_model.embed_tokens.weight"] = bf16((VOCAB, HIDDEN), rng)
    t["model.language_model.norm.weight"] = bf16((HIDDEN,), rng)
    head = bf16((VOCAB, HIDDEN), rng, scale=0.5)
    head[EOT] = np.zeros(HIDDEN, dtype=np.uint16)
    t["lm_head.weight"] = head
    for layer in range(LAYERS):
        p = f"model.language_model.layers.{layer}."
        t[p + "input_layernorm.weight"] = bf16((HIDDEN,), rng)
        t[p + "post_attention_layernorm.weight"] = bf16((HIDDEN,), rng)
        if layer < 3:
            if layer == 0:
                payload, scale_inv = fp8_weight((CHANNELS, HIDDEN), rng)
                t[p + "linear_attn.in_proj_qkv.weight"] = payload
                t[p + "linear_attn.in_proj_qkv.weight_scale_inv"] = scale_inv
            else:
                t[p + "linear_attn.in_proj_qkv.weight"] = \
                    bf16((CHANNELS, HIDDEN), rng)
            t[p + "linear_attn.conv1d.weight"] = bf16((CHANNELS, 1, CONV), rng)
            t[p + "linear_attn.A_log"] = bf16((GDN_V_HEADS,), rng, scale=0.1)
            t[p + "linear_attn.dt_bias"] = bf16((GDN_V_HEADS,), rng, scale=0.1)
            t[p + "linear_attn.in_proj_a.weight"] = bf16((GDN_V_HEADS, HIDDEN),
                                                         rng)
            t[p + "linear_attn.in_proj_b.weight"] = bf16((GDN_V_HEADS, HIDDEN),
                                                         rng)
            t[p + "linear_attn.in_proj_z.weight"] = bf16((GDN_V_HEADS * KD,
                                                          HIDDEN), rng)
            t[p + "linear_attn.norm.weight"] = bf16((KD,), rng)
            t[p + "linear_attn.out_proj.weight"] = bf16((HIDDEN,
                                                         GDN_V_HEADS * KD),
                                                        rng)
        else:
            t[p + "self_attn.q_proj.weight"] = bf16((HEADS * 2 * HEAD_DIM,
                                                     HIDDEN), rng)
            t[p + "self_attn.k_proj.weight"] = bf16((KV_HEADS * HEAD_DIM,
                                                     HIDDEN), rng)
            t[p + "self_attn.v_proj.weight"] = bf16((KV_HEADS * HEAD_DIM,
                                                     HIDDEN), rng)
            t[p + "self_attn.o_proj.weight"] = bf16((HIDDEN,
                                                     HEADS * HEAD_DIM), rng)
            t[p + "self_attn.q_norm.weight"] = bf16((HEAD_DIM,), rng)
            t[p + "self_attn.k_norm.weight"] = bf16((HEAD_DIM,), rng)
        if layer == 1:
            for name, rows, cols in (("gate_proj", INTERMEDIATE, HIDDEN),
                                     ("up_proj", INTERMEDIATE, HIDDEN),
                                     ("down_proj", HIDDEN, INTERMEDIATE)):
                payload, scale_inv = fp8_weight((rows, cols), rng)
                t[p + f"mlp.{name}.weight"] = payload
                t[p + f"mlp.{name}.weight_scale_inv"] = scale_inv
        elif layer == 2:
            for name, rows, cols in (("gate_proj", INTERMEDIATE, HIDDEN),
                                     ("up_proj", INTERMEDIATE, HIDDEN),
                                     ("down_proj", HIDDEN, INTERMEDIATE)):
                packed, scales, global_scale = nvfp4_weight(rows, cols, rng)
                t[p + f"mlp.{name}.weight_packed"] = packed
                t[p + f"mlp.{name}.weight_scale"] = scales
                t[p + f"mlp.{name}.weight_global_scale"] = global_scale
        else:
            t[p + "mlp.gate_proj.weight"] = bf16((INTERMEDIATE, HIDDEN), rng)
            t[p + "mlp.up_proj.weight"] = bf16((INTERMEDIATE, HIDDEN), rng)
            t[p + "mlp.down_proj.weight"] = bf16((HIDDEN, INTERMEDIATE), rng)
    return t


def write_safetensors(path, tensors):
    header = {}
    offset = 0
    blobs = []
    dtypes = {np.dtype("uint16"): "BF16", np.dtype("uint8"): "U8",
              np.dtype("float32"): "F32"}
    for name in sorted(tensors):
        array = tensors[name]
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": dtypes[array.dtype],
                        "shape": list(array.shape),
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


def write_header(path, text=DEFINES):
    with open(path, "w") as fh:
        fh.write(text)


def write_prompts(path):
    document = {"prompts": [{
        "name": "synth_a",
        "prompt_token_ids": [3, 7, 11, 5],
        "new_tokens": 3,
        "capture_layers": [0, 3],
    }]}
    with open(path, "w") as fh:
        json.dump(document, fh)


def run_generator(checkpoint, header, prompts, output):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
         "--family", FAMILY, "--checkpoint", checkpoint,
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


OCP_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


class Nvfp4Store:
    def __init__(self, arrays):
        self.arrays = arrays

    def pread(self, name):
        return self.arrays[name]

    def entry(self, name):
        return self.arrays[name]


def nvfp4_scale_semantics():
    name = "w.weight"
    global_scale = np.float32(0.5)
    codes = np.array([list(range(16)), list(range(15, -1, -1))],
                     dtype=np.uint8)
    payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    scales = np.array([[0x38], [0x40]], dtype=np.uint8)
    want = np.array([[(-1.0 if c & 8 else 1.0) * OCP_E2M1[c & 7]
                      for c in row] for row in codes], dtype=np.float32)
    want *= np.array([[1.0], [2.0]], dtype=np.float32) / global_scale
    engine = object.__new__(engine_module.ENGINE_CLASS)
    engine.st = Nvfp4Store({name + "_packed": payload,
                            name + "_scale": scales,
                            name + "_global_scale":
                                np.array([global_scale], dtype=np.float32)})
    engine._weights = {}
    engine._memo_bytes = 0
    engine._memo_limit = 0
    got = engine.tensor(name)
    expect(np.array_equal(got, want),
           f"nvfp4 decode is not e2m1 x e4m3 / global_scale: {got} vs {want}")


def main():
    workspace = tempfile.mkdtemp(prefix="t1ref-q27-test-")
    try:
        checkpoint = os.path.join(workspace, "checkpoint")
        write_checkpoint(checkpoint)
        header = os.path.join(workspace, "llm_defines.h")
        write_header(header)
        prompts = os.path.join(workspace, "prompts.json")
        write_prompts(prompts)
        out_a = os.path.join(workspace, "run_a")
        out_b = os.path.join(workspace, "run_b")
        result_a = run_generator(checkpoint, header, prompts, out_a)
        expect(result_a.returncode == 0,
               f"generator failed: {result_a.stderr}")
        result_b = run_generator(checkpoint, header, prompts, out_b)
        expect(result_b.returncode == 0,
               f"generator rerun failed: {result_b.stderr}")
        fixture_a = os.path.join(out_a, FAMILY, "synth_a.t1r")
        fixture_b = os.path.join(out_b, FAMILY, "synth_a.t1r")
        expect(os.path.exists(fixture_a), "fixture missing after generation")
        _, arrays = read_fixture(fixture_a)
        expect("pos0000_layer0000_streams" in arrays,
               "GDN-layer anchor capture missing")
        expect("pos0005_layer0003_streams" in arrays,
               "full-attention-layer anchor capture missing")
        expect(arrays["pos0000_layer0000_streams"].shape[0] == HIDDEN,
               "streams width must equal the hidden dimension")
        expect(int(arrays["generated_token_ids"][0]) >= 0,
               "no generated tokens")
        expect(open(fixture_a, "rb").read() == open(fixture_b, "rb").read(),
               "generator is not byte-deterministic")
        manifest = json.load(open(os.path.join(out_a, FAMILY,
                                               "MANIFEST.json")))
        expect(manifest["fixtures"]["synth_a.t1r"]["sha256"] ==
               hashlib.sha256(open(fixture_a, "rb").read()).hexdigest(),
               "manifest sha mismatch")
        expect(manifest["defines_config_mismatches"] == [],
               "synthetic defines must close against the synthetic config")
        passed = compare(fixture_a, fixture_b)
        expect(passed.returncode == 0,
               f"identical fixtures must PASS: {passed.stdout} {passed.stderr}")
        corrupted = os.path.join(workspace, "synth_a_corrupt.t1r")
        corrupt = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "tools", "t1_reference_compare.py"),
             "corrupt-fixture", "--source", fixture_a, "--target", corrupted,
             "--array", "pos0001_layer0003_streams", "--offset", "9"],
            capture_output=True, text=True)
        expect(corrupt.returncode == 0,
               f"corrupt-fixture failed: {corrupt.stderr}")
        diverged = compare(fixture_a, corrupted)
        expect(diverged.returncode == 1, "corrupted fixture must FAIL")
        expect("pos0001_layer0003_streams" in diverged.stdout,
               f"FAIL must name the corrupted array: {diverged.stdout}")
        for line, broken_line, needle in BROKEN_DEFINES:
            expect(line in DEFINES, f"selftest define drift: {line}")
            bad_header = os.path.join(workspace, "llm_defines_bad.h")
            write_header(bad_header, DEFINES.replace(line, broken_line))
            mismatch = run_generator(checkpoint, bad_header, prompts,
                                     os.path.join(workspace, "run_c"))
            expect(mismatch.returncode != 0,
                   f"defines/config disagreement must fail loud for {needle}")
            expect(needle in mismatch.stderr,
                   f"failure must name {needle}: {mismatch.stderr}")
        nvfp4_scale_semantics()
        parsed = parse_llm_defines(header)
        config = json.load(open(os.path.join(checkpoint, "config.json")))
        engine = engine_module.ENGINE_CLASS(checkpoint, parsed, config)
        expect(engine.mismatches == [], "clean engine must report no mismatch")
        streams = engine.decode_step(3, 0, {}, {}, {})
        expect(np.isfinite(streams).all(), "decode must stay finite")
        try:
            engine.decode_step(VOCAB, 1, {}, {}, {})
        except ValueError:
            pass
        else:
            raise AssertionError("token outside vocabulary must fail loud")
        shutil.rmtree(workspace, ignore_errors=True)
        print("PASS t1_reference_qwen38_27b synthetic proof: determinism, "
              "manifest sha, negative control, fp8+nvfp4+bf16 dequant paths, "
              "nvfp4 OCP scale semantics, "
              "defines/config fail-closed, vocabulary bounds")
        return 0
    except AssertionError:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
