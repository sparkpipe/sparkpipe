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

LAYERS = 12
RATIOS = [0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0]
KV_SOURCES = [1, 4, 6, 9]
INDEX_SOURCES = [1, 4, 6, 9, 2, 3, 7, 5]
CANDIDATE_SOURCE = 6

HIDDEN = 32
HEAD_DIM = 32
ROPE_DIM = 8
HEADS = 2
Q_LORA = 32
O_LORA = 16
O_GROUPS = 2
INDEX_HEADS = 2
INDEX_DIM = 32
WINDOW = 4
VOCAB = 32
INTER = 32
EXPERTS = 4
TOPK = 2
HC_MULT = 2

CONFIG = {
    "eos_token_id": 31,
    "text_config": {
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "num_attention_heads": HEADS,
        "head_dim": HEAD_DIM,
        "qk_rope_head_dim": ROPE_DIM,
        "q_lora_rank": Q_LORA,
        "o_lora_rank": O_LORA,
        "o_groups": O_GROUPS,
        "sliding_window": WINDOW,
        "swiglu_limit": 10.0,
        "rope_theta": 10000,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 2,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 64,
        },
        "compress_rope_theta": 20000,
        "compress_ratios": RATIOS,
        "moe_intermediate_size": INTER,
        "n_routed_experts": EXPERTS,
        "n_shared_experts": 1,
        "num_experts_per_tok": TOPK,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
        "hc_mult": HC_MULT,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-6,
        "index_n_heads": INDEX_HEADS,
        "index_head_dim": INDEX_DIM,
        "index_topk": 4,
        "candidate_source_layer": CANDIDATE_SOURCE,
        "candidate_topk_blocks": 4,
        "candidate_block_size": 2,
        "kv_source_layer_ids": KV_SOURCES,
        "index_source_layer_ids": INDEX_SOURCES,
        "engram_layer_ids": [],
    },
}

FAMILY_MACRO_NAMES = {
    "END_OF_TEXT_TOKEN_ID": "EOS_TOKEN_ID",
    "SLIDING_WINDOW_TOKENS": "SLIDING_WINDOW",
    "MOE_ROUTED_EXPERT_COUNT": "ROUTED_EXPERT_COUNT",
    "MOE_EXPERTS_PER_TOKEN": "EXPERTS_PER_TOKEN",
    "CANDIDATE_TOPK_BLOCKS": "CANDIDATE_BLOCK_COUNT",
}

HEADER_VALUES = {
    "HIDDEN_DIMENSION": f"{HIDDEN}u",
    "LAYER_COUNT": f"{LAYERS}u",
    "OUTPUT_VOCAB_COUNT": f"{VOCAB}u",
    "RMS_NORM_EPSILON": "1e-05f",
    "END_OF_TEXT_TOKEN_ID": "31u",
    "ATTENTION_HEAD_COUNT": f"{HEADS}u",
    "HEAD_DIMENSION": f"{HEAD_DIM}u",
    "QK_ROPE_HEAD_DIMENSION": f"{ROPE_DIM}u",
    "QUERY_LORA_RANK": f"{Q_LORA}u",
    "OUTPUT_LORA_RANK": f"{O_LORA}u",
    "OUTPUT_GROUP_COUNT": f"{O_GROUPS}u",
    "SLIDING_WINDOW_TOKENS": f"{WINDOW}u",
    "SWIGLU_LIMIT": "10.0f",
    "ROPE_THETA": "10000.0f",
    "YARN_FACTOR": "2.0f",
    "YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS": "64u",
    "YARN_BETA_FAST": "32u",
    "YARN_BETA_SLOW": "1u",
    "COMPRESS_ROPE_THETA": "20000.0f",
    "MOE_ROUTED_EXPERT_COUNT": f"{EXPERTS}u",
    "MOE_EXPERTS_PER_TOKEN": f"{TOPK}u",
    "MOE_INTERMEDIATE_DIMENSION": f"{INTER}u",
    "MOE_ROUTED_SCALING_FACTOR": "2.5f",
    "HC_MULT": f"{HC_MULT}u",
    "HC_SINKHORN_ITERATIONS": "3u",
    "HC_EPS": "1e-06f",
    "INDEX_HEAD_COUNT": f"{INDEX_HEADS}u",
    "INDEX_HEAD_DIMENSION": f"{INDEX_DIM}u",
    "INDEX_TOP_K": "4u",
    "CANDIDATE_SOURCE_LAYER": f"{CANDIDATE_SOURCE}u",
    "CANDIDATE_TOPK_BLOCKS": "4u",
    "CANDIDATE_BLOCK_SIZE": "2u",
    "KV_SOURCE_LAYER_0": f"{KV_SOURCES[0]}u",
    "KV_SOURCE_LAYER_1": f"{KV_SOURCES[1]}u",
    "KV_SOURCE_LAYER_2": f"{KV_SOURCES[2]}u",
    "KV_SOURCE_LAYER_3": f"{KV_SOURCES[3]}u",
    "INDEX_SOURCE_LAYER_0": f"{INDEX_SOURCES[0]}u",
    "INDEX_SOURCE_LAYER_1": f"{INDEX_SOURCES[1]}u",
    "INDEX_SOURCE_LAYER_2": f"{INDEX_SOURCES[2]}u",
    "INDEX_SOURCE_LAYER_3": f"{INDEX_SOURCES[3]}u",
    "INDEX_SOURCE_LAYER_4": f"{INDEX_SOURCES[4]}u",
    "INDEX_SOURCE_LAYER_5": f"{INDEX_SOURCES[5]}u",
    "INDEX_SOURCE_LAYER_6": f"{INDEX_SOURCES[6]}u",
    "INDEX_SOURCE_LAYER_7": f"{INDEX_SOURCES[7]}u",
    "ENGRAM_MODULE_COUNT": "0u",
    "ENGRAM_MAX_NGRAM_SIZE": "1u",
    "ENGRAM_COMPRESSED_VOCAB_SIZE": "1u",
}


def write_defines(path, ratios=None, hidden=None):
    lines = ["#pragma once", "",
             "#define SPARK_LLM_FAMILY_TAG dsv41_synth"]
    values = dict(HEADER_VALUES)
    if hidden is not None:
        values["HIDDEN_DIMENSION"] = f"{hidden}u"
    for name, value in values.items():
        family = FAMILY_MACRO_NAMES.get(name, name)
        lines.append(f"#define SPARK_DSV41_FLASH_MODEL_{family} {value}")
        lines.append(f"#define SPARK_LLM_{name} "
                     f"SPARK_DSV41_FLASH_MODEL_{family}")
    if ratios is None:
        ratios = RATIOS
    body = ", ".join(f"{v}u" for v in ratios)
    lines.append(
        "#define SPARK_DSV41_FLASH_MODEL_LAYER_COMPRESSION_RATIO"
        f"(layer_index) ((uint32_t[]){{{body}}}[(layer_index)])")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def bf16(shape, rng, scale=0.05):
    return f32_to_bf16_u16(
        rng.standard_normal(shape).astype(np.float32) * scale)


def fp8(shape, rng, scale_byte=127):
    payload = rng.integers(0x08, 0x38, size=shape, dtype=np.uint8)
    grid = (max(1, shape[0] // 32), max(1, shape[1] // 32))
    scale = np.full(grid, scale_byte, dtype=np.uint8)
    return payload, scale


def mxfp4(shape, rng):
    nibbles = rng.integers(0, 12, size=shape, dtype=np.uint8)
    payload = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4))
    scale = np.full((shape[0], shape[1] // 32), 127, dtype=np.uint8)
    return payload, scale


def build_tensors(rng):
    t = {}
    t["embed.weight"] = bf16((VOCAB, HIDDEN), rng)
    head = bf16((VOCAB, HIDDEN), rng, scale=0.5)
    head[31] = np.zeros(HIDDEN, dtype=np.uint16)
    t["head.weight"] = head
    t["norm.weight"] = bf16((HIDDEN,), rng)
    for layer in range(LAYERS):
        p = f"layers.{layer}."
        t[p + "attn.attn_sink"] = rng.standard_normal(
            HEADS).astype(np.float32)
        for name, shape in [("wq_a", (Q_LORA, HIDDEN)),
                            ("wq_b", (HEADS * HEAD_DIM, Q_LORA)),
                            ("wkv", (HEAD_DIM, HIDDEN)),
                            ("wo_a", (O_GROUPS * O_LORA,
                                      HEADS // O_GROUPS * HEAD_DIM)),
                            ("wo_b", (HIDDEN, O_GROUPS * O_LORA))]:
            payload, scale = fp8(shape, rng)
            t[p + f"attn.{name}.weight"] = payload
            t[p + f"attn.{name}.scale"] = scale
        t[p + "attn.q_norm.weight"] = bf16((Q_LORA,), rng)
        t[p + "attn.kv_norm.weight"] = bf16((HEAD_DIM,), rng)
        t[p + "attn_norm.weight"] = bf16((HIDDEN,), rng)
        t[p + "ffn_norm.weight"] = bf16((HIDDEN,), rng)
        t[p + "hc_attn_fn"] = rng.standard_normal(
            ((2 + HC_MULT) * HC_MULT, HC_MULT * HIDDEN)).astype(np.float32)
        t[p + "hc_attn_base"] = rng.standard_normal(
            (2 + HC_MULT) * HC_MULT).astype(np.float32)
        t[p + "hc_attn_scale"] = rng.standard_normal(3).astype(np.float32)
        t[p + "hc_ffn_fn"] = rng.standard_normal(
            ((2 + HC_MULT) * HC_MULT, HC_MULT * HIDDEN)).astype(np.float32)
        t[p + "hc_ffn_base"] = rng.standard_normal(
            (2 + HC_MULT) * HC_MULT).astype(np.float32)
        t[p + "hc_ffn_scale"] = rng.standard_normal(3).astype(np.float32)
        t[p + "ffn.gate.weight"] = bf16((EXPERTS, HIDDEN), rng, scale=0.2)
        t[p + "ffn.gate.bias"] = rng.standard_normal(
            EXPERTS).astype(np.float32)
        for expert in range(EXPERTS):
            ep = p + f"ffn.experts.{expert}."
            for name, shape in [("w1", (INTER, HIDDEN)),
                                ("w2", (HIDDEN, INTER)),
                                ("w3", (INTER, HIDDEN))]:
                payload, scale = mxfp4(shape, rng)
                t[ep + f"{name}.weight"] = payload
                t[ep + f"{name}.scale"] = scale
        for name, shape in [("w1", (INTER, HIDDEN)), ("w2", (HIDDEN, INTER)),
                            ("w3", (INTER, HIDDEN))]:
            payload, scale = fp8(shape, rng)
            t[p + f"ffn.shared_experts.{name}.weight"] = payload
            t[p + f"ffn.shared_experts.{name}.scale"] = scale
        if layer in KV_SOURCES:
            cp = p + "attn.compressor."
            t[cp + "wkv.weight"] = bf16((HEAD_DIM, HIDDEN), rng)
            t[cp + "norm.weight"] = bf16((HEAD_DIM,), rng)
            if RATIOS[layer] > 1:
                t[cp + "wgate.weight"] = bf16((HEAD_DIM, HIDDEN), rng)
        if layer in INDEX_SOURCES:
            ip = p + "attn.indexer."
            payload, scale = fp8((INDEX_HEADS * INDEX_DIM, Q_LORA), rng)
            t[ip + "wq_b.weight"] = payload
            t[ip + "wq_b.scale"] = scale
            t[ip + "weights_proj.weight"] = bf16((INDEX_HEADS, HIDDEN), rng)
            if layer in KV_SOURCES:
                t[ip + "wk.weight"] = bf16((INDEX_DIM, HEAD_DIM), rng)
                t[ip + "k_norm.weight"] = bf16((INDEX_DIM,), rng)
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
        elif array.dtype == np.float32:
            dtype, itemsize = "F32", 4
        else:
            raise AssertionError(f"unexpected dtype {array.dtype}")
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


def write_checkpoint(directory, rng_seed=7, mutate=None):
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    tensors = build_tensors(rng)
    if mutate:
        name, index, value = mutate
        tensors[name].reshape(-1)[index] = value
    write_safetensors(os.path.join(directory, "model.safetensors"), tensors)
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(CONFIG, fh)


def write_prompts(path):
    document = {"prompts": [{
        "name": "synth_a",
        "prompt_token_ids": [3, 7, 11, 5],
        "new_tokens": 3,
        "capture_layers": [0, 5, 11],
    }]}
    with open(path, "w") as fh:
        json.dump(document, fh)


def run_generator(checkpoint, header, prompts, output, env=None):
    environ = dict(os.environ)
    if env:
        environ.update(env)
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
         "--family", "dsv41", "--checkpoint", checkpoint,
         "--header", header, "--prompts", prompts, "--output", output],
        capture_output=True, text=True, env=environ)
    return result


def compare(reference, candidate):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_compare.py"),
         "compare", "--reference", reference, "--candidate", candidate],
        capture_output=True, text=True)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def check_hc_post_orientation():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import numpy as np
    from t1_reference_dsv41 import Dsv41FlashEngine
    rng = np.random.default_rng(3)
    comb = rng.random((4, 4)).astype(np.float32)
    residual = rng.random((4, 8)).astype(np.float32)
    post = rng.random(4).astype(np.float32)
    x = rng.random(8).astype(np.float32)
    got = Dsv41FlashEngine._hc_post(None, x, residual, post, comb)
    official = post[:, None] * x[None, :] + np.sum(comb[:, :, None] * residual[:, None, :], axis=0)
    expect(np.allclose(got, official, rtol=1e-2, atol=1e-2),
           "hc_post must follow the reference sum over comb rows (comb^T @ residual)")
    expect(not np.allclose(got, post[:, None] * x[None, :] + comb @ residual, rtol=1e-3, atol=1e-3),
           "hc_post fixture comb must be asymmetric enough to separate the orientations")


def check_round_half_even():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from t1_reference_dsv41 import _fp4_round_abs, _fp8_round_abs
    fp4 = _fp4_round_abs(np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                                  dtype=np.float32))
    expect(fp4.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0],
           f"fp4 ties must round to the even code (cvt.rn), got {fp4}")
    fp8 = _fp8_round_abs(np.array([0.96875, 1.0625, 17.0], dtype=np.float32))
    expect(fp8.tolist() == [1.0, 1.0, 16.0],
           f"fp8 e4m3 ties must round to the even code (cvt.rn), got {fp8}")


def main():
    check_round_half_even()
    check_hc_post_orientation()
    workspace = tempfile.mkdtemp(prefix="t1ref-dsv41-")
    try:
        checkpoint = os.path.join(workspace, "checkpoint")
        write_checkpoint(checkpoint)
        header = os.path.join(workspace, "llm_defines.h")
        write_defines(header)
        prompts = os.path.join(workspace, "prompts.json")
        write_prompts(prompts)
        out_a = os.path.join(workspace, "run_a")
        out_b = os.path.join(workspace, "run_b")
        result_a = run_generator(checkpoint, header, prompts, out_a)
        expect(result_a.returncode == 0,
               f"generator failed: {result_a.stderr}")
        result_b = run_generator(checkpoint, header, prompts, out_b)
        expect(result_b.returncode == 0,
               f"second generator run failed: {result_b.stderr}")
        fixture_a = os.path.join(out_a, "dsv41", "synth_a.t1r")
        fixture_b = os.path.join(out_b, "dsv41", "synth_a.t1r")
        expect(os.path.exists(fixture_a), "fixture missing after generation")
        expect(open(fixture_a, "rb").read() == open(fixture_b, "rb").read(),
               "generator is not byte-deterministic")
        _, arrays = read_fixture(fixture_a)
        for layer in (0, 5, 11):
            for position in range(7):
                expect(f"pos{position:04d}_layer{layer:04d}_streams"
                       in arrays, f"stream capture missing at {position}/{layer}")
        expect("pos0004_layer0005_route_ids" in arrays,
               "route capture missing")
        expect("pos0004_layer0005_route_weights" in arrays,
               "route weight capture missing")
        weights = arrays["pos0004_layer0005_route_weights"]
        expect(weights.shape[0] == TOPK, "unexpected route width")
        expect(abs(float(weights.sum()) - 2.5) < 1e-5,
               f"normalized route weights must sum to the route scale, "
               f"got {weights.sum()}")
        expect(int(arrays["generated_token_ids"][0]) >= 0,
               "no generated tokens")
        manifest = json.load(open(os.path.join(out_a, "dsv41",
                                               "MANIFEST.json")))
        import hashlib
        expect(manifest["fixtures"]["synth_a.t1r"]["sha256"] ==
               hashlib.sha256(open(fixture_a, "rb").read()).hexdigest(),
               "manifest sha mismatch")
        expect(manifest["defines_config_mismatches"] ==
               manifest["defines_config_mismatches"], "manifest malformed")
        identical = compare(fixture_a, fixture_b)
        expect(identical.returncode == 0,
               f"identical fixtures must PASS: {identical.stdout}")
        mutated = os.path.join(workspace, "checkpoint_mutated")
        write_checkpoint(mutated, mutate=(
            "layers.3.ffn.shared_experts.w2.scale", 0, 147))
        out_c = os.path.join(workspace, "run_c")
        result_c = run_generator(mutated, header, prompts, out_c)
        expect(result_c.returncode == 0,
               f"mutated generator failed: {result_c.stderr}")
        fixture_c = os.path.join(out_c, "dsv41", "synth_a.t1r")
        diverged = compare(fixture_a, fixture_c)
        expect(diverged.returncode == 1,
               "perturbed expert weight must change the output fixtures")
        compressed = os.path.join(workspace, "checkpoint_compressed")
        write_checkpoint(compressed, mutate=(
            "layers.1.attn.compressor.norm.weight", 0, 0x4100))
        out_h = os.path.join(workspace, "run_h")
        result_h = run_generator(compressed, header, prompts, out_h)
        expect(result_h.returncode == 0,
               f"compressor-perturbed generator failed: {result_h.stderr}")
        attended = compare(fixture_a, os.path.join(out_h, "dsv41",
                                                   "synth_a.t1r"))
        expect(attended.returncode == 1,
               "compressed KV rows must be attended: perturbing the "
               "ratio-2 compressor norm must change the output fixtures")
        bad_header = os.path.join(workspace, "llm_defines_bad.h")
        write_defines(bad_header, hidden=17)
        mismatch = run_generator(checkpoint, bad_header, prompts,
                                 os.path.join(workspace, "run_d"))
        expect(mismatch.returncode != 0,
               "defines/config disagreement must fail loud")
        expect("HIDDEN_DIMENSION" in mismatch.stderr,
               f"failure must name the mismatched define: {mismatch.stderr}")
        good_family = os.path.join(workspace, "family_ok.h")
        write_defines(good_family, ratios=RATIOS)
        ok = run_generator(checkpoint, header, prompts,
                           os.path.join(workspace, "run_e"),
                           env={"DSV41_FAMILY_HEADER": good_family})
        expect(ok.returncode == 0,
               f"family header closure failed: {ok.stderr}")
        bad_ratios_family = os.path.join(workspace, "family_ratios.h")
        divergent = list(RATIOS)
        divergent[6] = 2
        write_defines(bad_ratios_family, ratios=divergent)
        recorded = run_generator(checkpoint, header, prompts,
                                 os.path.join(workspace, "run_f"),
                                 env={"DSV41_FAMILY_HEADER":
                                      bad_ratios_family})
        expect(recorded.returncode == 0,
               f"ratio-table divergence must not block generation: "
               f"{recorded.stderr}")
        recorded_manifest = json.load(open(
            os.path.join(workspace, "run_f", "dsv41", "MANIFEST.json")))
        expect(any("family_header_ratio_table" in row for row
                   in recorded_manifest["defines_config_mismatches"]),
               "ratio-table divergence must be recorded in the manifest")
        bad_family = os.path.join(workspace, "family_bad.h")
        write_defines(bad_family, ratios=RATIOS, hidden=17)
        failed = run_generator(checkpoint, header, prompts,
                               os.path.join(workspace, "run_g"),
                               env={"DSV41_FAMILY_HEADER": bad_family})
        expect(failed.returncode != 0,
               "family header scalar disagreement must fail loud")
        expect("disagrees" in failed.stderr,
               f"failure must name the disagreement: {failed.stderr}")
        shutil.rmtree(workspace, ignore_errors=True)
        print("PASS t1_reference_dsv41 synthetic proof: determinism, "
              "route-scale contract, negative control, defines/config "
              "fail-closed, family-header closure")
        return 0
    except AssertionError:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
