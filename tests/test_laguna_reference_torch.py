import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import torch

import laguna_reference_torch as ref

HIDDEN = 64
HEAD_DIM = 16
KV_HEADS = 2
HEADS = [4, 6, 6]
EXPERTS = 8
TOP_K = 2
INTER = 32
DENSE = 96
VOCAB = 50


def config():
    return {
        "model_type": "laguna", "vocab_size": VOCAB, "hidden_size": HIDDEN, "intermediate_size": DENSE,
        "num_hidden_layers": 3, "num_attention_heads": HEADS[0], "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6, "num_experts": EXPERTS, "num_experts_per_tok": TOP_K, "moe_intermediate_size": INTER,
        "shared_expert_intermediate_size": INTER, "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": [0],
        "eos_token_id": [2, 24], "gating": "per-head", "sliding_window": 4, "moe_routed_scaling_factor": 2.5,
        "moe_apply_router_weight_on_input": False, "moe_router_logit_softcapping": 0.0,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention"],
        "num_attention_heads_per_layer": HEADS,
        "rope_parameters": {
            "full_attention": {"rope_theta": 500000.0, "rope_type": "yarn", "factor": 128.0, "original_max_position_embeddings": 8192,
                               "beta_slow": 1.0, "beta_fast": 32.0, "attention_factor": 1.4852030263919618, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0}},
    }


def tensors(bias_key, bias_expert):
    generator = torch.Generator().manual_seed(7)

    def rand(*shape, scale=0.2):
        return torch.randn(*shape, generator=generator) * scale

    out = {"model.embed_tokens.weight": rand(VOCAB, HIDDEN, scale=1.0), "model.norm.weight": 1 + rand(HIDDEN, scale=0.05), "lm_head.weight": rand(VOCAB, HIDDEN)}
    for layer, heads in enumerate(HEADS):
        p = f"model.layers.{layer}."
        out[p + "input_layernorm.weight"] = 1 + rand(HIDDEN, scale=0.05)
        out[p + "post_attention_layernorm.weight"] = 1 + rand(HIDDEN, scale=0.05)
        out[p + "self_attn.q_proj.weight"] = rand(heads * HEAD_DIM, HIDDEN)
        out[p + "self_attn.k_proj.weight"] = rand(KV_HEADS * HEAD_DIM, HIDDEN)
        out[p + "self_attn.v_proj.weight"] = rand(KV_HEADS * HEAD_DIM, HIDDEN)
        out[p + "self_attn.o_proj.weight"] = rand(HIDDEN, heads * HEAD_DIM)
        out[p + "self_attn.g_proj.weight"] = rand(heads, HIDDEN)
        out[p + "self_attn.q_norm.weight"] = 1 + rand(HEAD_DIM, scale=0.05)
        out[p + "self_attn.k_norm.weight"] = 1 + rand(HEAD_DIM, scale=0.05)
        if layer == 0:
            out[p + "mlp.gate_proj.weight"] = rand(DENSE, HIDDEN)
            out[p + "mlp.up_proj.weight"] = rand(DENSE, HIDDEN)
            out[p + "mlp.down_proj.weight"] = rand(HIDDEN, DENSE)
            continue
        out[p + "mlp.gate.weight"] = rand(EXPERTS, HIDDEN)
        if bias_key is not None:
            bias = torch.zeros(EXPERTS)
            bias[bias_expert] = 4.0
            out[p + f"mlp.{bias_key}.e_score_correction_bias"] = bias
        for name in ("shared_expert.", *[f"experts.{e}." for e in range(EXPERTS)]):
            out[p + "mlp." + name + "gate_proj.weight"] = rand(INTER, HIDDEN)
            out[p + "mlp." + name + "up_proj.weight"] = rand(INTER, HIDDEN)
            out[p + "mlp." + name + "down_proj.weight"] = rand(HIDDEN, INTER)
    return out


def write_checkpoint(directory, bias_key="experts", bias_expert=3):
    names = tensors(bias_key, bias_expert)
    header = {}
    blobs = []
    offset = 0
    for name, tensor in names.items():
        raw = tensor.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
        header[name] = {"dtype": "BF16", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    with open(os.path.join(directory, "model-00001-of-00001.safetensors"), "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for raw in blobs:
            handle.write(raw)
    json.dump({"weight_map": {name: "model-00001-of-00001.safetensors" for name in names}}, open(os.path.join(directory, "model.safetensors.index.json"), "w"))
    json.dump(config(), open(os.path.join(directory, "config.json"), "w"))


def full_recompute_logits(model, tokens):
    cache = [None] * model.layers
    positions = torch.arange(len(tokens))
    _, exact = model.forward(tokens, positions, cache, [], [])
    return exact


def check_incremental_matches_recompute(failures):
    with tempfile.TemporaryDirectory() as directory:
        write_checkpoint(directory)
        model = ref.Laguna(directory, torch.device("cpu"), 64)
        prompt = [5, 9, 11, 17, 23]
        generated, steps = model.generate(prompt, 6, set())
        tokens = list(prompt)
        for index, step in enumerate(steps):
            exact = full_recompute_logits(model, tokens)
            if int(torch.argmax(exact)) != step["fp32_top2_tokens"][0]:
                failures.append(f"step {index}: recompute top1 {int(torch.argmax(exact))} vs incremental {step['fp32_top2_tokens'][0]}")
            tokens.append(generated[index])
        if len(tokens) - len(prompt) != 6:
            failures.append("generation stopped early")
        if len(tokens) <= config()["sliding_window"]:
            failures.append("sequence never crossed the sliding window")


def check_correction_bias_is_read(failures):
    for key in ("experts", "gate"):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, bias_key=key, bias_expert=6)
            model = ref.Laguna(directory, torch.device("cpu"), 64)
            _, steps = model.generate([4, 8, 15], 2, set())
            for step in steps:
                for route in step["routes"]:
                    if 6 not in route["experts"]:
                        failures.append(f"bias under mlp.{key} ignored at layer {route['layer']}: experts {route['experts']}")
                    if abs(sum(route["weights"]) - 1.0) > 1e-2:
                        failures.append(f"route weights under mlp.{key} not renormalised: {route['weights']}")


def check_missing_bias_fails(failures):
    with tempfile.TemporaryDirectory() as directory:
        write_checkpoint(directory, bias_key=None)
        model = ref.Laguna(directory, torch.device("cpu"), 64)
        try:
            model.generate([4, 8], 1, set())
            failures.append("a checkpoint without a router correction bias must be refused")
        except ValueError as error:
            if "correction bias" not in str(error):
                failures.append(f"unexpected refusal text: {error}")


def check_yarn_matches_numpy_reference(failures):
    import t1_reference_laguna as numpy_reference
    rope = config()["rope_parameters"]["full_attention"]
    inv_freq, scaling = ref.yarn_inv_freq(rope, 128)
    engine = numpy_reference.LagunaEngine.__new__(numpy_reference.LagunaEngine)
    engine.rotary = 64
    engine.yarn_theta = rope["rope_theta"]
    engine.yarn_factor = rope["factor"]
    engine.yarn_positions = float(rope["original_max_position_embeddings"])
    engine.yarn_beta_fast = rope["beta_fast"]
    engine.yarn_beta_slow = rope["beta_slow"]
    table = engine.yarn_table().astype(np.float32)
    if inv_freq.shape[0] != 32 or not np.allclose(inv_freq.numpy(), table, rtol=1e-6, atol=0):
        failures.append("yarn inverse frequencies disagree with tools/t1_reference_laguna.py")
    if abs(scaling - rope["attention_factor"]) > 0:
        failures.append("yarn attention factor must be the configured value")
    sliding, one = ref.default_inv_freq(config()["rope_parameters"]["sliding_attention"], 128)
    expected = 1.0 / (10000.0 ** (np.arange(0, 128, 2) / 128.0))
    if one != 1.0 or not np.allclose(sliding.numpy(), expected, rtol=1e-6):
        failures.append("sliding rope must be theta 1e4 over the whole head")


def check_numpy_reference_bias(failures):
    import t1_reference_common as common
    import t1_reference_laguna as numpy_reference
    for key in ("experts", "gate", None):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoint(directory, bias_key=key, bias_expert=5)
            engine = numpy_reference.LagunaEngine.__new__(numpy_reference.LagunaEngine)
            engine.st = common.Safetensors(directory)
            try:
                bias = engine.correction_bias(1)
            except numpy_reference.LagunaConfigError:
                if key is not None:
                    failures.append(f"numpy reference refused the bias under mlp.{key}")
                continue
            if key is None:
                failures.append("numpy reference must refuse a layer without a correction bias")
            elif int(np.argmax(bias)) != 5 or abs(float(bias[5]) - 4.0) > 0:
                failures.append(f"numpy reference read the bias under mlp.{key} wrongly: {bias}")


def check_unsupported_config_fails(failures):
    with tempfile.TemporaryDirectory() as directory:
        write_checkpoint(directory)
        document = config()
        document["swa_attention_sink_enabled"] = True
        json.dump(document, open(os.path.join(directory, "config.json"), "w"))
        try:
            ref.Laguna(directory, torch.device("cpu"), 64)
            failures.append("attention sinks must be refused")
        except ValueError:
            pass


def main():
    failures = []
    check_incremental_matches_recompute(failures)
    check_correction_bias_is_read(failures)
    check_missing_bias_fails(failures)
    check_yarn_matches_numpy_reference(failures)
    check_unsupported_config_fails(failures)
    check_numpy_reference_bias(failures)
    for failure in failures:
        print("FAIL " + failure)
    if failures:
        return 1
    print("PASS laguna torch reference: incremental decode, correction bias, yarn, refusals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
