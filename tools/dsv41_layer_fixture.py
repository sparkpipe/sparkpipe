#!/usr/bin/env python3
"""Export one DeepSeek V4.1-Flash layer's attention weights, rope table and
publisher-code captures for tests/test_dsv41_flash_layer.cu.

Usage: dsv41_layer_fixture.py --checkpoint DIR --capture NPZ --layer 0 --output DIR
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import Safetensors  # noqa: E402

ATTENTION_TENSORS = ("wq_a.weight", "wq_a.scale", "q_norm.weight", "wq_b.weight", "wq_b.scale",
                     "wkv.weight", "wkv.scale", "kv_norm.weight", "attn_sink", "wo_a.weight", "wo_a.scale",
                     "wo_b.weight", "wo_b.scale")


def rope_table(inference_dir, config, layer, positions):
    import torch
    import dsv41_official_harness as harness
    sys.path.insert(0, inference_dir)
    harness.install_kernel_module()
    import model as publisher
    ratio = config["compress_ratios"][layer]
    if ratio:
        original, theta = config["original_seq_len"], config["compress_rope_theta"]
    else:
        original, theta = 0, config["rope_theta"]
    freqs = publisher.precompute_freqs_cis(config["rope_head_dim"], positions, original, theta,
                                           config["rope_factor"], config["beta_fast"], config["beta_slow"])
    return torch.view_as_real(freqs).float().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--call", type=int, default=0)
    arguments = parser.parse_args()
    inference_dir = os.path.join(arguments.checkpoint, "inference")
    config = json.load(open(os.path.join(inference_dir, "config.json")))
    tensors = Safetensors(arguments.checkpoint)
    os.makedirs(arguments.output, exist_ok=True)
    layer = arguments.layer
    manifest = {"layer": layer, "tensors": {}, "config": {key: config[key] for key in (
        "dim", "n_heads", "head_dim", "rope_head_dim", "q_lora_rank", "o_groups", "o_lora_rank",
        "window_size", "norm_eps", "n_routed_experts", "n_activated_experts", "moe_inter_dim", "route_scale",
        "swiglu_limit")}}
    for suffix in ATTENTION_TENSORS:
        name = f"layers.{layer}.attn.{suffix}"
        array = np.ascontiguousarray(tensors.pread(name))
        array.tofile(os.path.join(arguments.output, name + ".bin"))
        manifest["tensors"][name] = {"shape": list(array.shape), "dtype": tensors.entry(name)["dtype"]}
    ratio = config["compress_ratios"][layer]
    np.array([ratio if layer in config["kv_source_layers"] else 0], dtype=np.uint32).tofile(os.path.join(arguments.output, "ratio.bin"))
    manifest["compress_ratio"] = ratio
    if ratio > 1 and layer in config["kv_source_layers"]:
        for suffix in ("attn.compressor.wkv.weight", "attn.compressor.wgate.weight", "attn.compressor.norm.weight"):
            name = f"layers.{layer}.{suffix}"
            array = np.ascontiguousarray(tensors.pread(name))
            array.tofile(os.path.join(arguments.output, name + ".bin"))
            manifest["tensors"][name] = {"shape": list(array.shape), "dtype": tensors.entry(name)["dtype"]}
    for suffix in ("ffn.gate.weight", "ffn.gate.bias", "ffn.shared_experts.w1.weight", "ffn.shared_experts.w1.scale",
                   "ffn.shared_experts.w2.weight", "ffn.shared_experts.w2.scale", "ffn.shared_experts.w3.weight",
                   "ffn.shared_experts.w3.scale"):
        name = f"layers.{layer}.{suffix}"
        array = np.ascontiguousarray(tensors.pread(name))
        array.tofile(os.path.join(arguments.output, name + ".bin"))
        manifest["tensors"][name] = {"shape": list(array.shape), "dtype": tensors.entry(name)["dtype"]}
    for suffix in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
                   "attn_norm.weight", "ffn_norm.weight"):
        name = f"layers.{layer}.{suffix}"
        array = np.ascontiguousarray(tensors.pread(name))
        array.tofile(os.path.join(arguments.output, name + ".bin"))
        manifest["tensors"][name] = {"shape": list(array.shape), "dtype": tensors.entry(name)["dtype"]}
    captures = np.load(arguments.capture)
    prefix = f"call{arguments.call:03d}_layer{layer:02d}_"
    captures[prefix + "input"].tofile(os.path.join(arguments.output, "layer_in.bin"))
    captures[prefix + "streams"].tofile(os.path.join(arguments.output, "layer_out.bin"))
    streams_shape = captures[prefix + "input"].shape
    if layer == 0:
        pre_mix = np.zeros(streams_shape[:-1], dtype=np.float32)
        pre_mix[..., 0] = 1.0
    else:
        pre_mix = captures[f"call{arguments.call:03d}_layer{layer - 1:02d}_pre_mix_out"].astype(np.float32)
    pre_mix.tofile(os.path.join(arguments.output, "pre_mix_in.bin"))
    ffn_in = captures[prefix + "ffn_in"].reshape(-1, config["dim"])
    words = ffn_in.astype(np.uint32) << 16
    gate = tensors.pread(f"layers.{layer}.ffn.gate.weight").astype(np.uint32) << 16
    scores = words.view(np.float32) @ gate.view(np.float32).T
    routed = np.sqrt(np.where(scores > 20.0, scores, np.log1p(np.exp(np.minimum(scores, 20.0)))))
    biased = routed + tensors.pread(f"layers.{layer}.ffn.gate.bias")[None, :]
    experts = sorted(set(np.argsort(-biased, axis=1)[:, :8].reshape(-1).tolist()))
    for expert in experts:
        for part in ("w1", "w2", "w3"):
            for kind in ("weight", "scale"):
                name = f"layers.{layer}.ffn.experts.{expert}.{part}.{kind}"
                np.ascontiguousarray(tensors.pread(name)).tofile(os.path.join(arguments.output, name + ".bin"))
    manifest["experts"] = experts
    for key in ("attn_in", "attn_out", "ffn_in", "ffn_out"):
        array = captures[prefix + key]
        array.reshape(-1, array.shape[-1]).tofile(os.path.join(arguments.output, key + ".bin"))
        manifest[key] = list(array.shape)
    positions = manifest["attn_in"][1]
    rope_table(inference_dir, config, layer, positions).astype(np.float32).tofile(os.path.join(arguments.output, "rope.bin"))
    manifest["positions"] = positions
    json.dump(manifest, open(os.path.join(arguments.output, "manifest.json"), "w"), indent=1)
    print(json.dumps({"output": arguments.output, "positions": positions}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
