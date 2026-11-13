#!/usr/bin/env python3
"""The Qwen hybrid packer takes its geometry from the checkpoint's config.json: Qwen3.8-27B keeps its separate head, Qwen3.5-4B and 2B tie the head to the embedding, and a checkpoint whose fixed dimensions or layer pattern differ from the stages' is refused before any tensor is read. An F32 checkpoint tensor the stage reads as BF16 is rounded to nearest even, ties to the even neighbour, as the publisher's bfloat16 load rounds it."""
import copy
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen38_pack", ROOT / "tools" / "qwen38_27b_tp16_pack.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)
failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


def text_config(hidden, layers, ffn, value_heads, heads, kv_heads):
    return {"hidden_size": hidden, "num_hidden_layers": layers, "vocab_size": 248320, "intermediate_size": ffn, "num_attention_heads": heads,
            "num_key_value_heads": kv_heads, "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": value_heads,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "full_attention_interval": 4,
            "layer_types": ["full_attention" if layer % 4 == 3 else "linear_attention" for layer in range(layers)], "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_interleaved": True}}


models = {
    "qwen38_27b": ({"tie_word_embeddings": False, "text_config": text_config(5120, 64, 17408, 48, 24, 4)}, 16, False, 14336),
    "qwen35_4b": ({"tie_word_embeddings": True, "text_config": text_config(2560, 32, 9216, 32, 16, 4)}, 1, True, 10240),
    "qwen35_2b": ({"tie_word_embeddings": True, "text_config": text_config(2048, 24, 6144, 16, 8, 2)}, 1, True, 5120),
}
for name, (config, tp, tied, qkv) in models.items():
    pack.TP = tp
    pack.configure(config)
    check((pack.LAYERS, pack.HIDDEN, pack.TIED, pack.ATTN_QKV, pack.FULL_PHASE) == (config["text_config"]["num_hidden_layers"],
                                                                                    config["text_config"]["hidden_size"], tied, qkv, 3),
          f"{name}: layers {pack.LAYERS}, hidden {pack.HIDDEN}, tied {pack.TIED}, fused attention rows {pack.ATTN_QKV}")
pack.TP = 1
for label, change in (("head dimension", ("head_dim", 128)), ("convolution", ("linear_conv_kernel_dim", 2)),
                      ("rotary span", ("rope_parameters", {"rope_theta": 10000000, "partial_rotary_factor": 0.5})),
                      ("layer pattern", ("layer_types", ["linear_attention"] * 32))):
    config = copy.deepcopy(models["qwen35_4b"][0])
    config["text_config"][change[0]] = change[1]
    try:
        pack.configure(config)
        failures.append(f"a checkpoint with another {label} is refused")
    except pack.PackFailure:
        pass
pack.TP = 16
try:
    pack.configure(models["qwen35_2b"][0])
    failures.append("a geometry that does not split into the ranks is refused")
except pack.PackFailure:
    pass


class Source:
    def __init__(self, dtype, values):
        self.dtype, self.values = dtype, values

    def array(self, name):
        return self.dtype, self.values


tie_down, tie_up = 1.0 + 2.0 ** -8, 1.0 + 3.0 * 2.0 ** -8
values = np.array([1.0, -2.5, tie_down, tie_up, 1.0 + 2.0 ** -8 + 2.0 ** -20], dtype=np.float32)
rounded = pack.bf16_rounded(Source("F32", values), "norm", (5,))
check(((rounded.astype(np.uint32) << 16).view(np.float32) == np.array([1.0, -2.5, 1.0, 1.015625, 1.0078125], dtype=np.float32)).all(),
      f"F32 values round to the nearest BF16, ties to even: {(rounded.astype(np.uint32) << 16).view(np.float32)}")
raw = np.array([0x3F80, 0xC020], dtype=np.uint16)
check(pack.bf16_rounded(Source("BF16", raw), "norm", (2,)) is raw, "a BF16 tensor passes through untouched")
for failure in failures:
    print("FAIL test_qwen38_pack_geometry:", failure)
if failures:
    raise SystemExit(1)
print("PASS test_qwen38_pack_geometry: the packer reads the 27B, 4B and 2B geometries and their heads from config.json, refuses other fixed dimensions, and rounds F32 sources to BF16 to nearest even")
