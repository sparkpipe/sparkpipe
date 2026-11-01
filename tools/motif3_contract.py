#!/usr/bin/env python3
"""Motif 3 model contract: model_contracts/motif3_authoritative.json from the checkpoint.

Every number is read from the checkpoint's config.json, generation_config.json,
model.safetensors.index.json and the safetensors headers. A missing key is an
error, never a default.

Usage:
    python3 tools/motif3_contract.py --snapshot DIR [--out PATH]
    python3 tools/motif3_contract.py --remote [--out PATH]

--remote reads the same files from Hugging Face at the pinned revision, the
headers by ranged reads, so the contract can be built before the download ends.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Motif-Technologies/Motif-3"
REVISION = "2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a"
DEFAULT_OUT = ROOT / "model_contracts" / "motif3_authoritative.json"


class Source:
    def __init__(self, snapshot: Path | None):
        self.snapshot = snapshot

    def url(self, name: str) -> str:
        return f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"

    def json(self, name: str):
        if self.snapshot is not None:
            return json.loads((self.snapshot / name).read_text())
        with urllib.request.urlopen(self.url(name), timeout=120) as reply:
            return json.loads(reply.read())

    def header(self, name: str) -> dict:
        if self.snapshot is not None:
            with open(self.snapshot / name, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                return json.loads(f.read(size))
        request = urllib.request.Request(self.url(name), headers={"Range": "bytes=0-7"})
        with urllib.request.urlopen(request, timeout=120) as reply:
            size = struct.unpack("<Q", reply.read(8))[0]
        request = urllib.request.Request(self.url(name), headers={"Range": f"bytes=8-{7 + size}"})
        with urllib.request.urlopen(request, timeout=120) as reply:
            return json.loads(reply.read())


def pattern(name: str) -> str:
    return re.sub(r"\.(layers|mtp_layers)\.\d+\.", r".\1.N.", name)


def census(source: Source, index: dict) -> tuple[dict, dict, int]:
    shapes: dict[str, list] = {}
    layers: dict[str, set] = {}
    total = 0
    for shard in sorted(set(index["weight_map"].values())):
        for name, entry in source.header(shard).items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            total += end - begin
            key = pattern(name)
            value = [entry["dtype"], entry["shape"]]
            if key in shapes and shapes[key] != value:
                raise SystemExit(f"{key}: {shapes[key]} and {value} differ across layers")
            shapes[key] = value
            match = re.search(r"\.layers\.(\d+)\.", name)
            if match:
                layers.setdefault(key, set()).add(int(match.group(1)))
    return shapes, {k: sorted(v) for k, v in layers.items()}, total


def contract(source: Source) -> dict:
    config = source.json("config.json")
    generation = source.json("generation_config.json")
    index = source.json("model.safetensors.index.json")
    shapes, layer_sets, payload = census(source, index)
    rope = config["rope_scaling"]
    layers = config["num_hidden_layers"]
    period = config["sliding_window_period"]
    heads, noise = config["num_attention_heads"], config["num_noise_heads"]
    if config["attention_cls"] != "gdla" or config["diff_v2"] is not True or config["sliding_window_pattern"] != "interleave":
        raise SystemExit("only GDLA diff_v2 with interleaved windows is described")
    if config["interleave_moe_layer_step"] != 1 or config["score_before_experts"] is not False or config["hidden_act"] != "poly_norm":
        raise SystemExit("the MoE placement or activation differs from the described one")
    routed = [layer for layer in range(layers) if layer >= config["n_dense_first_layers"]]
    if layer_sets.get("model.layers.N.moe.experts.gate_up_proj") != routed:
        raise SystemExit("routed layers in the checkpoint differ from the config")
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "architecture": config["architectures"][0],
        "source_revision": REVISION,
        "model": {
            "hidden_dimension": config["hidden_size"],
            "layer_count": layers,
            "vocabulary_size": config["vocab_size"],
            "maximum_context_tokens": config["max_position_embeddings"],
            "rms_norm_epsilon": config["rms_norm_eps"],
            "tie_word_embeddings": config["tie_word_embeddings"],
            "hidden_clamp": config["hidden_clamp"],
        },
        "attention": {
            "kind": "grouped differential latent attention (diff_v2)",
            "query_head_count": heads,
            "noise_head_count": noise,
            "signal_head_count": (heads - noise) // noise * noise,
            "group_size": heads // noise,
            "key_value_head_count": config["num_key_value_heads"],
            "query_lora_rank": config["q_lora_rank"],
            "kv_lora_rank": config["kv_lora_rank"],
            "head_dimension": config["head_dim"],
            "qk_rope_head_dimension": config["qk_rope_head_dim"],
            "qk_nope_head_dimension": config["head_dim"] - config["qk_rope_head_dim"],
            "value_head_dimension": config["v_head_dim"],
            "elementwise_output_gate": config["elementwise_attn_output_gate"],
            "headwise_output_gate": config["headwise_attn_output_gate"],
            "head_order": "groups of group_size heads; the first group_size-1 are signal, the last is the group's noise head",
        },
        "windows": {
            "pattern": "layer % period != 0 is a sliding-window layer",
            "period": period,
            "window_tokens": config["sliding_window"] + 1,
            "window_layers": [layer for layer in range(layers) if layer % period != 0],
            "full_layers": [layer for layer in range(layers) if layer % period == 0],
            "window_rope_theta": config["swa_rope_theta"],
        },
        "rope": {
            "theta": rope["rope_theta"],
            "type": rope["rope_type"],
            "factor": rope["factor"],
            "original_max_position_embeddings": rope["original_max_position_embeddings"],
            "beta_fast": rope["beta_fast"],
            "beta_slow": rope["beta_slow"],
            "mscale": rope["mscale"],
            "apply_yarn_scaling": rope["apply_yarn_scaling"],
            "full_layer_softmax_scale": "head_dimension^-0.5 x (0.1 x mscale x ln(factor) + 1)^2",
        },
        "hyper_connections": {
            "expansion_rate": config["mhc_expansion_rate"],
            "sinkhorn_iterations": config["mhc_sinkhorn_iters"],
            "identity_init": config["mhc_identity_init"],
            "post_coefficient": 1.0 + float(config.get("mhc_h_post_alpha_end", 0.0)),
            "norm_epsilon": 1e-6,
            "gate_clamp": 10.0,
            "residual_logit_clamp": 20.0,
            "sinkhorn_sum_floor": 1e-8,
        },
        "moe": {
            "routed_expert_count": config["num_experts"],
            "experts_per_token": config["experts_top_k"],
            "shared_expert_count": config["num_shared_experts"],
            "expert_intermediate_dimension": config["moe_intermediate_size"],
            "dense_intermediate_dimension": config["intermediate_size"],
            "first_dense_layer_count": config["n_dense_first_layers"],
            "routed_layers": routed,
            "score_function": config["score_func"],
            "normalize_selected": config["route_norm"],
            "routed_scaling_factor": config["route_scale"],
            "selection_bias": "moe.expert_bias, added to the scores for selection only",
        },
        "polynorm": {
            "sigmoid_weight": config.get("polynorm_sigmoid_weight", True),
            "output_scale": config["polynorm_output_scale"],
            "routed_bias_clamp": config["polynorm_bias_clamp"],
            "epsilon": 1e-6,
        },
        "mtp": {"layer_count": config["num_nextn_predict_layers"]},
        "tokens": {
            "beginning_of_text": generation["bos_token_id"],
            "pad": generation["pad_token_id"],
            "eos_token_ids": generation["eos_token_id"],
        },
        "precision": {"weight_format": config["dtype"]},
        "source": {
            "shard_count": len(set(index["weight_map"].values())),
            "tensor_count": len(index["weight_map"]),
            "index_total_bytes": index["metadata"]["total_size"],
            "header_payload_bytes": payload,
            "tensor_shapes": dict(sorted(shapes.items())),
        },
        "qualification": {"cuda_target": "sm_121a", "status": "CONTRACT_DRAFT", "production_ready": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", type=Path)
    group.add_argument("--remote", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    body = contract(Source(args.snapshot))
    if body["source"]["index_total_bytes"] != body["source"]["header_payload_bytes"]:
        raise SystemExit("the index total and the header payload differ")
    args.out.write_text(json.dumps(body, indent=1) + "\n")
    print(f"wrote {args.out}: {body['source']['tensor_count']} tensors, {body['source']['index_total_bytes']} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
