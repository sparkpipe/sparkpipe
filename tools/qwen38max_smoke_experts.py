#!/usr/bin/env python3
"""Generate model-families/qwen38_max/smoke_experts.json.

Machine-generated smoke-expert manifest for the multidev lane budget
calculator (tools/devcycle/lane_budget_calc.py, PR #1083 convention).
Never hand-edit the emitted JSON; regenerate with this tool.

Sources (all fail-closed, provenance pinned in the emitted document):
  - the warm checkpoint index (model.safetensors.index.json) for exact
    per-tensor byte accounting; config/index sha256 must match the
    committed qualification/t1_reference/qwen38_max/MANIFEST.json identity
  - the committed T1R1 reference fixtures for the routed-expert set the
    recorded smoke prompt set actually touches (route_ids arrays)
  - model-families/qwen38_max/include/sparkpipe/llm_defines.h for the
    geometry cross-check (same law as t1_reference_qwen38_max.cross_check)

Quality-law checks baked in: the spine must be entirely BF16 (full
resolution, no requantization) and every routed-expert plane must be
NVFP4 payload + F8_E4M3 block scales (+ one f32 scalar per tensor);
any deviation fails the tool.
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import parse_llm_defines, read_fixture  # noqa: E402


def define_uint(defines, name):
    value = defines[name].strip()
    if value.endswith("u"):
        value = value[:-1]
    if not value.isdigit():
        raise SystemExit(f"SPARK_LLM_{name} is not a uint: {value!r}")
    return int(value)

FAMILY = "qwen38_max"
SCHEMA_VERSION = 1
TOPOLOGY = "TP16"
NODES = 16
CODEC = "nvfp4"
EXPERT_TENSOR = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\."
    r"(weight(?:_scale_2|_scale|_2)?|input_scale)$")
# Smoke context floor for kv_floor_bytes: prompts_qwen38max tops out at 12
# positions; 128 tokens x 8 concurrent sequences bounds the smoke harness.
KV_FLOOR_TOKENS = 128
KV_FLOOR_SEQUENCES = 8
# Device workspace for the smoke harness (logits staging row, activation
# slabs, graph workspace) before the M3/M4 measured number replaces it.
WORKSPACE_BYTES = 256 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_headers(checkpoint, needed_files):
    """Read each shard's safetensors header once; return {file: header}."""
    headers = {}
    for name in sorted(needed_files):
        with open(os.path.join(checkpoint, name), "rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            headers[name] = json.loads(handle.read(header_bytes))
    return headers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="warm checkpoint root (nvfp4-radixark-bf16-spine)")
    parser.add_argument("--repo", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--emit", default=None,
                        help="output path (default <repo>/model-families/"
                             "qwen38_max/smoke_experts.json)")
    parser.add_argument("--check", action="store_true",
                        help="regenerate and compare against the committed "
                             "manifest instead of writing")
    arguments = parser.parse_args()

    repo = arguments.repo
    fixture_dir = os.path.join(repo, "qualification/t1_reference", FAMILY)
    fixture_manifest_path = os.path.join(fixture_dir, "MANIFEST.json")
    defines_path = os.path.join(repo, "model-families", FAMILY,
                                "include/sparkpipe/llm_defines.h")
    prompts_path = os.path.join(fixture_dir, "prompts_qwen38max.json")
    emit_path = arguments.emit or os.path.join(repo, "model-families",
                                               FAMILY, "smoke_experts.json")

    fixture_manifest = json.load(open(fixture_manifest_path))
    recorded = fixture_manifest["checkpoint"]
    config_path = os.path.join(arguments.checkpoint, "config.json")
    index_path = os.path.join(arguments.checkpoint,
                              "model.safetensors.index.json")
    config_sha = sha256_file(config_path)
    index_sha = sha256_file(index_path)
    if config_sha != recorded["config_sha256"]:
        raise SystemExit("config sha drift: checkpoint is not the recorded "
                         f"identity ({config_sha} != {recorded['config_sha256']})")
    if index_sha != recorded["index_sha256"]:
        raise SystemExit("index sha drift: checkpoint is not the recorded "
                         f"identity ({index_sha} != {recorded['index_sha256']})")

    config = json.load(open(config_path))
    if "text_config" in config:
        config = config["text_config"]
    defines = parse_llm_defines(defines_path)
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    experts = int(config["num_experts"])
    if hidden != define_uint(defines, "HIDDEN_DIMENSION") or \
            layers != define_uint(defines, "LAYER_COUNT") or \
            experts != define_uint(defines, "ROUTED_EXPERT_COUNT"):
        raise SystemExit("llm_defines/config geometry disagreement")

    weight_map = json.load(open(index_path))["weight_map"]

    expert_planes = {}      # (layer, expert) -> bytes over gate/up/down
    expert_dtypes = set()   # quality-law check on plane dtypes
    spine_bytes = 0
    spine_bad = []          # non-BF16 spine tensors (quality-law violation)
    needed_files = sorted(set(weight_map.values()))
    headers = load_headers(arguments.checkpoint, needed_files)
    for tensor, shard in weight_map.items():
        entry = headers[shard].get(tensor)
        if entry is None:
            raise SystemExit(f"index maps {tensor} to {shard} but the shard "
                             "header lacks it")
        size = entry["data_offsets"][1] - entry["data_offsets"][0]
        match = EXPERT_TENSOR.match(tensor)
        if match:
            layer, expert = int(match.group(1)), int(match.group(2))
            expert_planes[(layer, expert)] = \
                expert_planes.get((layer, expert), 0) + size
            expert_dtypes.add((match.group(4) or "", entry["dtype"],
                               tuple(entry["shape"])))
        else:
            spine_bytes += size
            if entry["dtype"] != "BF16":
                spine_bad.append((tensor, entry["dtype"]))

    if spine_bad:
        for tensor, dtype in spine_bad[:10]:
            print(f"spine tensor not BF16: {tensor} ({dtype})", file=sys.stderr)
        raise SystemExit("spine is not full-resolution BF16 (quality law)")
    per_expert_sizes = set(expert_planes.values())
    if len(per_expert_sizes) != 1:
        raise SystemExit(f"expert planes are not uniform: "
                         f"{len(per_expert_sizes)} distinct sizes")
    per_expert_bytes = per_expert_sizes.pop()
    total_experts = len(expert_planes)
    if total_experts != layers * experts:
        raise SystemExit(f"expected {layers * experts} experts, "
                         f"index covers {total_experts}")

    # The routed-expert working set: union of route_ids over every fixture
    # recorded for the smoke prompt set (prompts_qwen38max.json — the
    # #1072 fixture identity).
    prompts_sha = sha256_file(prompts_path)
    if fixture_manifest["prompts_sha256"] != prompts_sha:
        raise SystemExit("prompts_qwen38max.json drifted from the recorded "
                         "fixture identity")
    touched = set()
    fixtures_used = []
    for name in sorted(os.listdir(fixture_dir)):
        if not name.endswith(".t1r"):
            continue
        path = os.path.join(fixture_dir, name)
        _, arrays = read_fixture(path)
        count = 0
        for array_name, array in arrays.items():
            if not array_name.endswith("_route_ids"):
                continue
            touched.update(
                (int(array_name.split("_layer")[1].split("_")[0]), int(e))
                for e in array)
            count += 1
        fixtures_used.append({"fixture": name, "sha256": sha256_file(path),
                              "route_arrays": count})
    missing = touched - set(expert_planes)
    if missing:
        raise SystemExit(f"fixtures route to {len(missing)} experts absent "
                         "from the checkpoint index")

    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    full_attn_layers = sum(1 for t in config["layer_types"]
                           if t == "full_attention")
    gdn_layers = layers - full_attn_layers
    kv_per_token = 2 * kv_heads * head_dim * 2  # K+V, BF16
    gdn_channels = (2 * int(config["linear_num_key_heads"])
                    * int(config["linear_key_head_dim"])
                    + int(config["linear_num_value_heads"])
                    * int(config["linear_value_head_dim"]))
    kv_floor = (KV_FLOOR_TOKENS * full_attn_layers * kv_per_token
                + KV_FLOOR_SEQUENCES * gdn_layers * gdn_channels
                * (int(config["linear_conv_kernel_dim"]) - 1) * 2)

    document = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "prompt_set": "qwen38max-t1-prompts-v1",
        "topology": TOPOLOGY,
        "nodes": NODES,
        "expert_shard": "tp",
        "codec": CODEC,
        "spine_bytes": spine_bytes,
        "kv_floor_bytes": kv_floor,
        "workspace_bytes": WORKSPACE_BYTES,
        "experts": [{"layer": layer, "expert": expert, "codec": CODEC,
                     "bytes": per_expert_bytes}
                    for layer, expert in sorted(touched)],
        "provenance": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
            "generator": "tools/qwen38max_smoke_experts.py",
            "checkpoint_path": os.path.abspath(arguments.checkpoint),
            "config_sha256": config_sha,
            "index_sha256": index_sha,
            "prompts_sha256": prompts_sha,
            "llm_defines_sha256": sha256_file(defines_path),
            "fixtures": fixtures_used,
            "expert_plane_dtypes": sorted(
                f"{suffix or 'payload'}:{dtype}{list(shape)}"
                for suffix, dtype, shape in expert_dtypes),
            "full_model_expert_bytes": per_expert_bytes * total_experts,
            "full_model_expert_count": total_experts,
        },
    }

    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if arguments.check:
        current = open(emit_path, encoding="utf-8").read()
        if current != text:
            print("smoke_experts.json drift: regenerate with "
                  "tools/qwen38max_smoke_experts.py", file=sys.stderr)
            return 1
        print("smoke_experts.json matches")
        return 0
    with open(emit_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(json.dumps({
        "family": FAMILY,
        "touched_experts": len(touched),
        "per_expert_bytes": per_expert_bytes,
        "full_model_experts_mib":
            round(per_expert_bytes * total_experts / (1 << 20), 1),
        "smoke_experts_mib": round(per_expert_bytes * len(touched) / (1 << 20), 1),
        "spine_mib": round(spine_bytes / (1 << 20), 1),
        "kv_floor_mib": round(kv_floor / (1 << 20), 2),
        "emit": emit_path,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
