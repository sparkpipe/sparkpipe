#!/usr/bin/env python3
"""GLM-5.2-FP8 source verification: freeze / verify / census.

Stdlib-only (no torch) so it runs on any spark node or the controller.
Policy note: the cold RAID6 (mounted read-only at
/mnt/cold-raid6/22tb0/models/hf/zai-org/GLM-5.2-FP8 on spark0) sustains
~406 MB/s per direct-read stream, so the freeze FULLY hashes every model
file (no sampling) with 4 parallel workers. If a future host cannot
afford a full pass, --stride N hashes shards on a stride and fully hashes
every file smaller than --small-bytes; the chosen policy is recorded in
the contract so a verify can never silently widen or narrow it.

Usage:
  glm52_verify_source.py freeze --source DIR [--workers 4] \
      [--contract model_contracts/glm52_authoritative.json]
  glm52_verify_source.py verify --source DIR --contract FILE [--workers 4]
  glm52_verify_source.py census --source DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

CONTRACT_RELATIVE = Path("model_contracts/glm52_authoritative.json")
READ_CHUNK = 16 * 1024 * 1024
BASELINE_SMALL_BYTES = 256 * 1024 * 1024  # files below this are always fully hashed

MODEL_KEYS_FROM_CONFIG = [
    # contract key -> config.json key path (None = derived in code)
    "hidden_dimension", "layer_count", "head_count", "latent_dimension",
    "rope_dimension", "rope_theta", "rope_interleave", "rms_norm_epsilon",
    "first_routed_layer", "moe_expert_count", "moe_top_k",
    "moe_intermediate_dimension", "moe_routed_scaling_factor",
    "dense_intermediate_dimension", "output_vocab_count",
    "qk_nope_head_dimension", "value_head_dimension", "query_a_dimension",
    "dsa_selected_token_count", "dsa_index_head_count",
    "dsa_index_head_dimension", "maximum_context_tokens",
    "fp8_scale_block",
]


def find_source_metadata(source: Path) -> Dict[str, Path]:
    files = {p.name: p for p in source.iterdir() if p.is_file()}
    for required in ("config.json",):
        if required not in files:
            raise SystemExit(f"source missing {required}: {source}")
    index = None
    for name in ("model.safetensors.index.json",):
        if name in files:
            index = files[name]
    return {"config": files["config.json"], "index": index, "all": files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(READ_CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def shard_names(all_files: Dict[str, Path]) -> List[str]:
    return sorted(n for n in all_files if n.endswith(".safetensors"))


def freeze(source: Path, workers: int, stride: int, small_bytes: int) -> Dict[str, Any]:
    meta = find_source_metadata(source)
    shards = shard_names(meta["all"])
    if not shards:
        raise SystemExit("no .safetensors shards found")
    small = sorted(
        n for n in meta["all"] if not n.endswith(".safetensors")
        and meta["all"][n].stat().st_size <= small_bytes
    )
    selected_shards = [n for i, n in enumerate(shards) if i % stride == 0]
    largest = max(shards, key=lambda n: meta["all"][n].stat().st_size)
    if largest not in selected_shards:
        selected_shards.append(largest)
        selected_shards.sort()
    targets = small + selected_shards
    model_bytes = sum(meta["all"][n].stat().st_size for n in meta["all"])
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = dict(zip(targets, pool.map(sha256_file, (meta["all"][t] for t in targets))))
    elapsed = time.time() - started
    contract: Dict[str, Any] = {
        "schema_version": 1,
        "lane": "glm52",
        "model_id": "zai-org/GLM-5.2",
        "quantization": "fp8",
        "architecture": None,  # filled from config.json below
        "source_path": str(source),
        "freeze_policy": {
            "shard_count": len(shards),
            "hashed_shard_count": len(selected_shards),
            "stride": stride,
            "small_files_fully_hashed": len(small),
            "small_bytes_threshold": small_bytes,
            "fully_hashed_fraction_of_model_bytes": round(
                sum(meta["all"][n].stat().st_size for n in targets) / model_bytes, 6),
            "elapsed_seconds": round(elapsed, 1),
        },
        "model_bytes": model_bytes,
        "file_count": len(meta["all"]),
        "digests": {name: digests[name] for name in sorted(digests)},
    }
    config = json.loads(meta["config"].read_text())
    contract["architecture"] = config.get("architectures", [None])[0]
    contract["config_sha256"] = sha256_file(meta["config"])
    contract["derived_geometry"] = geometry_from_config(config)
    if meta["index"] is not None:
        contract["index_sha256"] = sha256_file(meta["index"])
    return contract


def geometry_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    text = config.get("text_config", config)
    keys = {
        "hidden_size": "hidden_dimension",
        "num_hidden_layers": "layer_count",
        "num_attention_heads": "head_count",
        "kv_lora_rank": "latent_dimension",
        "qk_nope_head_dim": "qk_nope_head_dimension",
        "qk_rope_head_dim": "rope_dimension",
        "v_head_dim": "value_head_dimension",
        "q_lora_rank": "query_a_dimension",
        "n_routed_experts": "moe_expert_count",
        "num_experts_per_tok": "moe_top_k",
        "moe_intermediate_size": "moe_intermediate_dimension",
        "intermediate_size": "dense_intermediate_dimension",
        "vocab_size": "output_vocab_count",
        "first_k_dense_replace": "first_routed_layer",
        "max_position_embeddings": "maximum_context_tokens",
        "index_topk": "dsa_selected_token_count",
        "index_n_heads": "dsa_index_head_count",
        "index_head_dim": "dsa_index_head_dimension",
        "rms_norm_eps": "rms_norm_epsilon",
        "rope_theta": "rope_theta",
    }
    geometry = {}
    for config_key, contract_key in keys.items():
        if config_key in text:
            geometry[contract_key] = text[config_key]
    if "scoring_func" in text:
        geometry["router_scoring_func"] = text["scoring_func"]
    if "norm_topk_prob" in text:
        geometry["norm_topk_prob"] = text["norm_topk_prob"]
    return geometry


def verify(source: Path, contract_path: Path, workers: int) -> int:
    contract = json.loads(contract_path.read_text())
    policy = contract.get("freeze_policy", {})
    stride = int(policy.get("stride", 1))
    small_bytes = int(policy.get("small_bytes_threshold", BASELINE_SMALL_BYTES))
    fresh = freeze(source, workers, stride, small_bytes)
    # compare digests recorded vs re-derived
    recorded = contract["digests"]
    derived = fresh["digests"]
    mismatches = [n for n in recorded if derived.get(n) != recorded[n]]
    missing = [n for n in recorded if n not in derived]
    if mismatches or missing:
        for name in missing:
            print(f"FAIL {name}: recorded but not re-derived under the frozen policy")
        for name in mismatches:
            print(f"FAIL {name}: sha256 {derived[name]} != recorded {recorded[name]}")
        return 1
    if fresh["config_sha256"] != contract.get("config_sha256"):
        print("FAIL config.json sha256 drift")
        return 1
    if fresh.get("index_sha256") != contract.get("index_sha256"):
        print("FAIL model.safetensors.index.json sha256 drift")
        return 1
    if fresh["file_count"] != contract.get("file_count") or fresh["model_bytes"] != contract.get("model_bytes"):
        print(f"FAIL census drift: files {fresh['file_count']} vs {contract.get('file_count')},"
              f" bytes {fresh['model_bytes']} vs {contract.get('model_bytes')}")
        return 1
    if fresh["derived_geometry"] != contract.get("derived_geometry"):
        print("FAIL geometry drift vs frozen contract")
        return 1
    hashed = policy.get("hashed_shard_count", "?")
    total = policy.get("shard_count", "?")
    fraction = policy.get("fully_hashed_fraction_of_model_bytes", 1.0)
    print(f"PASS {len(recorded)} pinned files verified against {source}"
          f" (stride {stride}, {hashed}/{total} shards fully hashed,"
          f" {fraction:.4%} of model bytes); geometry re-derived from live"
          f" config.json matches the contract")
    return 0


def census(source: Path) -> int:
    if shutil_which_torch():
        pass  # census is metadata-only; torch not required
    meta = find_source_metadata(source)
    if meta["index"] is None:
        print("no index json; shard census unavailable")
        return 1
    index = json.loads(meta["index"].read_text())
    entries = index.get("weight_map", {})
    classes: Dict[str, int] = {}
    for tensor in entries:
        stem = tensor
        for digit in "0123456789":
            stem = stem.replace(digit, "#")
        classes[stem] = classes.get(stem, 0) + 1
    print(f"index tensors: {len(entries)}; pattern classes: {len(classes)}")
    for name in sorted(classes):
        print(f"{classes[name]:5d}  {name}")
    return 0


def shutil_which_torch() -> bool:
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--source", required=True)
    p_freeze.add_argument("--workers", type=int, default=4)
    p_freeze.add_argument("--stride", type=int, default=1,
                          help="hash shards on this stride (default 1 = full)")
    p_freeze.add_argument("--small-bytes", type=int, default=BASELINE_SMALL_BYTES)
    p_freeze.add_argument("--contract", default=str(CONTRACT_RELATIVE))
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--source", required=True)
    p_verify.add_argument("--contract", required=True)
    p_verify.add_argument("--workers", type=int, default=4)
    p_census = sub.add_parser("census")
    p_census.add_argument("--source", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    if args.mode == "freeze":
        contract = freeze(source, args.workers, args.stride, args.small_bytes)
        path = Path(args.contract)
        if not path.is_absolute():
            root = Path(__file__).resolve().parents[1]
            path = root / path
        path.write_text(json.dumps(contract, indent=1, sort_keys=True) + "\n")
        policy = contract["freeze_policy"]
        print(f"froze {len(contract['digests'])} digests to {path}"
              f" (stride {policy['stride']}, {policy['fully_hashed_fraction_of_model_bytes']:.4%}"
              f" of {contract['model_bytes']} model bytes in {policy['elapsed_seconds']}s)")
        return 0
    if args.mode == "verify":
        return verify(source, Path(args.contract), args.workers)
    return census(source)


if __name__ == "__main__":
    raise SystemExit(main())
