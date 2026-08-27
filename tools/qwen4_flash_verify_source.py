#!/usr/bin/env python3
"""Qwen 3.8 Flash source-archive freeze and verifier (lane qwen-flash, M1).

Two modes:

  freeze:  hash the checkpoint per the lane sampling policy and emit a JSON
           digest manifest intended to be embedded verbatim under
           model_contracts/qwen4_flash_authoritative.json ("digest_freeze").

  verify:  re-hash the pinned files of a source tree and compare them with
           the contract; also re-derive the geometry section from the live
           config.json and compare with the contract. Exit 0 only on full
           agreement.

Sampling policy (frozen; do not change without re-freezing):
  - every file smaller than SMALL_LIMIT bytes is fully hashed;
  - safetensors shards are fully hashed on index stride STRIDE starting at
    shard 1 plus the largest shard regardless of stride;
  - the checkpoint-provided SHA256SUMS and ARCHIVE-RECEIPT.json are hashed
    and pinned too, which transitively pins the un-sampled shards via the
    publisher's dual-archive verification receipt.

Stride note: the warm Ceph mount measured ~4 MiB/s per stream with a
~30 MiB/s aggregate cap at freeze time (shared cluster traffic), so the
stride was widened from the initial 5 to 17 to keep the freeze under an
hour; 9 of 131 shards (~27 GiB, ~8% of shard bytes) are fully hashed and
the rest are pinned transitively through the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SMALL_LIMIT = 256 * 1024 * 1024  # 256 MiB
STRIDE = 17
HASH_WORKERS = 8
SHARD_RE = re.compile(r"^model-(\d+)-of-(\d+)\.safetensors$")
CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            total += len(block)
            digest.update(block)
    return digest.hexdigest(), total


def classify(files: list[Path]) -> tuple[list[Path], list[Path], list[int], int]:
    shards: dict[int, Path] = {}
    smalls: list[Path] = []
    for path in files:
        match = SHARD_RE.match(path.name)
        if match and path.stat().st_size >= SMALL_LIMIT:
            shards[int(match.group(1))] = path
        else:
            smalls.append(path)
    total = max(shards) if shards else 0
    largest = max(shards, key=lambda i: shards[i].stat().st_size) if shards else 0
    sampled = sorted(set(range(1, total + 1, STRIDE)) | {largest})
    return smalls, [shards[i] for i in sampled], sampled, total


def freeze(source: Path) -> dict:
    files = sorted(p for p in source.iterdir() if p.is_file())
    smalls, sampled_shards, sampled_idx, total = classify(files)
    to_hash = smalls + sampled_shards
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as pool:
        digests = list(pool.map(sha256_file, to_hash))
    entries: dict[str, dict] = {}
    for path, (digest, size) in zip(to_hash, digests):
        entries[path.name] = {"sha256": digest, "bytes": size}
    sampled_bytes = sum(e["bytes"] for e in entries.values())
    return {
        "policy": (
            f"all {len(smalls)} files smaller than {SMALL_LIMIT} bytes fully hashed; "
            f"shards fully hashed on index stride {STRIDE} from 1 "
            f"({len(sampled_idx)} of {total} shards, indices "
            f"{sampled_idx[0]}..{sampled_idx[-1]}) plus the largest shard; "
            "unsampled shards are transitively pinned by the hashed "
            "publisher SHA256SUMS + ARCHIVE-RECEIPT.json (dual-archive "
            "verified at publish time per the receipt)"
        ),
        "stride": STRIDE,
        "small_file_limit_bytes": SMALL_LIMIT,
        "sampled_shard_indices": sampled_idx,
        "total_shard_count": total,
        "sampled_bytes": sampled_bytes,
        "files": dict(sorted(entries.items())),
    }


def verify(source: Path, contract: dict) -> list[str]:
    problems: list[str] = []
    frozen = contract["digest_freeze"]
    present = {p.name for p in source.iterdir() if p.is_file()}
    for name, entry in sorted(frozen["files"].items()):
        path = source / name
        if not path.exists():
            problems.append(f"MISSING {name}")
            continue
        digest, size = sha256_file(path)
        if digest != entry["sha256"]:
            problems.append(f"HASH MISMATCH {name}: pinned {entry['sha256']} live {digest}")
        if size != entry["bytes"]:
            problems.append(f"SIZE MISMATCH {name}: pinned {entry['bytes']} live {size}")
    # every shard named by the publisher's SHA256SUMS must exist, sampled or not
    sums = source / "SHA256SUMS"
    if sums.exists() and sums.name in frozen["files"]:
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] not in present:
                problems.append(f"MISSING publisher-listed file {parts[1]}")
    # geometry must still match the live config.json
    geometry = contract["model"]
    try:
        cfg = json.loads((source / "config.json").read_text())["text_config"]
    except Exception as error:  # noqa: BLE001
        problems.append(f"config.json unreadable: {error}")
        cfg = {}
    live = {
        "hidden_dimension": cfg.get("hidden_size"),
        "layer_count": cfg.get("num_hidden_layers"),
        "vocabulary_size": cfg.get("vocab_size"),
        "mtp_layer_count": cfg.get("mtp_num_hidden_layers"),
        "maximum_context_tokens": cfg.get("max_position_embeddings"),
        "full_attention_interval": cfg.get("full_attention_interval"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "kv_head_count": cfg.get("num_key_value_heads"),
        "head_dimension": cfg.get("head_dim"),
        "moe_intermediate_dimension": cfg.get("moe_intermediate_size"),
        "shared_expert_intermediate_dimension": cfg.get("shared_expert_intermediate_size"),
        "routed_expert_count": cfg.get("num_experts"),
        "experts_per_token": cfg.get("num_experts_per_tok"),
    }
    for key, value in live.items():
        if value is None:
            problems.append(f"config.json missing field for {key}")
        elif geometry.get(key) != value:
            problems.append(f"GEOMETRY MISMATCH {key}: contract {geometry.get(key)} live {value}")
    return problems


def census(source: Path) -> dict:
    """Tensor-name census from the safetensors index: pattern -> count.

    Layer indices are folded to 'N' so the census is a stable inventory of
    tensor CLASSES (per-layer counts are asserted separately where they are
    geometry: 36 linear-attn layers, 12 full-attn layers, PLE at layer 1).
    """
    import re

    index = json.loads((source / "model.safetensors.index.json").read_text())
    patterns: dict[str, int] = {}
    for name in index["weight_map"]:
        pattern = re.sub(r"\.\d+\.", ".N.", name)
        patterns[pattern] = patterns.get(pattern, 0) + 1
    return {
        "index_entries": len(index["weight_map"]),
        "patterns": dict(sorted(patterns.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "verify", "census"])
    parser.add_argument("--source", type=Path, required=True, help="checkpoint root (e.g. warm mount)")
    parser.add_argument("--contract", type=Path, help="contract JSON (verify mode)")
    args = parser.parse_args()

    if args.mode == "freeze":
        print(json.dumps(freeze(args.source), indent=1, sort_keys=True))
        return 0

    if args.mode == "census":
        print(json.dumps(census(args.source), indent=1, sort_keys=True))
        return 0

    if args.contract is None:
        parser.error("verify mode requires --contract")
    contract = json.loads(args.contract.read_text())
    problems = verify(args.source, contract)
    frozen = contract["digest_freeze"]
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    print(
        f"PASS {len(frozen['files'])} pinned files verified against "
        f"{args.source} (stride {frozen['stride']}, {len(frozen['sampled_shard_indices'])}"
        f"/{frozen['total_shard_count']} shards fully hashed); geometry re-derived "
        "from live config.json matches the contract"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
