#!/usr/bin/env python3
"""Generate model-families/glm5_next/smoke_experts.json (lane budget convention).

The smoke-expert manifest is the FIRST INSTANCE of the lane-budget convention
(tools/devcycle/lane_budget_calc.py, PR #1083): the deduplicated set of routed
experts the recorded smoke prompt set actually touches, with full-model bytes,
plus the full-resolution spine and the qualified KV/workspace floors. It is
machine-generated from fleet artifacts and never hand-edited:

  inputs
    --wset PATH           the qualified campaign working-set file
                          (.wset: N little-endian (layer u32, expert u32) key
                          pairs; 1..512 pairs, deduplicated). Its sha256 must
                          match the qualified campaign receipt's
                          working_set.input_sha256 when provided.
    --manifest GLOB       the per-rank expert range manifests (<pack>.experts,
                          wire format from include/sparkpipe/spark_weightd_manifest.h:
                          16-byte header magic/version/range_count/zero then
                          48-byte records). One per TP rank - full-model
                          per-expert bytes are the SUM across ranks.
    --pack-bytes N        exact size of every rank pack (must be uniform).
    --pack-sha256 SHA     the packs/*.sha256 digest sidecar value.
    --kv-floor-bytes N    default: the qualified smoke KV plan cap (2 GiB,
                          context 512 / 128 pages / B1, PR #1082 campaign).
    --workspace-bytes N   default: measured resident device allocation
                          (3,434 MiB) minus the KV floor - includes graph
                          workspace and CUDA context overhead (measured, not
                          analytically split; provenance recorded in output).

  output
    model-families/glm5_next/smoke_experts.json per the PR #1083 schema.

Quality law: experts are quantized fp8-source-native, the spine is full
resolution; this tool only reports bytes, it never re-encodes anything.
"""

from __future__ import annotations

import argparse
import glob as globlib
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "model-families/glm5_next/smoke_experts.json"

MANIFEST_MAGIC = 0x58504557
MANIFEST_VERSION = 2
WSET_PAIR_BYTES = 8
WSET_PAIRS_MAX = 512

# Qualified campaign floors (PR #1082 shared-resident receipts; B1, context
# capacity 512, 128 logical/physical KV pages, 2 GiB backing cap; measured
# resident device allocation 3,434 MiB including workspace and CUDA context).
QUALIFIED_KV_FLOOR_BYTES = 2 * 1024 * 1024 * 1024
QUALIFIED_RESIDENT_DEVICE_BYTES = 3434 * 1024 * 1024

FAMILY = "glm5_next"
MODEL = "glm53flash.fp8.tp16"
PROMPT_SET = "glm53flash-shared-smoke-v1"
TOPOLOGY = "TP16"
NODES = 16
CODEC = "fp8_e4m3"
EXPERT_SHARD = "tp"
SPINE_SHARD = "tp"

# Required provenance fields (lane-budget convention, PR #1083 review): every
# manifest must carry a non-empty kv-floor provenance, workspace provenance and
# qualification note; the writer refuses to emit an unprovenanced manifest.
QUALIFICATION_NOTE = (
    "smoke working-set budget for cold-launch preload / partial-pool debugging; "
    "the GPU-qualified serving configuration remains "
    "SPARK_GLM5_NEXT_GRAPH_PATH=1 + SPARK_GLM5_NEXT_PIN_EXPERTS=1 with the full "
    "pool (whole-pack residency per node)"
)
KV_FLOOR_PROVENANCE = "qualified PR #1082 smoke KV plan cap (2 GiB backing, context 512, B1)"
WORKSPACE_PROVENANCE = (
    "measured resident device allocation 3,434 MiB (PR #1082 receipts) minus the "
    "KV floor; includes graph workspace and CUDA context"
)


def fail(message: str) -> None:
    raise SystemExit(f"glm5_next_smoke_experts: FAIL: {message}")


def load_wset(path: Path, expect_sha256: str | None) -> list[tuple[int, int]]:
    raw = path.read_bytes()
    if not 0 < len(raw) <= WSET_PAIRS_MAX * WSET_PAIR_BYTES or len(raw) % WSET_PAIR_BYTES:
        fail(f"wset {path} is not 1..{WSET_PAIRS_MAX} complete key pairs ({len(raw)} bytes)")
    if expect_sha256 and hashlib.sha256(raw).hexdigest() != expect_sha256:
        fail(f"wset sha256 differs from the qualified campaign receipt ({expect_sha256})")
    keys = [struct.unpack_from("<II", raw, offset) for offset in range(0, len(raw), WSET_PAIR_BYTES)]
    if len(set(keys)) != len(keys):
        fail("wset contains duplicate expert keys - the smoke set must be deduplicated")
    return keys


def load_manifest(path: Path) -> dict[tuple[int, int], int]:
    raw = path.read_bytes()
    if len(raw) < 16:
        fail(f"manifest {path} truncated")
    magic, version, range_count, zero = struct.unpack_from("<IIII", raw, 0)
    if magic != MANIFEST_MAGIC or version != MANIFEST_VERSION or zero != 0:
        fail(f"manifest {path} header mismatch (magic={magic:#x} version={version} zero={zero})")
    if len(raw) != 16 + 48 * range_count:
        fail(f"manifest {path} size {len(raw)} disagrees with range_count {range_count}")
    per_expert: dict[tuple[int, int], int] = {}
    for index in range(range_count):
        offset = 16 + 48 * index
        layer, expert, _kind, _pad, _range_offset, range_bytes = struct.unpack_from("<IIIIQQ", raw, offset)
        per_expert[(layer, expert)] = per_expert.get((layer, expert), 0) + range_bytes
    if not per_expert:
        fail(f"manifest {path} carries no expert ranges")
    return per_expert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wset", type=Path, required=True)
    parser.add_argument("--manifest", action="append", required=True,
                        help="per-rank .experts manifest path or glob (repeat or glob across all TP ranks)")
    parser.add_argument("--pack-bytes", type=int, required=True)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--wset-sha256", default=None,
                        help="qualified campaign working_set.input_sha256 to pin")
    parser.add_argument("--kv-floor-bytes", type=int, default=QUALIFIED_KV_FLOOR_BYTES)
    parser.add_argument("--workspace-bytes", type=int,
                        default=QUALIFIED_RESIDENT_DEVICE_BYTES - QUALIFIED_KV_FLOOR_BYTES)
    parser.add_argument("--spine-shard", choices=("tp", "replicated"), default=SPINE_SHARD,
                        help="spine residency across nodes (default: %(default)s); tp divides "
                             "spine_bytes by nodes, replicated repeats it on every node")
    parser.add_argument("--kv-floor-provenance", default=KV_FLOOR_PROVENANCE)
    parser.add_argument("--workspace-provenance", default=WORKSPACE_PROVENANCE)
    parser.add_argument("--qualification-note", default=QUALIFICATION_NOTE)
    parser.add_argument("--prompt-set", default=PROMPT_SET,
                        help="recorded prompt-set id (default: %(default)s)")
    parser.add_argument("--batch-sha256", default="f6aa43dfc185367ef54a5158dcfce904793c4c85e29b9a8795e45422ef8e5d9a",
                        help="sha256 of the prompt batch json for --prompt-set")
    parser.add_argument("--reference-sha256", default="705da922b9e59070831bca80a9c362095955f28f25ec4dca4a63d3b2c7aa8743",
                        help="sha256 of the pinned token reference for --prompt-set")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    if arguments.pack_bytes <= 0:
        fail("pack bytes must be positive")
    if arguments.kv_floor_bytes <= 0 or arguments.workspace_bytes <= 0:
        fail("kv floor and workspace must be positive")
    for label, value in (("kv-floor provenance", arguments.kv_floor_provenance),
                         ("workspace provenance", arguments.workspace_provenance),
                         ("qualification note", arguments.qualification_note)):
        if not value.strip():
            fail(f"{label} is REQUIRED - the manifest must not carry unprovenanced floors")

    manifest_paths: list[Path] = []
    for pattern in arguments.manifest:
        matches = sorted(globlib.glob(pattern))
        if not matches:
            fail(f"manifest glob matched nothing: {pattern}")
        manifest_paths.extend(Path(match) for match in matches)
    if len(manifest_paths) != NODES:
        fail(f"expected {NODES} per-rank manifests for {TOPOLOGY}, got {len(manifest_paths)}")

    keys = load_wset(arguments.wset, arguments.wset_sha256)
    full_experts: dict[tuple[int, int], int] = {}
    spine_full_bytes = 0
    for path in manifest_paths:
        per_rank = load_manifest(path)
        rank_expert_bytes = sum(per_rank.values())
        if rank_expert_bytes >= arguments.pack_bytes:
            fail(f"manifest {path} expert bytes exceed the pack")
        spine_full_bytes += arguments.pack_bytes - rank_expert_bytes
        for key, value in per_rank.items():
            full_experts[key] = full_experts.get(key, 0) + value

    missing = [key for key in keys if key not in full_experts]
    if missing:
        fail(f"{len(missing)} wset keys absent from the pack manifests (first: {missing[0]})")

    experts = [
        {"layer": layer, "expert": expert, "codec": CODEC, "bytes": full_experts[(layer, expert)]}
        for layer, expert in sorted(keys)
    ]
    manifest = {
        "schema_version": 1,
        "family": FAMILY,
        "model": MODEL,
        "prompt_set": arguments.prompt_set,
        "topology": TOPOLOGY,
        "nodes": NODES,
        "expert_shard": EXPERT_SHARD,
        "spine_shard": arguments.spine_shard,
        "spine_bytes": spine_full_bytes,
        "kv_floor_bytes": arguments.kv_floor_bytes,
        "workspace_bytes": arguments.workspace_bytes,
        "experts": experts,
        "provenance": {
            "generated_by": "tools/glm5_next_smoke_experts.py",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wset_path": str(arguments.wset),
            "wset_sha256": hashlib.sha256(arguments.wset.read_bytes()).hexdigest(),
            "prompt_batch_sha256": arguments.batch_sha256,
            "token_reference_sha256": arguments.reference_sha256,
            "pack_bytes": arguments.pack_bytes,
            "pack_sha256": arguments.pack_sha256,
            "manifest_count": len(manifest_paths),
            "expert_codec": "fp8-source-native",
            "quality_law": "quantized routed experts, full-resolution spine, no requantization",
            "kv_floor_provenance": arguments.kv_floor_provenance,
            "workspace_provenance": arguments.workspace_provenance,
            "qualification_note": arguments.qualification_note,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=1, sort_keys=False) + "\n")
    total = sum(entry["bytes"] for entry in experts)
    print(f"wrote {arguments.output}")
    print(f"  experts      : {len(experts)} keys, full-model {total} bytes ({total / 1048576:.1f} MiB), "
          f"per-node({NODES}) {total / NODES / 1048576:.1f} MiB")
    print(f"  spine        : full-model {spine_full_bytes} ({spine_full_bytes / 1048576:.1f} MiB), "
          f"per-node {spine_full_bytes / NODES / 1048576:.1f} MiB")
    print(f"  floors       : kv {arguments.kv_floor_bytes} ({arguments.kv_floor_bytes / 1048576:.0f} MiB), "
          f"workspace {arguments.workspace_bytes} ({arguments.workspace_bytes / 1048576:.0f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
