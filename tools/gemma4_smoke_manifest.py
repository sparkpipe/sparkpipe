#!/usr/bin/env python3
"""Generate the gemma4-31b smoke working-set manifest (lane 6).

Emits model-families/gemma4/smoke_experts.json in the lane manifest
convention (tools/devcycle/lane_budget_calc.py, PR #1083): the manifest is
machine-generated from landed facts and never hand-edited.

Inputs (all fail closed):
  - the 16 per-rank pack placement receipts of the landed TP16 set
    (~/sparkdata/gemma4_31b.bf16.tp16/packs/*.receipt.json on each spark):
    ranks 0..15 present, placement proofs passed, payload uniform
  - model_contracts/gemma4_31b_authoritative.json for geometry

gemma4-31b is DENSE (receipts: expert_bytes 0, experts_manifest null): the
routed-expert list is provably empty and the smoke working set is the
full-resolution spine itself (the quality law: no requantization, the
whole bf16 rank pack is the resident set). spine_bytes is therefore the
fleet-resident total (per-rank payload x 16; the calculator's tp sharding
divides it back to the per-node pack).

kv_floor_bytes and workspace_bytes are PER NODE (the calculator applies
them per node directly):
  - kv_floor: smoke sequences x smoke context tokens x per-token per-rank
    KV bytes, derived from the contract geometry and the receipts' KV-head
    boundary facts (sliding_kv_heads_per_rank, full_kv_replication)
  - workspace: the recorded declared bound (--workspace-mib) until a
    measured value replaces it (M3 measurement)

Usage:
  python3 tools/gemma4_smoke_manifest.py --receipts /path/to/receipts \
      [--output model-families/gemma4/smoke_experts.json] \
      [--prompt-set smoke-standard-v1] [--smoke-sequences 16] \
      [--smoke-context-tokens 1024] [--workspace-mib 512]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "model_contracts/gemma4_31b_authoritative.json"
DEFAULT_OUTPUT = REPOSITORY / "model-families/gemma4/smoke_experts.json"
RANKS = 16
BF16_BYTES = 2
SCHEMA_VERSION = 1


def fail(message: str) -> None:
    raise SystemExit(f"gemma4_smoke_manifest: FAIL: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", required=True,
                        help="directory holding the 16 per-rank pack receipts")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prompt-set", default="smoke-standard-v1")
    parser.add_argument("--smoke-sequences", type=int, default=16)
    parser.add_argument("--smoke-context-tokens", type=int, default=1024)
    parser.add_argument("--workspace-mib", type=int, default=512,
                        help="per-node workspace declared bound in MiB "
                             "(replaced by the measured value at M3)")
    arguments = parser.parse_args()

    receipts_dir = Path(arguments.receipts)
    contract = json.loads(CONTRACT_PATH.read_text())["model"]
    receipts = {}
    for path in sorted(receipts_dir.glob("*.json")):
        receipt = json.loads(path.read_text())
        rank = receipt.get("tp_rank")
        if rank is None or rank in receipts:
            fail(f"bad or duplicate tp_rank in {path}")
        receipts[rank] = receipt
    if sorted(receipts) != list(range(RANKS)):
        fail(f"receipts must cover ranks 0..{RANKS - 1}, got {sorted(receipts)}")

    payload_bytes = None
    spine_total = 0
    for rank, receipt in receipts.items():
        if receipt.get("topology") != "tp16pp1" or receipt.get("tp_degree") != RANKS:
            fail(f"rank {rank} receipt is not tp16pp1/tp16")
        proof = receipt.get("placement_proof", {}).get("passed")
        if proof is not True:
            fail(f"rank {rank} placement proof not passed")
        if receipt.get("expert_bytes", 0) != 0 or receipt.get("experts_manifest") is not None:
            fail(f"rank {rank} carries expert bytes/manifest; the dense arm must have none")
        rank_payload = receipt.get("payload_bytes")
        if not isinstance(rank_payload, int) or rank_payload <= 0:
            fail(f"rank {rank} payload_bytes missing")
        if payload_bytes is None:
            payload_bytes = rank_payload
        elif rank_payload != payload_bytes:
            fail(f"non-uniform payload: rank {rank} {rank_payload} != {payload_bytes}")
        spine_total += rank_payload

    boundaries = receipts[0].get("boundary_ranks", [])
    sliding_heads_per_rank = next(
        (b["sliding_kv_heads_per_rank"] for b in boundaries if b.get("rank") == 0), None)
    full_replication = next(
        (b["full_kv_replication"] for b in boundaries if b.get("rank") == 0), None)
    if sliding_heads_per_rank is None or full_replication is None:
        fail("receipt boundary_ranks lacks rank-0 KV-head facts")
    full_heads_per_rank = contract["full_kv_head_count"] // full_replication
    if (contract["full_kv_head_count"] % full_replication) != 0:
        fail("full_kv_head_count not divisible by the receipt replication")

    layer_count = contract["layer_count"]
    full_layers = sum(
        1 for layer in range(layer_count)
        if layer % contract["full_layer_period"] == contract["full_layer_phase"])
    sliding_layers = layer_count - full_layers
    sliding_kv_bytes_per_token = (sliding_layers * sliding_heads_per_rank *
                                  contract["sliding_head_dimension"] * 2 * BF16_BYTES)
    full_kv_bytes_per_token = (full_layers * full_heads_per_rank *
                               contract["full_head_dimension"] * 2 * BF16_BYTES)
    kv_bytes_per_token = sliding_kv_bytes_per_token + full_kv_bytes_per_token
    kv_floor_bytes = (arguments.smoke_sequences * arguments.smoke_context_tokens *
                      kv_bytes_per_token)
    workspace_bytes = arguments.workspace_mib * 1024 * 1024

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "family": "gemma4",
        "model": "google/gemma-4-31B-it",
        "prompt_set": arguments.prompt_set,
        "topology": "TP16",
        "nodes": RANKS,
        "expert_shard": "tp",
        "spine_bytes": spine_total,
        "kv_floor_bytes": kv_floor_bytes,
        "workspace_bytes": workspace_bytes,
        "notes": (
            "dense family: routed-expert set provably empty (all 16 receipts "
            "expert_bytes=0); the smoke working set is the full-resolution bf16 "
            "spine (per-rank payload 3880953464 x 16, tp-sharded). kv_floor "
            f"derived: {sliding_layers} sliding layers x {sliding_heads_per_rank} "
            f"head/rank x 256d + {full_layers} full layers x {full_heads_per_rank} "
            f"head/rank x 512d, K+V, bf16 = {kv_bytes_per_token} B/token/rank x "
            f"{arguments.smoke_sequences} seqs x {arguments.smoke_context_tokens} "
            "tokens. workspace is the declared bound until the M3 measurement."
        ),
        "experts": [],
    }
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{output}: dense smoke manifest ({RANKS} ranks, spine "
          f"{spine_total} B total, kv_floor {kv_floor_bytes} B/node, "
          f"workspace {workspace_bytes} B/node, experts 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
