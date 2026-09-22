#!/usr/bin/env python3
"""Emit the dsv41_flash canonical smoke-expert manifest (PR #1083
convention, manager ruling 2026-09-22): the deduplicated routed-expert
set the CANONICAL smoke prompt set actually touches, with bytes, plus
the lane budget inputs (spine/KV/workspace floors).

Canonical selection rule (smoke-canonical-v1): position 0 of every
prompt in the recorded prompt set - the first prefill row of every
smoke request. Rationale (the lane-4 ruling): the preload working set
must stay <= ~2 GiB per node amortized at TP4 (~450 pairs); position 0
already routes at all 40 layers, both recorded prompts stay
represented, and generation beyond the first row rides the qualified
lazy M7 attach path (strays are covered, not silently dropped).

Bytes come from the family geometry single source of truth
(tools/dsv41_flash_stagepack.py, receipt-cross-checked against the
placed mxfp4 packs: 3 projections x (5,898,240 payload + 368,640
scale) = 18,800,640 B per (layer, expert) - the m7 attach receipt's
36,097,228,800 B per TP8 rank / (40 layers x 48 experts) is exactly
this number).

Spine note: the kind-map tp=1 spine (9,531,704,768 B) matches the
placed r2 mxfp4 tp8 packs exactly (plan.json on the r2 build: tp8
spine_bytes 3,061,536,448 == kind-map tp8; pack 39,158,874,624 =
spine + experts 36,097,228,800). The older m7_attach_receipt spine
census (2,914,592,448 B/rank) is the r1 pack, before the o_a
rows-slice repack (+146,944,000 B/rank = o_a rows x FULL 4096 cols +
[32,128] scale x 40 layers, commit bbb8821c) - no unexplained delta
remains. The pack/expert bytes live in the operator's shared weightd
arena (tracked 29,184 MiB/node, additional to lane budgets, manager
ruling 2026-09-22); the lane DEVICE budget covers the resident side
only.

Usage:
  python3 tools/dsv41_flash_smoke_experts.py \
      --output model-families/dsv41_flash/smoke_experts.json
  python3 tools/dsv41_flash_smoke_experts.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsv41_flash_stagepack as sp  # noqa: E402  (family geometry source of truth)

FIXTURES_DIR = Path("qualification/t1_reference/dsv41")
PROMPTS_JSON = FIXTURES_DIR / "prompts.json"
MANIFEST_PATH = Path("model-families/dsv41_flash/smoke_experts.json")
TOPOLOGY = "TP4"
NODES = 4
KV_PER_TOKEN_BYTES = 890          # contract kv_cache.per_token_global_bytes
KV_FLOOR_POSITIONS = 128          # swa window bound (sliding_window 128)
WORKSPACE_BYTES = 8 * 1024 * 1024  # boundary frames + head-screen scratch + row buffers
PROMPT_SET = "dsv41-smoke-canonical-v1"


def fail(message: str):
    raise SystemExit(f"dsv41_flash_smoke_experts: FAIL: {message}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_fixture_routes(path: Path) -> dict:
    """T1R1 fixture (tools/t1_reference_common.py write_fixture): magic,
    u64 header length, JSON array metadata, then one u64-length-prefixed
    zlib blob per array in metadata order."""
    data = path.read_bytes()
    if data[:4] != b"T1R1":
        fail(f"{path}: not a T1R1 fixture")
    header_len = struct.unpack("<Q", data[4:12])[0]
    meta = json.loads(data[12:12 + header_len])
    offset = 12 + header_len
    routes = {}
    for entry in meta["arrays"]:
        compressed_len = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        raw = zlib.decompress(data[offset:offset + compressed_len])
        offset += compressed_len
        name = entry["name"]
        if not name.endswith("_route_ids"):
            continue
        if entry["dtype"] != "I32" or entry["bytes"] != len(raw):
            fail(f"{path}: bad route array {name}")
        routes[name] = set(struct.unpack("<%di" % (entry["bytes"] // 4), raw))
    return routes


def canonical_pairs(fixtures: dict) -> set:
    """(layer, expert) pairs touched by position 0 of every prompt."""
    pairs = set()
    for name, routes in fixtures.items():
        selected = [n for n in routes if re.match(r"pos0000_layer\d+_route_ids$", n)]
        if not selected:
            fail(f"{name}: no position-0 route arrays")
        layers = {int(re.search(r"layer(\d+)", n).group(1)) for n in selected}
        if layers != set(range(sp.LAYER_COUNT)):
            fail(f"{name}: position 0 does not route at every layer "
                 f"(missing {sorted(set(range(sp.LAYER_COUNT)) - layers)[:8]})")
        for array_name, ids in routes.items():
            if not array_name.startswith("pos0000_"):
                continue
            layer = int(re.search(r"layer(\d+)", array_name).group(1))
            pairs.update((layer, expert) for expert in ids)
    return pairs


def per_expert_bytes() -> int:
    """mxfp4 wire bytes per (layer, expert): the three projections (W1, W2,
    W3) payload plus scale planes, derived from the stagepack kind map at
    tp=1 - each kind's layout already multiplies in the full expert group
    count, so one expert's share is (payload + scale) / groups."""
    total = 0
    for kind in sp.EXPERT_KINDS:
        _, _, _, payload, scale = sp.entry_layout(kind, 1)
        _, _, groups = sp.entry_shape(kind, 1)
        if groups != sp.ROUTED_EXPERTS:
            fail(f"kind {kind}: groups {groups} != {sp.ROUTED_EXPERTS}")
        if (payload + scale) % groups:
            fail(f"kind {kind}: expert bytes do not divide evenly")
        total += (payload + scale) // groups
    return total


def spine_bytes_full_model() -> int:
    total = 0
    for kind, _layer, _spec in sp.entry_specs():
        if kind in sp.EXPERT_KINDS:
            continue
        _, _, _, payload, scale = sp.entry_layout(kind, 1)
        total += payload + scale
    return total


def spine_bytes_per_node(tp: int) -> int:
    total = 0
    for kind, _layer, _spec in sp.entry_specs():
        if kind in sp.EXPERT_KINDS:
            continue
        _, _, _, payload, scale = sp.entry_layout(kind, tp)
        total += payload + scale
    return total


def build_manifest() -> dict:
    prompts = json.loads(PROMPTS_JSON.read_text())
    fixtures = {}
    for spec in prompts["prompts"]:
        path = FIXTURES_DIR / f"{spec['name']}.t1r"
        if not path.is_file():
            fail(f"missing fixture {path}")
        fixtures[spec["name"]] = read_fixture_routes(path)
    pairs = canonical_pairs(fixtures)
    expert_bytes = per_expert_bytes()
    experts = [{"layer": layer, "expert": expert,
                "codec": "mxfp4_e2m1", "bytes": expert_bytes}
               for layer, expert in sorted(pairs)]

    per_node_counts = [0] * NODES
    for _layer, expert in pairs:
        per_node_counts[expert // (sp.ROUTED_EXPERTS // NODES)] += 1
    worst_node_bytes = max(per_node_counts) * expert_bytes

    return {
        "schema_version": 1,
        "family": "dsv41_flash",
        "prompt_set": PROMPT_SET,
        "prompt_set_definition": "position 0 (first prefill row) of every "
                                 "prompt in qualification/t1_reference/dsv41/prompts.json",
        "topology": TOPOLOGY,
        "nodes": NODES,
        "expert_shard": "tp",
        "spine_bytes": spine_bytes_full_model(),
        "kv_floor_bytes": KV_PER_TOKEN_BYTES * KV_FLOOR_POSITIONS,
        "workspace_bytes": WORKSPACE_BYTES,
        "experts": experts,
        "provenance": {
            "prompts_sha256": sha256_file(PROMPTS_JSON),
            "fixtures": {name: sha256_file(FIXTURES_DIR / f"{name}.t1r")
                         for name in sorted(fixtures)},
            "per_expert_bytes": expert_bytes,
            "per_expert_bytes_receipt": "examples/release/dsv41_flash_tp8/"
                                        "m7_attach_receipt.json expert_bytes "
                                        "36097228800 / (40 layers x 48 experts/rank)",
            "expert_shard_rule": "rank r owns experts [r*%d, (r+1)*%d)"
                                 % (sp.ROUTED_EXPERTS // NODES,
                                    sp.ROUTED_EXPERTS // NODES),
            "per_node_pair_counts": per_node_counts,
            "per_node_worst_expert_bytes": worst_node_bytes,
            "per_node_spine_bytes_tp4_kind_map": spine_bytes_per_node(NODES),
            "spine_note": "kind-map tp=1 spine; matches the placed r2 "
                          "mxfp4 tp8 packs exactly (r2 plan.json tp8 spine "
                          "3,061,536,448 B == kind-map tp8; the older "
                          "m7_attach_receipt 2,914,592,448 B/rank census is "
                          "the r1 pack before the o_a rows-slice repack "
                          "+146,944,000 B/rank, commit bbb8821c - no "
                          "unexplained delta remains). Pack/expert bytes "
                          "live in the shared weightd arena (tracked "
                          "29,184 MiB/node, additional to lane budgets, "
                          "manager ruling 2026-09-22); lane DEVICE covers "
                          "the resident side only",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(MANIFEST_PATH))
    parser.add_argument("--emit-wset", metavar="PATH",
                        help="also write the canonical set as a weightd "
                             ".wset file (native-endian u32 (layer, expert) "
                             "pairs, tools/weightd_warm.c format)")
    parser.add_argument("--wset-rank", type=int, metavar="R", default=None,
                        help="with --emit-wset: keep only the pairs owned by "
                             "tp rank R (contiguous shard: rank r owns "
                             "experts [r*384/4, (r+1)*384/4)); the warmer "
                             "validates keys against the rank pack's own "
                             ".experts manifest")
    parser.add_argument("--check", action="store_true",
                        help="regenerate and compare against the committed manifest")
    args = parser.parse_args()

    manifest = build_manifest()
    rendered = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    if args.check:
        committed = Path(args.output).read_text()
        if committed != rendered:
            fail(f"{args.output} differs from regeneration (regen with --output)")
        print("dsv41_flash smoke-expert manifest matches regeneration")
        return 0
    Path(args.output).write_text(rendered)
    if args.emit_wset:
        shard = sp.ROUTED_EXPERTS // NODES
        selected = [entry for entry in manifest["experts"]
                    if args.wset_rank is None
                    or entry["expert"] // shard == args.wset_rank]
        if args.wset_rank is not None and not selected:
            fail(f"--wset-rank {args.wset_rank}: no canonical pairs on that shard")
        blob = bytearray()
        for entry in selected:
            # The rank pack's .experts manifest keys experts LOCALLY
            # (0..shard-1: the manifest tool writes the per-rank group
            # index), so a rank-filtered wset remaps global -> local.
            expert = entry["expert"] if args.wset_rank is None \
                else entry["expert"] - shard * args.wset_rank
            blob += struct.pack("<II", entry["layer"], expert)
        Path(args.emit_wset).write_bytes(bytes(blob))
        print(f"{args.emit_wset}: {len(selected)} wset pairs "
              f"({len(blob)} bytes)"
              + (f" rank {args.wset_rank} (local expert ids)" if args.wset_rank is not None else ""))
    amortized = len(manifest["experts"]) * manifest["provenance"]["per_expert_bytes"] / NODES
    print(f"{args.output}: {len(manifest['experts'])} canonical pairs, "
          f"amortized {amortized / 2**20:.0f} MiB/node, "
          f"worst node {manifest['provenance']['per_node_worst_expert_bytes'] / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
