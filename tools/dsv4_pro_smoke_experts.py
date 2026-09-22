#!/usr/bin/env python3
"""DSV4 Pro smoke-expert manifest generator (lane 5, shared multidev).

Produces model-families/dsv4/smoke_experts.json for
tools/devcycle/lane_budget_calc.py plus an exact per-rank byte plan.

What is exact and what is measured later:

* Byte plan: exact. Re-derived from model_contracts/dsv4_pro.json through
  the same pro record builder + TP/PP planner the rank packs were sharded
  with (tools/dsv4_pro_stagepack.py patching tools/dsv4_tp16_stagepack.py),
  so per-rank spine and expert bytes match the deployed packs by
  construction. Anchor: planned bytes + header/directory must equal the
  rank pack's own header file_bytes (--check-pack).
* Hash-routed layers (0..2): exact. Routing is the token-id -> 6-expert
  table ffn.gate.tid2eid replicated in every rank pack (--pack), so the
  preload set for the recorded smoke prompt prefix is computed from the
  deployed bytes, not estimated.
* Learned-gate layers (3..60): NOT invented. No CPU routing oracle exists
  for Pro (no CPU reference engine), so v1 emits coverage scenarios from
  the exact per-rank byte plan and the manifest's preload list covers the
  hash layers plus the shared-expert spine. The measured touch-set growth
  curve (route dump from the first shared E2E run) regenerates this
  manifest wholesale; entries are never hand-edited.
* Route-tape regen (M4 wiring): the shared weightd records every
  (layer, expert) an arena's leases touch into <PACK>.wset next to the
  pack (atomic rewrite, deduped; runtime/spark_weightd.c). `manifest
  --route-tape <PACK>.wset` merges that measured tape with the exact
  prompt computation: tape keys in hash layers beyond the prompt set are
  measured decode routing (kept, never invented), tape keys in learned
  layers are the measured learned-route entries. Tapes and packs are
  per-rank artifacts: regenerate per node against its own pack.
* Non-speculative only: the replicated DSpark draft block (mtp.*) is
  excluded from the device working set (charter: non-spec decode first).

Usage:
  python3 tools/dsv4_pro_smoke_experts.py plan            # byte plan + scenarios
  python3 tools/dsv4_pro_smoke_experts.py manifest \
      --pack /home/sparkN/sparkdata/dsv4_pro.tp4pp4/packs/...spstage \
      --batch tools/devcycle/batches/o24_batch.json --prefix 16 \
      --output model-families/dsv4/smoke_experts.json
  python3 tools/dsv4_pro_smoke_experts.py manifest ... --route-tape \
      /home/sparkN/sparkdata/dsv4_pro.tp4pp4/packs/...spstage.wset \
      --route-tape-sha256 <sha> --route-run "<queue attempt id>"  # M4 regen
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

MIB = 1024 * 1024
LANE_DEVICE_MIB = 6400       # lane 5 per-node device budget (lane table)
LANE_TOTAL_MIB = 9792
CALC_MARGIN = 1.08           # lane_budget_calc margin


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pro = _load("dsv4_pro_stagepack_smoke", TOOLS / "dsv4_pro_stagepack.py")
flash = pro.flash
tp16 = _load("dsv4_tp16_stagepack_smoke", TOOLS / "dsv4_tp16_stagepack.py")

tp16.apply_model_geometry("pro")

CONTRACT = json.loads((ROOT / "model_contracts" / "dsv4_pro.json").read_text())

TP_DEGREE = 4
PP_STAGES = 4
WORLD_RANKS = TP_DEGREE * PP_STAGES
EXPERT_KINDS = (tp16.KIND_EXPERTS_W1, tp16.KIND_EXPERTS_W2, tp16.KIND_EXPERTS_W3)
DRAFT_LAYER_FIRST = tp16.MTP_LAYER_FIRST
DRAFT_MARKERS = tuple(DRAFT_LAYER_FIRST + i for i in range(3))
HASH_LAYERS = tuple(range(CONTRACT["moe"]["hash_routed_layer_count"]))
LEARNED_LAYERS = tuple(range(CONTRACT["moe"]["hash_routed_layer_count"],
                             CONTRACT["model"]["layer_count"]))
EXPERTS = CONTRACT["moe"]["routed_expert_count"]
EXPERTS_PER_TOKEN = CONTRACT["moe"]["experts_per_token"]
INDEX_HEADS = CONTRACT["attention"]["index_head_count"]
INDEX_DIM = CONTRACT["attention"]["index_head_dimension"]
HEAD_DIM = CONTRACT["model"]["head_dimension"]
ROPE_DIM = CONTRACT["model"]["qk_rope_head_dimension"]
SLIDING = CONTRACT["attention"]["sliding_window_tokens"]
COMPRESSION_RATIOS = CONTRACT["attention"]["compression_ratios"]

# Declared planning bounds (replaced by measured numbers at the first
# shared E2E run; see PR text). KV floor for one B*=1 smoke sequence at
# KV_FLOOR_TOKENS: per token per backbone layer (k+v+rope) bf16 plus the
# DSA indexer KV on ratio-4 layers, all treated as uncompressed - generous
# by construction because the compression ratios only shrink the real cache.
KV_FLOOR_TOKENS = 512
WORKSPACE_MIB = 512

HEADER = struct.Struct("<16I2Q")
ENTRY = struct.Struct("<6I2Q")


def is_draft(layer: int, kind: int) -> bool:
    """The replicated DSpark block: draft-layer markers plus the global
    draft head records (markov/confidence/hc/norms/main_proj)."""
    if layer in DRAFT_MARKERS:
        return True
    return layer == tp16.GLOBAL_LAYER and kind in tp16.KIND_MTP_SET


def rank_plan(rank: int) -> dict:
    """Exact per-rank byte plan from the contract records alone."""
    tp16.TP_DEGREE = TP_DEGREE
    records = pro.pro_build_records(CONTRACT, 0, pro.PRO_LAYERS)
    plan = {
        "rank": rank,
        "pp_stage": rank // TP_DEGREE,
        "tp_rank": rank % TP_DEGREE,
        "spine_bytes": 0,
        "expert_pool_bytes": 0,       # all 384 experts of its backbone layers
        "draft_bytes": 0,             # replicated DSpark block (non-spec: excluded)
        "spine_kinds": {},
        "backbone_layers": [],
        "expert_bytes_per_kind": 0,
        "expert_bytes_per_expert": 0,
    }
    first, count = tp16.layer_slice(PP_STAGES, plan["pp_stage"])
    plan["backbone_layers"] = list(range(first, first + count))
    for record in records:
        entry = (record.kind, record.layer, record.weight_format,
                 record.rows, record.columns, 0, 0, 0)
        try:
            planned, _, _, scale_total = tp16.plan_entry(
                entry, plan["tp_rank"], PP_STAGES, plan["pp_stage"])
        except tp16.PackFailure as error:
            if str(error) == "filtered":
                continue
            raise
        rows, columns = planned[3], planned[4]
        payload = tp16.payload_bytes(record.weight_format, rows, columns)
        total = payload + scale_total
        if is_draft(record.layer, record.kind):
            plan["draft_bytes"] += total
            continue
        if record.kind in EXPERT_KINDS:
            plan["expert_pool_bytes"] += total
            per_kind = total // EXPERTS
            if per_kind * EXPERTS != total:
                raise RuntimeError("expert record bytes are not expert-uniform")
            if plan["expert_bytes_per_kind"] not in (0, per_kind):
                raise RuntimeError("per-expert bytes differ across records")
            plan["expert_bytes_per_kind"] = per_kind
            # A routed expert is W1+W2+W3: three sharded records per layer.
            plan["expert_bytes_per_expert"] = per_kind * len(EXPERT_KINDS)
            continue
        plan["spine_bytes"] += total
        plan["spine_kinds"][record.kind] = (
            plan["spine_kinds"].get(record.kind, 0) + total)
    return plan


def kv_floor_bytes(plan: dict) -> int:
    """Upper-bound KV for one B*=1 smoke sequence on one rank.

    Per token per backbone layer: compressed k + v (head_dim each, bf16)
    plus the rope dimension, all charged at the full context floor with no
    compression-ratio credit; ratio-4 (indexer) layers additionally pay
    index_heads x index_dim bf16 per token.
    """
    kv_codec_bytes = 2  # bf16 cache codec (deployed default)
    per_token = (2 * HEAD_DIM + ROPE_DIM) * kv_codec_bytes
    indexer_per_token = INDEX_HEADS * INDEX_DIM * kv_codec_bytes
    total = 0
    for layer in plan["backbone_layers"]:
        total += KV_FLOOR_TOKENS * per_token
        if COMPRESSION_RATIOS[layer] == 4:
            total += KV_FLOOR_TOKENS * indexer_per_token
    return total


def read_tid2eid(pack_path: Path) -> Dict[int, List[bytes]]:
    """layer -> per-token-id 6-expert rows, from a deployed rank pack."""
    tables = {}
    with pack_path.open("rb") as pack:
        header = list(HEADER.unpack(pack.read(HEADER.size)))
        if header[0] != 0x34565344:
            raise RuntimeError(f"{pack_path}: not a DSV4 stage pack")
        if header[11] != pro.PRO_LAYERS or header[14] != EXPERTS:
            raise RuntimeError(f"{pack_path}: unexpected geometry in header")
        pack.seek(header[16])
        directory = [ENTRY.unpack(pack.read(ENTRY.size))
                     for _ in range(header[8])]
        for kind, layer, weight, rows, columns, _r, payload, _s in directory:
            if kind != flash.KIND_GATE_TID2EID or layer not in HASH_LAYERS:
                continue
            if rows != pro.PRO_VOCAB or columns != EXPERTS_PER_TOKEN or \
                    weight != tp16.WEIGHT_U32:
                raise RuntimeError(f"tid2eid record geometry mismatch: {layer}")
            pack.seek(payload)
            raw = pack.read(rows * columns * 4)
            if len(raw) != rows * columns * 4:
                raise RuntimeError("short tid2eid payload")
            tables[layer] = raw
    missing = [layer for layer in HASH_LAYERS if layer not in tables]
    if missing:
        raise RuntimeError(f"pack missing tid2eid tables for layers {missing}")
    return tables


def hash_layer_sets(tables: Dict[int, bytes],
                    token_ids: Sequence[int]) -> Dict[int, List[int]]:
    """Deduplicated per-hash-layer expert ids for the prompt token ids."""
    width = EXPERTS_PER_TOKEN
    sets: Dict[int, set] = {layer: set() for layer in HASH_LAYERS}
    for token in token_ids:
        if not 0 <= token < pro.PRO_VOCAB:
            raise RuntimeError(f"token id {token} outside vocabulary")
        for layer, raw in tables.items():
            base = (token * width) * 4
            ids = struct.unpack_from(f"<{width}I", raw, base)
            sets[layer].update(ids)
    for layer, ids in sets.items():
        for expert in ids:
            if not 0 <= expert < EXPERTS:
                raise RuntimeError(
                    f"layer {layer}: tid2eid expert {expert} out of range")
    return {layer: sorted(ids) for layer, ids in sets.items()}


def load_route_tape(path: Path, expect_sha256: str | None = None) -> List[Tuple[int, int]]:
    """Daemon-recorded <PACK>.wset route tape: deduplicated, sorted
    (layer, expert) pairs written by the shared weightd's working-set
    recorder (atomic mkstemp+rename rewrite; runtime/spark_weightd.c)."""
    raw = path.read_bytes()
    if not raw or len(raw) % 8:
        raise RuntimeError(f"{path}: tape must be nonempty (layer,expert) "
                           "u32 pairs")
    digest = hashlib.sha256(raw).hexdigest()
    if expect_sha256 is not None and digest != expect_sha256:
        raise RuntimeError(f"{path}: tape sha256 {digest} does not pin the "
                           f"cited {expect_sha256}")
    keys = sorted({struct.unpack_from("<II", raw, offset)
                   for offset in range(0, len(raw), 8)})
    for layer, expert in keys:
        if not 0 <= layer < CONTRACT["model"]["layer_count"] or \
                not 0 <= expert < EXPERTS:
            raise RuntimeError(f"{path}: tape key {layer}/{expert} outside "
                               "contract geometry")
    return keys


def load_pack_expert_keys(pack_path: Path) -> set:
    """The (layer, expert) keys the pack's .experts sidecar carries
    (16-byte magic/version/count/zero header + 48-byte records, native
    endianness; mirrored from SparkWeightdManifestLoad)."""
    path = Path(str(pack_path) + ".experts")
    data = path.read_bytes()
    if len(data) < 16:
        raise RuntimeError(f"{path}: shorter than the manifest header")
    magic, version, count, zero = struct.unpack_from("=IIII", data, 0)
    if zero != 0 or count == 0 or len(data) != 16 + 48 * count:
        raise RuntimeError(f"{path}: malformed expert manifest (magic="
                           f"0x{magic:08x} version={version} count={count} "
                           f"size={len(data)})")
    return {struct.unpack_from("=II", data, 16 + 48 * index)
            for index in range(count)}


def check_pack(plan: dict, pack_path: Path) -> dict:
    """Anchor the byte plan against a deployed rank pack's own header."""
    with pack_path.open("rb") as pack:
        header = list(HEADER.unpack(pack.read(HEADER.size)))
        tensors = header[8]
        planned = plan["spine_bytes"] + plan["expert_pool_bytes"] + \
            plan["draft_bytes"]
        directory = HEADER.size + ENTRY.size * tensors
        return {
            "pack": str(pack_path),
            "pack_file_bytes": header[17],
            "plan_plus_directory_bytes": planned + directory,
            "tensor_count": tensors,
            "delta_bytes": header[17] - (planned + directory),
        }


def build_manifest(pack_provenance: dict, prompt_provenance: dict,
                   sets: Dict[int, List[int]], per_expert_full: int,
                   worst_rank: dict, kv_floor: int,
                   learned_sets: Dict[int, List[int]] | None = None,
                   route: dict | None = None) -> dict:
    experts = []
    for layer in HASH_LAYERS:
        for expert in sets[layer]:
            experts.append({"layer": layer, "expert": expert,
                            "codec": "mxfp4_e2m1", "bytes": per_expert_full,
                            "source": "route-tape" if route is not None and
                            (layer, expert) not in route["_computed"]
                            else "prompt-hash"})
    for layer in sorted(learned_sets or {}):
        for expert in learned_sets[layer]:
            experts.append({"layer": layer, "expert": expert,
                            "codec": "mxfp4_e2m1", "bytes": per_expert_full,
                            "source": "route-tape"})
    if route is not None:
        route.pop("_computed", None)
        learned_route: dict | str = {
            "layers": [LEARNED_LAYERS[0], LEARNED_LAYERS[-1]],
            "routing": "learned gate (noaux_tc, top-6)",
            "entries": "measured",
            "measurement": route,
            "note": "regenerated wholesale by this tool from the daemon "
                    "route tape; never hand-edited",
        }
    else:
        learned_route = {
            "layers": [LEARNED_LAYERS[0], LEARNED_LAYERS[-1]],
            "routing": "learned gate (noaux_tc, top-6)",
            "entries": "pending measured route dump from the first shared "
                       "E2E run; regenerated wholesale by this tool, never "
                       "hand-edited",
        }
    manifest = {
        "schema_version": 1,
        "family": "dsv4",
        "model_id": CONTRACT["model_id"],
        "prompt_set": prompt_provenance["prompt_set"],
        "topology": "TP4xPP4",
        "nodes": WORLD_RANKS,
        # DSV4 experts are tensor-parallel inside every expert (row-split
        # W1/W3, column-split W2): every rank holds 1/TP of EACH selected
        # expert, and PP bounds the per-rank layer count. "tp" is the
        # calculator's envelope model; exact_per_rank carries stage truth.
        "expert_shard": "tp",
        "spine_bytes": worst_rank["spine_bytes"] * WORLD_RANKS,
        "kv_floor_bytes": kv_floor,
        "workspace_bytes": WORKSPACE_MIB * MIB,
        "experts": experts,
        "learned_route_layers": learned_route,
        "excluded": {
            "dspark_draft_block": "replicated on every rank pack; excluded "
                                  "from the non-speculative device working set",
        },
        "exact_per_rank": {
            "note": "worst rank (stage 0: 16 backbone layers + embedding)",
            "rank": worst_rank["rank"],
            "spine_bytes": worst_rank["spine_bytes"],
            "expert_bytes_per_expert": worst_rank["expert_bytes_per_expert"],
            "full_model_expert_bytes_per_expert": per_expert_full,
        },
        "provenance": {
            "pack": pack_provenance,
            "prompt": prompt_provenance,
            "contract": "model_contracts/dsv4_pro.json",
            "generator": "tools/dsv4_pro_smoke_experts.py",
        },
    }
    return manifest


def scenario_table(plans: Sequence[dict]) -> List[dict]:
    """Learned-layer coverage scenarios for the lane budget, worst rank.

    The worst rank is stage 0 (16 backbone layers + the full replicated
    embedding). Hash-routed layers (0-2) are charged at their measured
    tid2eid coverage when a manifest exists (hash coverage H below), never
    at the full 384.
    """
    stage0 = next(p for p in plans if p["pp_stage"] == 0)
    per_expert = stage0["expert_bytes_per_expert"]
    hash_count = len([l for l in stage0["backbone_layers"]
                      if l in HASH_LAYERS])
    learned_count = len(stage0["backbone_layers"]) - hash_count
    kv = kv_floor_bytes(stage0)
    fixed = stage0["spine_bytes"] / MIB + kv / MIB + WORKSPACE_MIB
    hash_measured = getattr(scenario_table, "hash_coverage", None)

    def device_mib(hash_covered: int, learned_covered: int) -> float:
        experts = (hash_count * hash_covered + learned_count * learned_covered) \
            * per_expert / MIB
        return (fixed + experts) * CALC_MARGIN

    rows = []
    if hash_measured is None:
        # No manifest yet: report the direct decision number instead.
        affordable = (LANE_DEVICE_MIB / CALC_MARGIN - fixed) * MIB / per_expert
        rows.append({
            "decision_number": "max_expert_layer_units_on_worst_rank",
            "lane_device_mib": LANE_DEVICE_MIB,
            "fixed_mib_spine_kv_workspace": round(fixed, 1),
            "affordable_expert_layer_units": round(affordable, 1),
            "note": "units = distinct (layer, expert) pairs at "
                    f"{per_expert} B each; e.g. uniform H hash + L learned "
                    f"coverage needs 3*H + 13*L units on stage 0",
        })
        return rows
    for covered in (6, 12, 18, 24, 32, 48, 64, 96, 192, 330, EXPERTS):
        device = device_mib(hash_measured, covered)
        rows.append({
            "hash_coverage_measured": hash_measured,
            "learned_layer_coverage": covered,
            "expert_pool_mib": round(
                (hash_count * hash_measured + learned_count * covered)
                * per_expert / MIB, 1),
            "device_mib_with_margin": round(device, 1),
            "fits_lane_6400": device <= LANE_DEVICE_MIB,
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="exact per-rank byte plan + scenarios")

    curve_parser = sub.add_parser(
        "curve", help="tid2eid touch-set growth curve vs prompt prefix")
    curve_parser.add_argument("--pack", type=Path, required=True)
    curve_parser.add_argument("--batch", type=Path, required=True)
    curve_parser.add_argument("--request", type=int, default=0)
    curve_parser.add_argument("--prefixes", default="1,2,4,8,12,16,24,32,64,128")

    manifest_parser = sub.add_parser("manifest", help="emit smoke_experts.json")
    manifest_parser.add_argument("--pack", type=Path, required=True,
                                 help="any deployed rank pack (tid2eid source)")
    manifest_parser.add_argument("--batch", type=Path, required=True,
                                 help="recorded batch JSON with prompt_token_ids")
    manifest_parser.add_argument("--request", type=int, default=0)
    manifest_parser.add_argument("--prefix", type=int, default=16,
                                 help="smoke prompt prefix length in tokens")
    manifest_parser.add_argument("--prompt-set", default="smoke-standard-v1")
    manifest_parser.add_argument("--pack-sha256", default=None,
                                 help="known sha256 of --pack (skip re-hashing "
                                      "a ~95 GiB pack; cite the placement "
                                      "receipt that measured it)")
    manifest_parser.add_argument("--output", type=Path)
    manifest_parser.add_argument("--wset", type=Path,
                                 help="also emit the binary (layer, expert) "
                                      "u32-pair file for "
                                      "tools/weightd_warm.c --wset")
    manifest_parser.add_argument("--route-tape", type=Path, default=None,
                                 help="daemon-recorded <PACK>.wset tape from "
                                      "a shared E2E run (M4 regen: merges "
                                      "measured decode routing and learned-"
                                      "layer entries; per-rank artifact - "
                                      "regenerate against the same node's "
                                      "pack)")
    manifest_parser.add_argument("--route-tape-sha256", default=None,
                                 help="pin the tape's sha256 (receipt-"
                                      "citable; refuses a drifted tape)")
    manifest_parser.add_argument("--route-run", default="",
                                 help="provenance label for the run that "
                                      "produced the tape (e.g. the queue "
                                      "attempt id)")

    check_parser = sub.add_parser("check-pack", help="anchor plan vs a pack")
    check_parser.add_argument("--rank", type=int, required=True)
    check_parser.add_argument("--pack", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "check-pack":
        print(json.dumps(check_pack(rank_plan(args.rank), args.pack), indent=1))
        return 0

    if args.command == "curve":
        batch = json.loads(args.batch.read_text())
        tokens = batch["requests"][args.request]["prompt_token_ids"]
        tables = read_tid2eid(args.pack)
        rows = []
        seen = {layer: set() for layer in HASH_LAYERS}
        per_rank = rank_plan(0)["expert_bytes_per_expert"]
        for prefix in [int(p) for p in args.prefixes.split(",")]:
            if prefix > len(tokens):
                break
            for token in tokens[:prefix]:
                for layer, raw in tables.items():
                    ids = struct.unpack_from(f"<{EXPERTS_PER_TOKEN}I", raw,
                                             (token * EXPERTS_PER_TOKEN) * 4)
                    seen[layer].update(ids)
            units = sum(len(v) for v in seen.values())
            rows.append({
                "prefix_tokens": prefix,
                "distinct_token_ids": len(set(tokens[:prefix])),
                "per_layer": {str(l): len(v) for l, v in sorted(seen.items())},
                "hash_units_total": units,
                "stage0_preload_mib": round(units * per_rank / MIB, 1),
            })
        print(json.dumps({"curve": rows,
                          "bytes_per_unit_per_rank": per_rank}, indent=1))
        return 0

    plans = [rank_plan(rank) for rank in range(WORLD_RANKS)]
    if args.command == "plan":
        summary = []
        for plan in plans:
            summary.append({
                "rank": plan["rank"], "pp_stage": plan["pp_stage"],
                "backbone_layers": len(plan["backbone_layers"]),
                "spine_mib": round(plan["spine_bytes"] / MIB, 1),
                "expert_pool_mib": round(plan["expert_pool_bytes"] / MIB, 1),
                "draft_mib_excluded": round(plan["draft_bytes"] / MIB, 1),
                "expert_mib_per_expert":
                    round(plan["expert_bytes_per_expert"] / MIB, 3),
            })
        full_per_expert = plans[0]["expert_bytes_per_expert"] * TP_DEGREE
        print(json.dumps({
            "ranks": summary,
            "full_model_bytes_per_expert": full_per_expert,
            "kv_floor_mib_declared": round(kv_floor_bytes(plans[0]) / MIB, 1),
            "workspace_mib_declared": WORKSPACE_MIB,
            "budget_scenarios_stage0_worst_rank": scenario_table(plans),
        }, indent=1))
        return 0

    # manifest
    batch = json.loads(args.batch.read_text())
    request = batch["requests"][args.request]
    tokens = request["prompt_token_ids"][:args.prefix]
    distinct = sorted(set(tokens))
    tables = read_tid2eid(args.pack)
    sets = hash_layer_sets(tables, tokens)
    per_expert_full = plans[0]["expert_bytes_per_expert"] * TP_DEGREE
    worst = max(plans, key=lambda p: p["spine_bytes"])
    route = None
    learned_sets: Dict[int, List[int]] = {}
    if args.route_tape is not None:
        pack_keys = load_pack_expert_keys(args.pack)
        tape = load_route_tape(args.route_tape, args.route_tape_sha256)
        unknown = [key for key in tape if key not in pack_keys]
        if unknown:
            raise RuntimeError(f"{args.route_tape}: keys absent from this "
                               f"pack's expert manifest: {unknown[:4]} (the "
                               "tape is a per-rank artifact - regenerate "
                               "against the same node's pack)")
        computed = {(layer, expert) for layer in sets
                    for expert in sets[layer]}
        measured_hash = {key for key in tape if key[0] in HASH_LAYERS}
        learned = {key for key in tape if key[0] not in HASH_LAYERS}
        for layer, expert in measured_hash - computed:
            sets[layer].append(expert)
        for layer in sets:
            sets[layer] = sorted(set(sets[layer]))
        learned_sets = {}
        for layer, expert in learned:
            learned_sets.setdefault(layer, set()).add(expert)
        learned_sets = {layer: sorted(keys)
                        for layer, keys in sorted(learned_sets.items())}
        route = {
            "source": "weightd working-set recorder (<PACK>.wset)",
            "tape": str(args.route_tape),
            "tape_sha256": hashlib.sha256(
                args.route_tape.read_bytes()).hexdigest(),
            "run": args.route_run,
            "hash_layer_keys_measured": len(measured_hash),
            "hash_layer_keys_beyond_prompt": len(measured_hash - computed),
            "learned_layer_keys_measured": len(learned),
            "learned_layers_touched": sorted({key[0] for key in learned}),
            "learned_per_layer": {str(layer): len(keys)
                                  for layer, keys in learned_sets.items()},
            "_computed": computed,
        }
    scenario_table.hash_coverage = max(len(v) for v in sets.values())
    manifest = build_manifest(
        pack_provenance={
            "path": str(args.pack),
            "sha256": args.pack_sha256 or
                      "run with --pack-sha256 (cite the placement receipt)",
            "anchor": "check-pack delta_bytes=0 against this pack",
        },
        prompt_provenance={
            "prompt_set": args.prompt_set,
            "batch": str(args.batch),
            "request_id": request.get("request_id"),
            "prefix_tokens": args.prefix,
            "distinct_token_ids": len(distinct),
        },
        sets=sets, per_expert_full=per_expert_full, worst_rank=worst,
        kv_floor=kv_floor_bytes(worst), learned_sets=learned_sets or None,
        route=route)
    manifest["budget_scenarios_stage0_worst_rank"] = scenario_table(plans)
    text = json.dumps(manifest, indent=1, sort_keys=False) + "\n"
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output} ({len(manifest['experts'])} experts, "
              f"{sum(len(v) for v in sets.values())} hash-layer entries; "
              f"hash coverage per layer "
              f"{[len(sets[l]) for l in HASH_LAYERS]})")
    else:
        print(text)
    if args.wset:
        pairs = sorted([(layer, expert) for layer in sorted(sets)
                        for expert in sets[layer]] +
                       [(layer, expert) for layer in learned_sets
                        for expert in learned_sets[layer]])
        keys = b"".join(struct.pack("<II", layer, expert)
                        for layer, expert in pairs)
        if not keys:
            raise RuntimeError("refusing to write an empty .wset")
        args.wset.write_bytes(keys)
        print(f"wrote {args.wset} ({len(keys)//8} (layer,expert) pairs"
              + (f", {sum(len(v) for v in learned_sets.values())} "
                 "route-tape learned" if learned_sets else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
