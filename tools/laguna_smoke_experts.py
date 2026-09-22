#!/usr/bin/env python3
"""Generate model-families/laguna/smoke_experts.json.

Machine-generated smoke-expert manifest for the multidev lane budget
calculator (tools/devcycle/lane_budget_calc.py, PR #1083 convention).
Never hand-edit the emitted JSON; regenerate with this tool.

The manifest IS the batch-preload working set for the < 5 s cold-launch
milestone - never an aspiration list. Laguna routes top-10 of 256
experts per layer with the expert set REPLICATED across every TP rank of
its stage (the packer shards the intermediate dimension, not the expert
identity), so one decode position touches 47 x 10 distinct (layer,
expert) pairs spread over both pipeline stages and no honest smoke
prompt set fits the <= 2 GiB/node preload target by trimming alone. The
sanctioned mechanism (the k3 M2 manager ruling of 2026-09-23, applied
here): experts[] = the repetition head of the touched census (pairs
routed in at least CUT_RULE positions) and the lazy-attach path covers
the strays; the FULL census, the cut rule and the coverage fraction are
pinned in provenance. The M3 receipt must measure the real lazy-stray
rate against that coverage claim.

Byte bases (both, fail-closed, provenance pinned):
  1. per-rank TP8-shard expert spans, straight from the deployed .lgsp
     pack directories (kinds 14/15 payload geometry - uniform bf16
     per-expert interleaves), cross-checked against the .experts
     sidecars when present;
  2. the spine bytes per rank pack (the complement of the expert spans
     inside the pack file, full-resolution bf16 - the quality law: no
     requantization).
The full-model expert bytes use laguna's own chunk-union factor - the
number of rank packs that hold a given (layer, expert) pair - computed
from the collected summaries, never assumed from another family.
Laguna's shape differs from both precedents: the factor is the TP
degree 8 (every expert of a stage's layers lives in all 8 packs of that
stage's group), and a node's preload cost is its OWN stage's pairs at
the per-rank span, not the full-model bytes over 16.

Inputs
  --ranks-json : per-rank pack summaries collected with --dump-rank on
                 every rank (16 records for TP8xPP2)
  --repo       : repository root (fixtures, defines, emit path)
  --dump-rank  : alternative mode - print ONE rank summary as JSON for
                 collection: python3 --dump-rank PACK <this-file

Modes: default emits; --check regenerates and compares byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import zlib
from collections import Counter
from pathlib import Path

FAMILY = "laguna"
SCHEMA_VERSION = 1
TOPOLOGY = "TP8xPP2"
NODES = 16
TP = 8
CODEC = "bf16"
CUT_RULE = 3                  # pairs routed in >= this many positions (the
                              # <= ~2 GiB/node smoke law: cut 2 measures
                              # 2,112.8 MiB on the worst stage - over; the
                              # full tradeoff curve is pinned in provenance
                              # and moves with M3 stray data, k3-style)
KV_FLOOR_TOKENS = 128         # bound above the fixture set (20 positions)
KV_FLOOR_SEQUENCES = 1        # shared-lane smoke keeps B1 (template law)
WORKSPACE_BYTES = 256 * 1024 * 1024   # placeholder; M3/M4 measured number
PACK_MAGIC = 0x334C4147       # "GAL3" little-endian
PACK_VERSION = 1
HEADER_BYTES = 264
ENTRY_BYTES = 64
K_EXPERT_GATE_UP, K_EXPERT_DOWN = 14, 15
PAYLOAD_BF16, CODEC_BF16 = 1, 1
T1R_MAGIC = b"T1R1"
EXPERTS_MAGIC = 0x58504557

UINT_DEFINES = {
    "layer_count": "SPARK_LLM_LAYER_COUNT",
    "first_routed_layer": "SPARK_LLM_FIRST_ROUTED_LAYER",
    "expert_count": "SPARK_LLM_MOE_EXPERT_COUNT",
    "moe_top_k": "SPARK_LLM_MOE_TOP_K",
    "kv_bits": "SPARK_LLM_KV_BITS",
    "kv_head_count": "SPARK_LLM_FULL_KV_HEAD_COUNT",
    "tp_degree": "SPARK_LLM_TENSOR_PARALLEL_DEGREE",
    "head_dimension": "SPARK_LLM_HEAD_DIMENSION",
}


def fail(message):
    raise SystemExit(f"laguna_smoke_experts: FAIL: {message}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_defines(path):
    """Numeric defines over the family literal set (k3_smoke_experts
    precedent: derived parenthesised expressions resolve over the file's
    own literals)."""
    text = open(path, encoding="utf-8").read()
    literals = {}
    for match in re.finditer(
            r"^\s*#define\s+(SPARK_LLM_\w+)\s+(\d+)u?\b", text, re.M):
        literals[match.group(1)] = int(match.group(2))

    def resolve(expression):
        stripped = re.sub(r"(\d+)u\b", r"\1", expression)
        names = set(re.findall(r"[A-Za-z_]\w*", stripped))
        for name in names:
            if name in literals:
                continue
            derived = re.search(
                rf"^\s*#define\s+{name}\s+\(([^)]*)\)", text, re.M)
            if not derived:
                fail(f"cannot resolve {name} in {path}")
            literals[name] = resolve(derived.group(1))
        clean = re.sub(r"[A-Za-z_]\w*", lambda m: str(literals[m.group(0)]),
                       stripped)
        if not re.fullmatch(r"[\d\s()/+*-]*", clean):
            fail(f"unsupported define expression: {expression}")
        return int(eval(clean))  # noqa: S307 - header-controlled arithmetic

    values = {}
    for name, macro in UINT_DEFINES.items():
        match = re.search(rf"^\s*#define\s+{macro}\s+(\d+)u?\b", text, re.M)
        if match:
            values[name] = int(match.group(1))
            continue
        derived = re.search(rf"^\s*#define\s+{macro}\s+\(([^)]*)\)", text,
                            re.M)
        if not derived:
            fail(f"{macro} not found in {path}")
        values[name] = resolve(derived.group(1))
    return values


def kv_floor_bytes(defines):
    """B1 smoke KV floor per NODE from the committed geometry.

    Laguna caches full key+value rows (GQA, one kv head per TP rank at
    TP8): per-layer slot = kv_heads/TP x 2 x head_dim x 2B. Every layer
    carries the same slot shape (the sliding window bounds live
    positions, not the slot bytes); the floor bounds the token count
    above the fixture set.
    """
    slot = (defines["kv_head_count"] // defines["tp_degree"]) \
        * 2 * defines["head_dimension"] * (defines["kv_bits"] // 8)
    return defines["layer_count"] * slot * KV_FLOOR_TOKENS \
        * KV_FLOOR_SEQUENCES


def dump_rank(pack_path):
    """One rank summary: expert spans + spine bytes + digest identity."""
    pack_path = os.path.abspath(pack_path)
    pack_bytes = os.path.getsize(pack_path)
    with open(pack_path, "rb") as handle:
        raw = handle.read(HEADER_BYTES)
        if len(raw) != HEADER_BYTES:
            fail(f"{pack_path}: shorter than the fixed header")
        header = struct.unpack_from("<20I2Q65s32s32s32s", raw, 0)
        (magic, version, header_bytes, entry_bytes, _abi, _flags,
         tensor_count, _stage_count, _stage_index, _first_layer,
         _layer_count, _total_layers, _hidden, _vocab, _routed_experts,
         _linear_codec, expert_codec, _kv_codec, _r0, _r1,
         directory_offset, file_bytes) = header[:22]
        if magic != PACK_MAGIC or version != PACK_VERSION:
            fail(f"{pack_path}: not a laguna v1 pack")
        if header_bytes != HEADER_BYTES or entry_bytes != ENTRY_BYTES:
            fail(f"{pack_path}: unexpected directory layout")
        if file_bytes != pack_bytes:
            fail(f"{pack_path}: header file_bytes {file_bytes} != {pack_bytes}")
        if expert_codec != CODEC_BF16:
            fail(f"{pack_path}: expert codec is not the placed bf16 arm")
        handle.seek(directory_offset)
        directory = handle.read(tensor_count * ENTRY_BYTES)
        if len(directory) != tensor_count * ENTRY_BYTES:
            fail(f"{pack_path}: short directory read")
    expert_spans = {}       # layer -> {"w1": bytes, "w2": bytes} per expert
    expert_bytes_total = 0
    for index in range(tensor_count):
        entry = struct.unpack_from("<8I4Q", directory, index * ENTRY_BYTES)
        (tensor_kind, layer_index, payload_type, weight_codec,
         scale_encoding, group_count, rows, columns, payload_offset,
         payload_bytes, _scale_offset, scale_bytes) = entry
        if tensor_kind not in (K_EXPERT_GATE_UP, K_EXPERT_DOWN):
            continue
        if payload_type != PAYLOAD_BF16 or weight_codec != CODEC_BF16 \
                or scale_bytes != 0:
            fail(f"{pack_path}: layer {layer_index} expert tensor not bf16")
        if group_count < 2 or payload_bytes % group_count != 0:
            fail(f"{pack_path}: layer {layer_index} non-uniform interleave")
        per_expert = payload_bytes // group_count
        if rows * columns * 2 != per_expert:
            fail(f"{pack_path}: layer {layer_index} per-expert {per_expert} "
                 f"!= rows*cols*2 = {rows * columns * 2}")
        which = "w1" if tensor_kind == K_EXPERT_GATE_UP else "w2"
        spans = expert_spans.setdefault(int(layer_index), {})
        if which in spans:
            fail(f"{pack_path}: duplicate expert tensor on layer {layer_index}")
        spans[which] = per_expert
        expert_bytes_total += payload_bytes
    sidecar = pack_path + ".experts"
    experts_records = None
    if os.path.isfile(sidecar):
        with open(sidecar, "rb") as handle:
            head = handle.read(16)
        magic, version, count, _ = struct.unpack("<IIII", head)
        if magic != EXPERTS_MAGIC or version != 2:
            fail(f"{sidecar}: not a v2 experts sidecar")
        experts_records = count
    digest_path = pack_path + ".sha256"
    digest = None
    if os.path.isfile(digest_path):
        text = open(digest_path).read().split()
        if text:
            digest = text[0]
    record = {
        "pack": os.path.basename(pack_path),
        "pack_bytes": pack_bytes,
        "pack_sha256": digest,
        "spine_bytes": pack_bytes - expert_bytes_total,
        "expert_bytes": expert_bytes_total,
        "expert_layers": len(expert_spans),
        "expert_records": experts_records,
        "expert_span_bytes": {
            str(layer): {"w1": spans["w1"], "w2": spans["w2"]}
            for layer, spans in sorted(expert_spans.items())},
    }
    print(json.dumps(record, sort_keys=True))


def read_rank_summaries(path):
    summaries = [json.loads(line) for line in open(path)
                 if line.strip()]
    if len(summaries) != NODES:
        fail(f"expected {NODES} rank summaries, got {len(summaries)}")
    return summaries


def byte_bases(summaries, defines):
    """Both byte bases + laguna's own chunk-union factor from the spans."""
    layer_count = defines["layer_count"]
    first_routed = defines["first_routed_layer"]
    experts = defines["expert_count"]
    placements = Counter()      # routed layer -> packs holding its experts
    span_shapes = set()
    spine_total = 0
    for record in summaries:
        for layer, spans in record["expert_span_bytes"].items():
            span_shapes.add((spans["w1"], spans["w2"]))
            placements[int(layer)] += 1
        spine_total += record["spine_bytes"]
        expected_pairs = 2 * experts * record["expert_layers"]
        if record["expert_records"] is not None \
                and record["expert_records"] != expected_pairs:
            fail(f"{record['pack']}: .experts records "
                 f"{record['expert_records']} != 2 x {experts} x "
                 f"{record['expert_layers']}")
    if len(span_shapes) != 1:
        fail(f"expert spans are not uniform across ranks: {span_shapes}")
    w1, w2 = span_shapes.pop()
    routed_layers = sorted(placements)
    if routed_layers != list(range(first_routed, layer_count)):
        fail(f"routed layer coverage is not {first_routed}.."
             f"{layer_count - 1} across the rank packs")
    chunk_union = sorted(set(placements.values()))
    if chunk_union != [TP]:
        fail(f"chunk-union factor is not {TP}: {chunk_union} (laguna "
             f"replicates every expert across the stage's TP group)")
    per_rank_expert = w1 + w2
    full_expert = per_rank_expert * TP
    return {"per_rank_expert_span_bytes": {"w1": w1, "w2": w2},
            "full_model_expert_bytes": full_expert,
            "chunk_union_factor": TP,
            "spine_bytes": spine_total}


def fixture_census(repo):
    """Touched (layer, expert) census from the committed T1R fixtures."""
    fixture_dir = os.path.join(repo, "qualification/t1_reference", FAMILY)
    manifest = json.load(open(os.path.join(fixture_dir, "MANIFEST.json")))
    prompts_sha = sha256_file(os.path.join(fixture_dir, "prompts.json"))
    if manifest["prompts_sha256"] != prompts_sha:
        fail("prompts.json drifted from the recorded fixture identity")
    touched = Counter()
    positions = 0
    used = []
    for name in sorted(manifest["fixtures"]):
        path = os.path.join(fixture_dir, name)
        if sha256_file(path) != manifest["fixtures"][name]["sha256"]:
            fail(f"fixture {name} drifted from the recorded identity")
        data = open(path, "rb").read()
        if data[:4] != T1R_MAGIC:
            fail(f"{name}: not a T1R1 fixture")
        length = struct.unpack("<Q", data[4:12])[0]
        meta = json.loads(data[12:12 + length])
        offset = 12 + length
        arrays = 0
        for entry in meta["arrays"]:
            size = struct.unpack("<Q", data[offset:offset + 8])[0]
            raw = zlib.decompress(data[offset + 8:offset + 8 + size])
            offset += 8 + size
            name_in = entry["name"]
            if not name_in.endswith("_route_ids"):
                continue
            arrays += 1
            layer = int(name_in.split("_layer")[1].split("_")[0])
            count = len(raw) // 4
            ids = struct.unpack(f"<{count}i", raw)
            if len(set(ids)) != count:
                fail(f"{name}: duplicate ids in one position's route set")
            touched.update((layer, expert) for expert in ids)
            positions += 1
        used.append({"fixture": name, "route_arrays": arrays})
    return manifest, touched, positions, used


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump-rank", metavar="PACK",
                        help="print one rank pack summary (collection mode)")
    parser.add_argument("--ranks-json", metavar="PATH",
                        help="collected per-rank summaries (producer mode)")
    parser.add_argument("--repo", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--emit", default=None)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    if arguments.dump_rank:
        dump_rank(arguments.dump_rank)
        return 0
    if not arguments.ranks_json:
        parser.error("producer mode requires --ranks-json")

    repo = arguments.repo
    emit_path = arguments.emit or os.path.join(
        repo, "model-families", FAMILY, "smoke_experts.json")

    defines = parse_defines(os.path.join(
        repo, "model-families", FAMILY,
        "include/sparkpipe/llm_defines.h"))
    if defines["layer_count"] * defines["moe_top_k"] < 1:
        fail("degenerate defines")
    summaries = read_rank_summaries(arguments.ranks_json)
    bases = byte_bases(summaries, defines)
    manifest, touched, positions, fixture_used = fixture_census(repo)

    head = sorted(pair for pair, count in touched.items()
                  if count >= CUT_RULE)
    selections = sum(touched.values())
    covered = sum(count for count in touched.values() if count >= CUT_RULE)
    if not head:
        fail("repetition head is empty; the cut rule excludes everything")
    full_bytes = bases["full_model_expert_bytes"]
    per_rank = sum(bases["per_rank_expert_span_bytes"].values())
    if bases["spine_bytes"] <= 0:
        fail("spine byte base is empty")

    # The cut stays honest the same way the k3 ruling pins it: the whole
    # tradeoff curve is recorded. Per-NODE preload cost for laguna is the
    # pairs of ONE stage's layers at the per-rank span (the stage's node
    # holds every expert of those layers), reported per stage; the
    # full/16 metric is kept alongside for fleet comparability.
    curve = []
    for cut in range(1, CUT_RULE + 3):
        kept = [count for count in touched.values() if count >= cut]
        if not kept:
            break
        per_stage = {stage: 0 for stage in (0, 1)}
        for (layer, _expert), count in touched.items():
            if count >= cut:
                per_stage[layer // (defines["layer_count"] // 2)] += 1
        curve.append({
            "min_positions": cut,
            "pairs": len(kept),
            "experts_gib_per_node": round(
                len(kept) * full_bytes / NODES / 2 ** 30, 3),
            "experts_mib_per_node_worst_stage": round(
                max(per_stage.values()) * per_rank / 2 ** 20, 2),
            "selection_coverage": round(sum(kept) / selections, 4),
        })

    document = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "prompt_set": "laguna-t1-prompts-v1",
        "topology": TOPOLOGY,
        "nodes": NODES,
        "expert_shard": "tp",
        "codec": CODEC,
        "spine_bytes": bases["spine_bytes"],
        "kv_floor_bytes": kv_floor_bytes(defines),
        "workspace_bytes": WORKSPACE_BYTES,
        "experts": [
            {"layer": layer, "expert": expert,
             "codec": CODEC, "bytes": full_bytes}
            for layer, expert in head],
        "provenance": {
            "byte_bases": {
                "chunk_union_factor": bases["chunk_union_factor"],
                "full_model_expert_bytes": full_bytes,
                "per_rank_expert_span_bytes":
                    bases["per_rank_expert_span_bytes"],
                "per_node_note": (
                    "a node's preload cost is its stage's pairs at the "
                    "per-rank span (laguna replicates expert identity "
                    "across the stage TP group; the span is the "
                    "intermediate-dim shard); experts_gib_per_node in "
                    "the curve is the fleet-comparable full/16 metric, "
                    "experts_mib_per_node_worst_stage is the real cost"),
            },
            "census": {
                "coverage_curve": curve,
                "cut_rule": (
                    f">= {CUT_RULE} positions (repetition head; the "
                    f"<= ~2 GiB/node law picks {CUT_RULE}: cut "
                    f"{CUT_RULE - 1} measures "
                    f"{curve[0]['experts_mib_per_node_worst_stage']} MiB "
                    f"on the worst stage - over; the curve is pinned and "
                    f"moves with M3 stray data, k3-style)"),
                "head_pairs": len(head),
                "head_selection_coverage": round(covered / selections, 4),
                "positions": positions,
                "prompt_set_sha256": manifest["prompts_sha256"],
                "route_arrays": sum(entry["route_arrays"]
                                    for entry in fixture_used),
                "selections": selections,
                "stray_rule": ("lazy attach covers pairs below the cut; "
                               "the M3 receipt measures the real stray "
                               "rate against head_selection_coverage"),
                "touched_pairs": len(touched),
                "used_fixtures": fixture_used,
            },
            "fixtures": [
                {"fixture": entry["fixture"],
                 "sha256": manifest["fixtures"][entry["fixture"]]["sha256"],
                 "route_arrays": entry["route_arrays"]}
                for entry in fixture_used],
            "kv_floor": {
                "tokens": KV_FLOOR_TOKENS,
                "sequences": KV_FLOOR_SEQUENCES,
            },
            "llm_defines_sha256": sha256_file(os.path.join(
                repo, "model-families", FAMILY,
                "include/sparkpipe/llm_defines.h")),
            "ranks": summaries,
        },
    }
    rendered = json.dumps(document, indent=1, sort_keys=True) + "\n"
    if arguments.check:
        current = open(emit_path, encoding="utf-8").read()
        if current != rendered:
            print(f"drift in {emit_path}: regenerate with "
                  f"tools/{FAMILY}_smoke_experts.py --ranks-json ...",
                  file=sys.stderr)
            return 1
        print("smoke experts manifest matches")
        return 0
    with open(emit_path, "w", encoding="utf-8") as handle:
        handle.write(rendered)
    worst = max(
        Counter(layer // (defines["layer_count"] // 2)
                for layer, _expert in head).values())
    print(json.dumps({
        "emitted": emit_path,
        "experts": len(head),
        "coverage": round(covered / selections, 4),
        "worst_stage_pairs": worst,
        "worst_stage_mib_per_node": round(worst * per_rank / 2 ** 20, 2),
        "chunk_union_factor": bases["chunk_union_factor"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
