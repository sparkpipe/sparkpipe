#!/usr/bin/env python3
"""Generate model-families/ling/smoke_experts.json.

Machine-generated smoke-expert manifest for the multidev lane budget
calculator (tools/devcycle/lane_budget_calc.py, PR #1083 convention; the
k3 repetition-head precedent #1131 and the dsv41_flash canonical set
#1093 are the family precedents). Never hand-edit the emitted JSON;
regenerate with this tool.

The manifest IS the batch-preload working set for the < 5 s cold-launch
milestone - never an aspiration list. At ling geometry one forward
position routes 8 x 40 = 320 distinct (layer, expert) pairs and the 21
recorded smoke positions touch 3,592 distinct pairs; the >= 1 union
would preload ~2.45 GiB/node, over the <= ~2 GiB/node fleet smoke
target. The sanctioned mechanism (ruling of 2026-09-23): experts[] =
the repetition head of the touched census (pairs routed in at least
CUT_RULE positions) and the lazy-attach path covers the strays; the
FULL census, the cut rule and the coverage fraction are pinned in
provenance. The M3 receipt must measure the real lazy-stray rate
against that coverage claim.

Byte bases (both, fail-closed, provenance pinned):
  1. per-rank TP16-shard expert spans, straight from the deployed pack
     directories (ling .lspk format: kind 16 expert_up_gate + kind 17
     expert_down entries, group_count = 512 experts per rank - the
     experts are COL-SHARDED on the inter dimension, so every rank pack
     holds every (layer, expert) pair's shard), cross-checked against
     the .experts sidecar records and the rank0 emission receipt
     (spine 625,519,384 B + expert payload 15,099,494,400 B per rank);
  2. the spine bytes per rank pack (every non-expert tensor extent,
     full-resolution bf16 - the quality law: no requantization).
The full-model expert bytes use ling's own chunk-union factor - the
number of rank packs that hold a given (layer, expert) pair - computed
from the collected summaries, never assumed from another family.

Inputs
  --ranks-json : per-rank pack summaries collected with --dump-rank on
                 every rank (16 records for TP16)
  --repo       : repository root (fixtures, defines, emit path)
  --dump-rank  : alternative mode - print ONE rank summary as JSON for
                 collection: python3 <this-file> --dump-rank PACK

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

# Pack-format constants, kept in lockstep with tools/ling_stagepack.py
# (the packer). Inlined - not imported - so --dump-rank runs from a bare
# python3 on any fleet node without the tool tree; the committed test
# pins these against ling_stagepack.py so drift fails the gate.
PACK_MAGIC = 0x33474E4C
PACK_ENTRY_BYTES = 64
PACK_HIDDEN = 2560
PACK_VOCAB = 157184
PACK_EXPERTS = 512
PACK_LAYERS = 42
PACK_CODEC_BF16 = 1
PACK_CODEC_FP8 = 5

FAMILY = "ling"
SCHEMA_VERSION = 1
TOPOLOGY = "TP16"
NODES = 16
TP = 16
CODEC = "bf16"                 # the sizing arm: wrapper default, and the
                               # larger of the two placed arms (fp8 packs
                               # smaller), so the budget bounds both
CUT_RULE = 2                   # pairs routed in >= this many positions
KV_FLOOR_TOKENS = 128          # bound above the fixture set (21 positions)
KV_FLOOR_SEQUENCES = 1         # shared-lane smoke keeps B1 (template law)
WORKSPACE_BYTES = 256 * 1024 * 1024   # placeholder; M3/M4 measured number

PACK_HEADER_BYTES = 264
PACK_DIRECTORY_OFFSET = 512
KIND_EXPERT_UP_GATE = 16
KIND_EXPERT_DOWN = 17
T1R_MAGIC = b"T1R1"
EXPERTS_SIDECAR_MAGIC = 0x58504557

# The rank0 emission receipt recorded at placement time (lingfin shares
# the byte plan; ling's own receipts/rankN.json carries the same fields).
RECEIPT_SPINE_BYTES = 625519384
RECEIPT_EXPERT_BYTES = 15099494400

UINT_DEFINES = {
    "layer_count": "SPARK_LLM_LAYER_COUNT",
    "first_routed_layer": "SPARK_LLM_FIRST_ROUTED_LAYER",
    "mtp_layer": "SPARK_LLM_MTP_LAYER_INDEX",
    "expert_count": "SPARK_LLM_MOE_EXPERT_COUNT",
    "moe_top_k": "SPARK_LLM_MOE_TOP_K",
    "attention_period": "SPARK_LLM_ATTENTION_PERIOD",
    "global_attention_phase": "SPARK_LLM_GLOBAL_ATTENTION_PHASE",
    "mla_layers": "SPARK_LLM_MLA_LAYER_COUNT",
    "kda_layers": "SPARK_LLM_KDA_LAYER_COUNT",
    "mla_latent": "SPARK_LLM_MLA_LATENT_DIMENSION",
    "mla_rope": "SPARK_LLM_MLA_QK_ROPE_HEAD_DIMENSION",
    "kv_bytes_per_scalar": "SPARK_LING_KV_BYTES_PER_SCALAR",
    "kda_heads": "SPARK_LLM_KDA_HEAD_COUNT",
    "kda_key_dim": "SPARK_LLM_KDA_HEAD_KEY_DIMENSION",
    "kda_value_dim": "SPARK_LLM_KDA_HEAD_VALUE_DIMENSION",
    "kda_state_element_bytes": "SPARK_LLM_KDA_STATE_ELEMENT_BYTES",
    "kda_conv_kernel": "SPARK_LLM_KDA_CONV_KERNEL",
}


def fail(message):
    raise SystemExit(f"ling_smoke_experts: FAIL: {message}")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_defines(repo):
    """Numeric defines from the committed ling headers (the model header
    chains llm_defines.h; the KV geometry header carries the scalar)."""
    text = ""
    for name in ("spark_ling_model.h", "llm_defines.h",
                 "spark_ling_kv_geometry.h"):
        text += open(os.path.join(
            repo, "model-families", FAMILY, "include", "sparkpipe",
            name), encoding="utf-8").read()
    values = {}
    for name, macro in UINT_DEFINES.items():
        match = re.search(rf"^\s*#define\s+{macro}\s+(\d+)u?\b", text, re.M)
        if match:
            values[name] = int(match.group(1))
            continue
        # SPARK_LLM_MTP_LAYER_INDEX aliases SPARK_LLM_LAYER_COUNT (the
        # MTP layer rides after the dense stack; packs omit it).
        alias = re.search(rf"^\s*#define\s+{macro}\s+(SPARK_\w+)u?\b",
                          text, re.M)
        if alias and alias.group(1) == "SPARK_LLM_LAYER_COUNT":
            values[name] = values["layer_count"]
            continue
        fail(f"{macro} not found in the ling headers")
    return values


def kv_floor_bytes(defines):
    """B1 smoke KV floor from the committed geometry (no fixtures needed).

    MLA layers cache (latent + rope) scalars per token at bf16; KDA
    layers carry a per-sequence f32 state plus a bf16 conv window per
    layer. The KDA state dominates - this is why the floor uses one
    sequence; shared-lane smoke keeps B1.
    """
    scalars = defines["mla_latent"] + defines["mla_rope"]
    mla = defines["mla_layers"] * scalars * defines["kv_bytes_per_scalar"] \
        * KV_FLOOR_TOKENS * KV_FLOOR_SEQUENCES
    state = defines["kda_heads"] * defines["kda_key_dim"] \
        * defines["kda_value_dim"] * defines["kda_state_element_bytes"]
    conv = 3 * (defines["kda_heads"] * defines["kda_key_dim"]) \
        * defines["kda_conv_kernel"] * defines["kv_bytes_per_scalar"]
    kda = defines["kda_layers"] * (state + conv) * KV_FLOOR_SEQUENCES
    return mla + kda


def dump_rank(pack_path):
    """One rank summary: expert spans + spine bytes + digest identity.

    Walks the ling .lspk directory (the tools/ling_verify_pack.py layout
    contract): 264-byte header, directory at 512, 64-byte entries. The
    expert tier is kinds 16+17 (grouped, one group per expert, scale
    planes included on the fp8 arm); everything else is spine.
    """
    pack_path = os.path.abspath(pack_path)
    with open(pack_path, "rb") as handle:
        head = handle.read(PACK_HEADER_BYTES)
        if len(head) != PACK_HEADER_BYTES:
            fail(f"{pack_path}: short header")
        (magic, version, header_bytes, entry_bytes, codec_abi, flags,
         tensor_count, stage_count, stage_index, first_layer, layer_count,
         total_layers, hidden, vocab, experts, linear_codec, expert_codec,
         kv_codec, tp32, rank32) = struct.unpack("<20I", head[:80])
        directory_offset, file_bytes = struct.unpack("<QQ", head[80:96])
        if magic != PACK_MAGIC or version != 1:
            fail(f"{pack_path}: not a ling v1 pack")
        if flags != 0:
            fail(f"{pack_path}: flags {flags:#x}, MTP must be omitted")
        if (hidden, vocab, experts) != (PACK_HIDDEN, PACK_VOCAB,
                                         PACK_EXPERTS):
            fail(f"{pack_path}: geometry {hidden}x{vocab}x{experts}")
        if expert_codec not in (PACK_CODEC_BF16, PACK_CODEC_FP8) \
                or linear_codec != PACK_CODEC_BF16 \
                or kv_codec != PACK_CODEC_BF16:
            fail(f"{pack_path}: codecs linear {linear_codec} expert "
                 f"{expert_codec} kv {kv_codec}")
        if (tp32, rank32) != (TP, int(re.search(r"rank([0-9a-f]+)\.",
                                                os.path.basename(pack_path))
                                    .group(1), 16)):
            fail(f"{pack_path}: header tp{tp32} rank {rank32}")
        if directory_offset != PACK_DIRECTORY_OFFSET:
            fail(f"{pack_path}: directory at {directory_offset}")
        if directory_offset + tensor_count * PACK_ENTRY_BYTES > file_bytes:
            fail(f"{pack_path}: directory exceeds file")
        handle.seek(directory_offset)
        directory = handle.read(tensor_count * PACK_ENTRY_BYTES)
        expert_spans = {}   # layer -> {16: bytes, 17: bytes} total per kind
        expert_scale_bytes = 0
        spine_bytes = 0
        expert_groups = Counter()
        for index in range(tensor_count):
            (kind, layer, payload_type, weight_codec, scale_encoding,
             group_count, rows, columns, payload_offset, payload_bytes,
             scale_offset, scale_bytes) = struct.unpack_from(
                "<IIIIIIIIQQQQ", directory, index * PACK_ENTRY_BYTES)
            if payload_type not in (1, 2, 4):
                fail(f"{pack_path}[{index}] kind{kind}: payload "
                     f"{payload_type}")
            if kind in (KIND_EXPERT_UP_GATE, KIND_EXPERT_DOWN):
                if group_count != PACK_EXPERTS:
                    fail(f"{pack_path}[{index}] kind{kind}: groups "
                         f"{group_count} != {PACK_EXPERTS}")
                expert_spans.setdefault(layer, {})[kind] = payload_bytes
                expert_scale_bytes += scale_bytes
                expert_groups[layer] += group_count
            else:
                spine_bytes += payload_bytes + scale_bytes
    sidecar = pack_path + ".experts"
    with open(sidecar, "rb") as handle:
        magic, version, count, _ = struct.unpack("<IIII", handle.read(16))
        if magic != EXPERTS_SIDECAR_MAGIC or version != 2:
            fail(f"{sidecar}: not a v2 experts sidecar")
        experts_records = count
    digest_path = pack_path + ".sha256"
    digest = open(digest_path).read().split()[0] \
        if os.path.isfile(digest_path) else None
    layers = sorted(expert_spans)
    record = {
        "pack": os.path.basename(pack_path),
        "pack_bytes": os.path.getsize(pack_path),
        "pack_sha256": digest,
        "expert_codec": {PACK_CODEC_BF16: "bf16",
                         PACK_CODEC_FP8: "fp8"}[expert_codec],
        "spine_bytes": spine_bytes,
        "expert_layers": len(layers),
        "expert_groups_per_layer": sorted(set(expert_groups.values())),
        "expert_records": experts_records,
        "expert_scale_bytes": expert_scale_bytes,
        "expert_span_bytes": {
            str(layer): {"up_gate": expert_spans[layer].get(
                             KIND_EXPERT_UP_GATE, 0),
                         "down": expert_spans[layer].get(
                             KIND_EXPERT_DOWN, 0)}
            for layer in layers},
    }
    print(json.dumps(record, sort_keys=True))


def read_rank_summaries(path):
    summaries = [json.loads(line) for line in open(path) if line.strip()]
    if len(summaries) != NODES:
        fail(f"expected {NODES} rank summaries, got {len(summaries)}")
    return summaries


def byte_bases(summaries, defines):
    """Both byte bases + ling's own chunk-union factor from the spans."""
    layer_count = defines["layer_count"]
    first_routed = defines["first_routed_layer"]
    mtp_layer = defines["mtp_layer"]
    experts = defines["expert_count"]
    bf16 = [r for r in summaries if r["expert_codec"] == CODEC]
    if len(bf16) != NODES:
        fail(f"the sizing arm is {CODEC}: {len(bf16)}/{NODES} summaries")
    placements = Counter()      # (layer, expert) slot presence per pack
    span_shapes = set()
    spine_total = 0
    routed_layers = set()
    for record in bf16:
        if record["expert_codec"] != CODEC:
            continue
        layers = record["expert_layers"]
        routed = mtp_layer - first_routed   # layers first_routed..mtp-1
        if layers != routed:
            fail(f"{record['pack']}: {layers} routed layers != {routed}")
        for layer, spans in record["expert_span_bytes"].items():
            span_shapes.add((spans["up_gate"], spans["down"]))
            # grouped entries: every rank pack carries ALL experts
            placements[int(layer)] += experts
        routed_layers.update(int(l) for l in record["expert_span_bytes"])
        spine_total += record["spine_bytes"]
        if record["expert_scale_bytes"] != 0:
            fail(f"{record['pack']}: bf16 arm must carry no scale planes")
        # 2 grouped expert tensors x layers x experts records
        if record["expert_records"] != 2 * layers * experts:
            fail(f"{record['pack']}: .experts records "
                 f"{record['expert_records']} != 2 x {experts} x {layers}")
        # spine cross-check against the placement receipt (rank0)
        if record["spine_bytes"] != RECEIPT_SPINE_BYTES:
            fail(f"{record['pack']}: spine {record['spine_bytes']} != "
                 f"receipt {RECEIPT_SPINE_BYTES}")
    if len(span_shapes) != 1:
        fail(f"expert spans are not uniform across ranks: {span_shapes}")
    up_gate, down = span_shapes.pop()
    expected_routed = set(range(first_routed, mtp_layer))
    if routed_layers != expected_routed:
        fail("routed layer coverage is not "
             f"{first_routed}..{mtp_layer - 1} across the rank packs")
    # chunk-union: rank packs holding a given (layer, expert) pair. The
    # experts are col-sharded on the inter dimension (group_count 512 on
    # every rank), so each of the 16 packs holds every pair's shard.
    per_layer_packs = sorted(set(placements.values()))
    if len(per_layer_packs) != 1:
        fail(f"layer presence is not uniform: {per_layer_packs}")
    chunk_union, _ = divmod(per_layer_packs[0], experts)
    if chunk_union != TP:
        fail(f"chunk-union factor is not {TP}: {chunk_union}")
    # up_gate/down are PER-LAYER totals (group_count x per-group bytes);
    # the per-(layer, expert) shard span divides by the expert count.
    per_layer_expert = up_gate + down
    if per_layer_expert % experts != 0 or up_gate % experts != 0 \
            or down % experts != 0:
        fail("expert spans are not divisible by the expert count")
    per_pair_expert = per_layer_expert // experts
    full_pair_expert = per_pair_expert * chunk_union
    if per_pair_expert * (mtp_layer - first_routed) * experts \
            != RECEIPT_EXPERT_BYTES:
        fail("per-pair expert bytes disagree with the emission receipt")
    return {"per_pair_expert_span_bytes": {
                "up_gate": up_gate // experts, "down": down // experts},
            "per_layer_expert_span_bytes": {"up_gate": up_gate,
                                            "down": down},
            "full_model_pair_expert_bytes": full_pair_expert,
            "chunk_union_factor": chunk_union,
            "spine_bytes": spine_total}


def fixture_census(repo, defines):
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
        fixture_sha = sha256_file(path)
        if fixture_sha != manifest["fixtures"][name]["sha256"]:
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
            if count != defines["moe_top_k"]:
                fail(f"{name_in}: topk {count} != "
                     f"{defines['moe_top_k']} (committed defines)")
            if not defines["first_routed_layer"] <= layer \
                    < defines["mtp_layer"]:
                fail(f"{name_in}: layer {layer} outside the routed span")
            ids = struct.unpack(f"<{count}i", raw)
            if min(ids) < 0 or max(ids) >= defines["expert_count"]:
                fail(f"{name_in}: expert id outside 0.."
                     f"{defines['expert_count'] - 1}")
            if len(set(ids)) != count:
                fail(f"{name_in}: duplicate ids in one position's route set")
            touched.update((layer, expert) for expert in ids)
            positions += 1
        used.append({"fixture": name, "route_arrays": arrays,
                     "sha256": fixture_sha,
                     "bytes": manifest["fixtures"][name]["bytes"]})
    if positions == 0:
        fail("no route arrays found in the fixture set")
    return touched, positions, used, prompts_sha


def repetition_curve(touched, positions):
    curve = []
    for cut in range(2, min(positions, 9)):
        kept = sum(1 for value in touched.values() if value >= cut)
        covered = sum(value for value in touched.values() if value >= cut)
        curve.append({"min_positions": cut, "pairs": kept,
                      "selection_coverage": round(
                          covered / sum(touched.values()), 3)})
    return curve


def main():
    parser = argparse.ArgumentParser(
        description="Generate model-families/ling/smoke_experts.json")
    parser.add_argument("--ranks-json")
    parser.add_argument("--repo",
                        default=os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))))
    parser.add_argument("--dump-rank", metavar="PACK")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()

    if arguments.dump_rank:
        dump_rank(arguments.dump_rank)
        return 0

    if not arguments.ranks_json:
        fail("--ranks-json is required (16 collected --dump-rank records)")

    defines = parse_defines(arguments.repo)
    summaries = read_rank_summaries(arguments.ranks_json)
    bases = byte_bases(summaries, defines)
    touched, positions, used, fixture_prompts_sha = fixture_census(
        arguments.repo, defines)
    curve = repetition_curve(touched, positions)

    experts_list = sorted(
        ({"layer": layer, "expert": expert} for (layer, expert), count
         in touched.items() if count >= CUT_RULE),
        key=lambda item: (item["layer"], item["expert"]))
    if not experts_list:
        fail("the ruled cut keeps no experts")
    kept_pairs = len(experts_list)
    kept_selections = sum(count for count in touched.values()
                          if count >= CUT_RULE)
    per_node_expert_bytes = kept_pairs * (
        bases["per_pair_expert_span_bytes"]["up_gate"]
        + bases["per_pair_expert_span_bytes"]["down"])
    if per_node_expert_bytes > 2 * 1024 ** 3:
        fail(f"ruled cut preloads {per_node_expert_bytes} B/node, over "
             "the <= 2 GiB fleet smoke target")
    kv_floor = kv_floor_bytes(defines)

    manifest = {
        # The flat calculator contract (k3 #1131 / qwen38_max #1085
        # schema): experts[].bytes = the FULL-model pair bytes (per-rank
        # shard x chunk-union) and expert_shard=tp, so
        # tools/devcycle/lane_budget_calc.py dividing by nodes reproduces
        # the per-node shard exactly.
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "topology": TOPOLOGY,
        "codec": CODEC,
        "nodes": NODES,
        "expert_shard": "tp",
        "prompt_set": "ling-t1-prompts-v1",
        "experts": [{"layer": item["layer"], "expert": item["expert"],
                     "codec": CODEC, "bytes":
                         bases["full_model_pair_expert_bytes"]}
                    for item in experts_list],
        "spine_bytes": bases["spine_bytes"],
        "kv_floor_bytes": kv_floor,
        "workspace_bytes": WORKSPACE_BYTES,
        "provenance": {
            "generator": "tools/ling_smoke_experts.py",
            "defines_sha256": hashlib.sha256(
                "".join(open(os.path.join(
                    arguments.repo, "model-families", FAMILY, "include",
                    "sparkpipe", name), encoding="utf-8").read()
                    for name in ("spark_ling_model.h", "llm_defines.h",
                                 "spark_ling_kv_geometry.h"))
                .encode("utf-8")).hexdigest(),
            "prompts_sha256": fixture_prompts_sha,
            "fixtures": used,
            "cut_rule": {"min_positions": CUT_RULE,
                         "positions": positions},
            "expert_pair_count": kept_pairs,
            "chunk_union_factor": bases["chunk_union_factor"],
            "span_bytes": {
                "per_pair_per_rank": bases["per_pair_expert_span_bytes"],
                "per_layer_per_rank": bases[
                    "per_layer_expert_span_bytes"],
                "per_node_experts": per_node_expert_bytes,
                "spine_per_node": bases["spine_bytes"] // NODES,
            },
            "defines": {
                "layer_count": defines["layer_count"],
                "first_routed_layer": defines["first_routed_layer"],
                "mtp_layer": defines["mtp_layer"],
                "expert_count": defines["expert_count"],
                "moe_top_k": defines["moe_top_k"],
                "mla_layers": defines["mla_layers"],
                "kda_layers": defines["kda_layers"],
            },
            "census": {
                "selections": sum(touched.values()),
                "distinct_pairs": len(touched),
                "positions": positions,
                "route_arrays": used,
                "kept_selection_coverage": round(
                    kept_selections / sum(touched.values()), 3),
                "repetition_curve": curve,
                "note": "experts[] = pairs routed in >= CUT_RULE of the "
                        "recorded positions (repetition head); lazy "
                        "attach covers the strays and the M3 receipt "
                        "must measure the real stray rate against "
                        "kept_selection_coverage",
            },
            "pack_identity": {
                "packs": [record["pack"] for record in summaries
                          if record["expert_codec"] == CODEC],
                "pack_sha256_prefixes": sorted({
                    (record["pack_sha256"] or "")[:8] for record in summaries
                    if record["expert_codec"] == CODEC}),
                "emission_receipt_spine_bytes": RECEIPT_SPINE_BYTES,
                "emission_receipt_expert_payload_bytes": RECEIPT_EXPERT_BYTES,
                "note": "per-rank spans computed from the pack "
                        "directories (kinds 16+17, groups=512 per rank: "
                        "the experts are col-sharded on the inter "
                        "dimension so every rank pack holds every "
                        "pair's shard - chunk-union 16, computed not "
                        "assumed); cross-checked against the .experts "
                        "record counts and the placement receipt",
            },
            "kv_floor_formula": "mla(7) x (latent+rope) x 2B x 128 "
                                "+ kda(35) x (heads x key x value x 4B "
                                "+ 3 x qk x conv x 2B); KDA state "
                                "dominates - B1 keeps the floor honest",
        },
    }

    emit_path = os.path.join(arguments.repo, "model-families", FAMILY,
                             "smoke_experts.json")
    rendered = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    if arguments.check:
        current = open(emit_path, encoding="utf-8").read()
        if current != rendered:
            print(f"drift in {emit_path}: regenerate with "
                  "tools/ling_smoke_experts.py", file=sys.stderr)
            return 1
        print("smoke_experts.json matches the collected bases")
        return 0
    with open(emit_path, "w", encoding="utf-8") as handle:
        handle.write(rendered)
    print(json.dumps({
        "emit": emit_path,
        "pairs": kept_pairs,
        "per_node_expert_mib": round(per_node_expert_bytes / 2 ** 20, 1),
        "coverage": manifest["provenance"]["census"][
            "kept_selection_coverage"],
        "chunk_union": bases["chunk_union_factor"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
