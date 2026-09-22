#!/usr/bin/env python3
"""Generate model-families/k3/smoke_experts.json.

Machine-generated smoke-expert manifest for the multidev lane budget
calculator (tools/devcycle/lane_budget_calc.py, PR #1083 convention; the
k3 repetition-head precedent was manager-ruled for M2). Never hand-edit
the emitted JSON; regenerate with this tool.

The manifest IS the batch-preload working set for the < 5 s cold-launch
milestone - never an aspiration list. Under the shared-socket lazy
attach, one k3 forward position alone routes 16 x 92 = 1,472 distinct
(layer, expert) pairs (~1.5 GiB/node at TP4xPP4), so no honest smoke
prompt set fits the <= 2 GiB/node preload target by trimming alone. The
sanctioned mechanism (ruling of 2026-09-23): experts[] = the repetition
head of the touched census (pairs routed in at least CUT_RULE positions)
and the lazy-attach path covers the strays; the FULL census, the cut
rule and the coverage fraction are pinned in provenance. The M3 receipt
must measure the real lazy-stray rate against that coverage claim.

Byte bases (both, fail-closed, provenance pinned):
  1. per-rank TP4-shard expert spans, straight from the deployed pack
     tensor manifests (model.layers.L.expert_w{1,2}_weight interleave
     geometry), cross-checked against the .experts sidecar records;
  2. the spine bytes per rank pack (every non-expert tensor extent,
     full-resolution bf16 - the quality law: no requantization).
The full-model expert bytes use k3's own chunk-union factor - the
number of rank packs that hold a given (layer, expert) pair - computed
from the collected summaries, never assumed from another family.

Inputs
  --ranks-json : per-rank pack summaries collected with --dump-rank on
                 every rank (16 records for TP4xPP4)
  --repo       : repository root (fixtures, defines, emit path)
  --dump-rank  : alternative mode - print ONE rank summary as JSON for
                 collection: python3 - --dump-rank PACK <this-file

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

FAMILY = "k3"
SCHEMA_VERSION = 1
TOPOLOGY = "TP4xPP4"
NODES = 16
TP = 4
CODEC = "mxfp4"
CUT_RULE = 5                  # pairs routed in >= this many positions
KV_FLOOR_TOKENS = 128         # bound above the fixture set (16 positions)
KV_FLOOR_SEQUENCES = 1        # shared-lane smoke keeps B1 (template law)
WORKSPACE_BYTES = 256 * 1024 * 1024   # placeholder; M3/M4 measured number
PACK_MAGIC = 0x4B33504B       # "K3PK"
PACK_VERSION = 2
EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.expert_w([12])_weight$")
T1R_MAGIC = b"T1R1"

UINT_DEFINES = {
    "layer_count": "SPARK_K3_MODEL_LAYER_COUNT",
    "first_routed_layer": "SPARK_K3_MODEL_FIRST_ROUTED_LAYER",
    "expert_count": "SPARK_K3_MODEL_MOE_EXPERT_COUNT",
    "moe_top_k": "SPARK_K3_MODEL_MOE_TOP_K",
    "attention_period": "SPARK_K3_MODEL_ATTENTION_PERIOD",
    "global_attention_phase": "SPARK_K3_MODEL_GLOBAL_ATTENTION_PHASE",
    "mla_latent": "SPARK_K3_MODEL_MLA_LATENT_DIMENSION",
    "mla_unrotated": "SPARK_K3_MODEL_MLA_UNROTATED_DIMENSION",
    "kv_bytes_per_scalar": "SPARK_K3_KV_BYTES_PER_SCALAR",
    "kda_heads": "SPARK_K3_MODEL_KDA_HEAD_COUNT",
    "kda_key_dim": "SPARK_K3_MODEL_KDA_HEAD_KEY_DIMENSION",
    "kda_value_dim": "SPARK_K3_MODEL_KDA_HEAD_VALUE_DIMENSION",
    "kda_state_element_bytes": "SPARK_K3_MODEL_KDA_STATE_ELEMENT_BYTES",
    "kda_conv_kernel": "SPARK_K3_MODEL_KDA_CONV_KERNEL",
}


def fail(message):
    raise SystemExit(f"k3_smoke_experts: FAIL: {message}")


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
    """Numeric defines; derived ones (e.g. KV_BYTES_PER_SCALAR =
    (KV_BITS / 8u)) resolve over the header's own literal set."""
    text = open(path, encoding="utf-8").read()
    literals = {}
    for match in re.finditer(
            r"^\s*#define\s+(SPARK_K3_\w+)\s+(\d+)u?\b", text, re.M):
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
    """B1 smoke KV floor from the committed geometry (no fixtures needed).

    MLA layers cache (latent + unrotated) scalars per token; KDA layers
    carry a per-sequence state plus a conv window per layer. All bf16
    except the KDA state (f32).
    """
    layers = defines["layer_count"]
    period = defines["attention_period"]
    phase = defines["global_attention_phase"]
    mla_layers = layers // period + 1
    kda_layers = layers - mla_layers
    scalars = defines["mla_latent"] + defines["mla_unrotated"]
    mla = mla_layers * scalars * defines["kv_bytes_per_scalar"] \
        * KV_FLOOR_TOKENS * KV_FLOOR_SEQUENCES
    state = defines["kda_heads"] * defines["kda_key_dim"] \
        * defines["kda_value_dim"] * defines["kda_state_element_bytes"]
    conv = ((2 * defines["kda_heads"] * defines["kda_key_dim"])
            + (defines["kda_heads"] * defines["kda_value_dim"])) \
        * defines["kda_conv_kernel"] * defines["kv_bytes_per_scalar"]
    kda = kda_layers * (state + conv) * KV_FLOOR_SEQUENCES
    return mla + kda


def dump_rank(pack_path):
    """One rank summary: expert spans + spine bytes + digest identity."""
    pack_path = os.path.abspath(pack_path)
    with open(pack_path, "rb") as handle:
        magic, version, length = struct.unpack("<IIQ", handle.read(16))
        if magic != PACK_MAGIC or version != PACK_VERSION:
            fail(f"{pack_path}: not a k3 v2 pack")
        manifest = json.loads(handle.read(length))
    tensors = manifest["tensors"]
    expert_spans = {}       # layer -> {1: w1 bytes, 2: w2 bytes} (per expert)
    spine_bytes = 0
    for name, entry in tensors.items():
        match = EXPERT_RE.match(name)
        if match:
            layer, which = int(match.group(1)), int(match.group(2))
            geometry = entry.get("interleave") or {}
            span = geometry.get("expert_bytes", entry["bytes"]
                                // entry["shape"][0])
            expert_spans.setdefault(layer, {})[which] = span
        else:
            spine_bytes += entry["bytes"]
    sidecar = pack_path + ".experts"
    experts_records = 0
    with open(sidecar, "rb") as handle:
        magic, version, count, _ = struct.unpack("<IIII", handle.read(16))
        if magic != 0x58504557 or version != 2:
            fail(f"{sidecar}: not a v2 experts sidecar")
        experts_records = count
    digest_path = pack_path + ".sha256"
    digest = open(digest_path).read().split()[0] \
        if os.path.isfile(digest_path) else None
    record = {
        "pack": os.path.basename(pack_path),
        "pack_bytes": os.path.getsize(pack_path),
        "pack_sha256": digest,
        "spine_bytes": spine_bytes,
        "expert_layers": len(expert_spans),
        "expert_records": experts_records,
        "expert_span_bytes": {
            str(layer): {"w1": spans[1], "w2": spans[2]}
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
    """Both byte bases + k3's own chunk-union factor from the spans."""
    layer_count = defines["layer_count"]
    first_routed = defines["first_routed_layer"]
    experts = defines["expert_count"]
    placements = Counter()      # (layer, expert-slot presence per pack)
    span_shapes = set()
    spine_total = 0
    for record in summaries:
        layers = record["expert_layers"]
        expected = layer_count - first_routed - 0
        # stage packs hold their own contiguous layer range; across the 16
        # packs every routed layer must appear exactly TP times.
        for layer, spans in record["expert_span_bytes"].items():
            span_shapes.add((spans["w1"], spans["w2"]))
            placements[int(layer)] += 1
        spine_total += record["spine_bytes"]
        if record["expert_records"] != layers * experts * 2:
            fail(f"{record['pack']}: .experts records "
                 f"{record['expert_records']} != 2 x {experts} x {layers}")
    if len(span_shapes) != 1:
        fail(f"expert spans are not uniform across ranks: {span_shapes}")
    w1, w2 = span_shapes.pop()
    routed_layers = sorted(placements)
    if routed_layers != list(range(first_routed, layer_count)):
        fail("routed layer coverage is not 1..92 across the rank packs")
    chunk_union = sorted(set(placements.values()))
    if chunk_union != [TP]:
        fail(f"chunk-union factor is not {TP}: {chunk_union}")
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


WSET_NAME = "smoke-k3-v1.wset"
STAGE_LAYERS = ((0, 23), (24, 46), (47, 69), (70, 92))


def wset_bytes(pairs):
    return b"".join(struct.pack("<II", layer, expert)
                    for layer, expert in sorted(pairs))


def stage_pairs(pairs, stage):
    """Rank-local preload pairs: the layers of this PP stage only.

    A rank pack's manifest holds only its stage's routed layers, so a
    full-model working set cannot validate there (weightd_warm fails
    closed on any key absent from the pack manifest). k3's head spans
    all 92 routed layers, unlike dsv4's stage-0-local set.
    """
    first, last = STAGE_LAYERS[stage]
    local = [pair for pair in pairs if first <= pair[0] <= last]
    if not local:
        raise SystemExit("k3_smoke_experts: FAIL: stage "
                         f"{stage} preload subset is empty")
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump-rank", metavar="PACK",
                        help="print one rank pack summary (collection mode)")
    parser.add_argument("--ranks-json", metavar="PATH",
                        help="collected per-rank summaries (producer mode)")
    parser.add_argument("--repo", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--emit", default=None)
    parser.add_argument("--emit-wset", action="store_true",
                        help="derive the binary working set from the "
                             "committed manifest (writes "
                             "model-families/k3/" + WSET_NAME + ")")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    repo = arguments.repo

    if arguments.emit_wset:
        manifest_path = os.path.join(
            repo, "model-families", FAMILY, "smoke_experts.json")
        document = json.load(open(manifest_path))
        pairs = [(entry["layer"], entry["expert"])
                 for entry in document["experts"]]
        if len(set(pairs)) != len(pairs):
            fail("committed manifest experts are not deduplicated")
        target = os.path.join(repo, "model-families", FAMILY, WSET_NAME)
        temporary = target + f".tmp.{os.getpid()}"
        with open(temporary, "wb") as handle:
            handle.write(wset_bytes(pairs))
        os.replace(temporary, target)
        print(f"wrote {target}: {len(pairs)} key pairs "
              f"({len(pairs) * 8} bytes); per-stage subsets: "
              + ", ".join(f"s{s}={len(stage_pairs(pairs, s))}"
                          for s in range(4)))
        return 0

    if arguments.dump_rank:
        dump_rank(arguments.dump_rank)
        return 0
    if not arguments.ranks_json:
        parser.error("producer mode requires --ranks-json")

    emit_path = arguments.emit or os.path.join(
        repo, "model-families", FAMILY, "smoke_experts.json")

    defines = parse_defines(os.path.join(
        repo, "model-families", FAMILY,
        "include/sparkpipe/spark_k3_llm_defines.h"))
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
    if bases["spine_bytes"] <= 0:
        fail("spine byte base is empty")

    # The ruling (2026-09-23) freezes the cut but pins the honest
    # coverage AND the whole tradeoff curve: if the M3 stray-tax
    # measurement shows lazy attach dominating steady state, the cut
    # moves with data, not taste.
    curve = []
    for cut in range(2, CUT_RULE + 3):
        kept = [count for count in touched.values() if count >= cut]
        if not kept:
            break
        curve.append({
            "min_positions": cut,
            "pairs": len(kept),
            "experts_gib_per_node": round(
                len(kept) * full_bytes / NODES / 2 ** 30, 3),
            "selection_coverage": round(sum(kept) / selections, 4),
        })

    document = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "prompt_set": "k3-t1-prompts-v1",
        "topology": TOPOLOGY,
        "nodes": NODES,
        "expert_shard": "tp",
        "codec": CODEC,
        "spine_bytes": bases["spine_bytes"],
        "kv_floor_bytes": kv_floor_bytes(defines),
        "workspace_bytes": WORKSPACE_BYTES,
        "experts": [{"layer": layer, "expert": expert,
                     "codec": CODEC, "bytes": full_bytes}
                    for layer, expert in head],
        "provenance": {
            "cut_rule": f"routed in >= {CUT_RULE} of {positions} positions",
            "census": {
                "touched_pairs": len(touched),
                "route_arrays": positions,
                "head_pairs": len(head),
                "selections": selections,
                "head_selection_coverage": covered / selections,
                "coverage_curve": curve,
            },
            "byte_bases": {
                "per_rank_expert_span_bytes":
                    bases["per_rank_expert_span_bytes"],
                "chunk_union_factor": bases["chunk_union_factor"],
                "full_model_expert_bytes": full_bytes,
            },
            "ranks": [{"pack": r["pack"], "pack_sha256": r["pack_sha256"],
                       "pack_bytes": r["pack_bytes"],
                       "spine_bytes": r["spine_bytes"]}
                      for r in summaries],
            "fixtures": fixture_used,
            "prompts_sha256": manifest["prompts_sha256"],
            "llm_defines_sha256": sha256_file(os.path.join(
                repo, "model-families", FAMILY,
                "include/sparkpipe/spark_k3_llm_defines.h")),
            "kv_floor": {"tokens": KV_FLOOR_TOKENS,
                         "sequences": KV_FLOOR_SEQUENCES},
            "workspace_note": "placeholder; M3/M4 measured number replaces",
        },
    }
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    if arguments.check:
        current = open(emit_path, encoding="utf-8").read()
        if current != text:
            fail(f"{emit_path} is stale; regenerate with --ranks-json")
        print(f"{emit_path} matches the collected byte bases and census")
        return 0
    with open(emit_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    per_node = len(head) * full_bytes / NODES / 2 ** 30
    print(f"wrote {emit_path}: {len(head)} preload experts "
          f"(full census {len(touched)}, coverage "
          f"{covered / selections:.3f}), {per_node:.2f} GiB/node experts, "
          f"spine {bases['spine_bytes'] / 2 ** 30:.1f} GiB full model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
