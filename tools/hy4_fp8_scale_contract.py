#!/usr/bin/env python3
"""hy4 FP8 scale-plane contract oracle.

Validates every F8_E4M3 payload + U8 E8M0 scale pair of a hy4-fp8-tp16-v1
rank pack against the scale-row-offset contract and emits the per-plane
table. Own math from the pinned publisher reference (llama.cpp hyv4.cpp
of the AngelSlim hy4-preview patch, vendored at
tools/hy4_dequant/vendor/hyv4_reference.cpp) and the checkpoint config;
no driver imports. Mirrors modules/hy4_resident_decode_stage/source/
spark_hy4_fp8_scale_contract.h rule for rule.

  python3 tools/hy4_fp8_scale_contract.py --header HEADER.json --tsv OUT.tsv
  python3 tools/hy4_fp8_scale_contract.py --pack rank-02.safetensors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tools" / "hy4_dequant" / "vendor" / "hyv4_reference.cpp"
REFERENCE_SHA256 = (
    "514ef62ae147171d5675229de59e8e6d3b8e1f84f20680056df8708601fe52fd")
REFERENCE_COMMIT = "0cea36222"
CHECKPOINT = "tencent/Hy4-preview-FP8"
CHECKPOINT_REVISION = "4215ec29de873a998e849cee902654490c7ff4d1"

GROUP = 32
HIDDEN = 6144
HEADS = 64
QK_HEAD = 256
QK_NOPE = 192
V_HEAD = 256
KV_LORA = 512
Q_LORA = 2048
INDEX_HEADS = 32
INDEX_HEAD_DIM = 128

ALIGNED = "ALIGNED"
REPL_ROWS = "REPLICATED_ROWS"
REPL_GROUPS = "REPLICATED_GROUPS"

RULES = {
    "self_attn.q_a_proj.weight": (ALIGNED, None, None),
    "self_attn.kv_a_proj_with_mqa.weight": (ALIGNED, None, None),
    "self_attn.indexer.wk.weight": (ALIGNED, None, None),
    "mlp.experts.gate_up_proj": (ALIGNED, None, None),
    "mlp.experts.down_proj": (ALIGNED, None, None),
    "mlp.shared_experts.gate_proj.weight": (ALIGNED, None, None),
    "mlp.shared_experts.up_proj.weight": (ALIGNED, None, None),
    "mlp.shared_experts.down_proj.weight": (ALIGNED, None, None),
    "mlp.gate_proj.weight": (ALIGNED, None, None),
    "mlp.up_proj.weight": (ALIGNED, None, None),
    "mlp.down_proj.weight": (ALIGNED, None, None),
    "self_attn.q_b_proj.weight": (REPL_ROWS, HEADS * QK_HEAD, Q_LORA),
    "self_attn.kv_b_proj.weight": (REPL_ROWS, HEADS * (QK_NOPE + V_HEAD),
                                   KV_LORA),
    "self_attn.indexer.wq_b.weight": (REPL_ROWS,
                                      INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA),
    "self_attn.o_proj.weight": (REPL_GROUPS, HIDDEN, HEADS * V_HEAD),
}
EXPERT_KINDS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


def kind_of(name):
    m = re.match(r"model\.(layers|mtp_layers)\.(\d+)\.(.*)$", name)
    if m is None:
        return None, None, None
    return m.group(3), int(m.group(2)), m.group(1) == "mtp_layers"


def contract_of(kind, rank, ranks, rows, cols):
    """Returns (rule, scale_rows, scale_groups, row_offset, group_offset)
    or a failure string. Same arithmetic as SparkHy4Fp8ScaleContractOf."""
    if kind not in RULES:
        return "no-rule"
    if ranks == 0 or rank >= ranks:
        return "rank"
    if rows == 0 or cols == 0 or cols % GROUP:
        return "geometry"
    rule, grows, gcols = RULES[kind]
    groups = cols // GROUP
    if rule == ALIGNED:
        return (rule, rows, groups, 0, 0)
    if rule == REPL_ROWS:
        if grows % ranks or rows != grows // ranks or cols != gcols:
            return "rows-tile"
        return (rule, grows, groups, rank * rows, 0)
    if gcols % ranks or cols != gcols // ranks or rows != grows:
        return "columns-tile"
    return (rule, rows, gcols // GROUP, 0, rank * groups)


def load_header(args):
    if args.header:
        return json.load(open(args.header))
    with open(args.pack, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def pin_reference():
    if not REFERENCE.exists():
        return "reference-absent"
    digest = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    return "pinned" if digest == REFERENCE_SHA256 else "REFERENCE-SHA-DRIFT"


def walk(header, rank, ranks):
    rows_out = []
    failures = 0
    for name in sorted(header):
        rec = header[name]
        if name == "__metadata__" or rec["dtype"] != "F8_E4M3":
            continue
        kind, layer, mtp = kind_of(name)
        shape = rec["shape"]
        scale = header.get(name + "_scale")
        prow = 1
        for d in shape[:-1]:
            prow *= d
        pcol = shape[-1]
        if scale is None or scale["dtype"] != "U8":
            verdict = "FAIL no-scale"
        elif kind is None:
            verdict = "FAIL no-kind"
        else:
            got = contract_of(kind, rank, ranks, prow, pcol)
            if isinstance(got, str):
                verdict = "FAIL " + got
            else:
                srow = 1
                for d in scale["shape"][:-1]:
                    srow *= d
                sgrp = scale["shape"][-1]
                rule, erow, egrp, roff, goff = got
                if (srow, sgrp) != (erow, egrp):
                    verdict = "FAIL scale-shape want %dx%d" % (erow, egrp)
                else:
                    verdict = "OK"
        if verdict != "OK":
            failures += 1
            rule, roff, goff, egrp = "-", 0, 0, 0
            print("SCALE-CONTRACT FAIL %s: %s payload=%s scale=%s" % (
                name, verdict, shape, scale["shape"] if scale else None),
                file=sys.stderr)
        klass = "EXPERT" if kind in EXPERT_KINDS else "SPINE"
        rows_out.append((-1 if mtp else layer, kind or name, klass, rule,
                         "x".join(str(d) for d in shape),
                         "x".join(str(d) for d in scale["shape"])
                         if scale else "-", roff, goff, egrp, verdict))
    return rows_out, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack")
    ap.add_argument("--header")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--ranks", type=int)
    ap.add_argument("--tsv")
    args = ap.parse_args()
    if bool(args.pack) == bool(args.header):
        ap.error("exactly one of --pack / --header")
    header = load_header(args)
    meta = header.get("__metadata__", {})
    rank = args.rank if args.rank is not None else int(meta["tp_rank"])
    ranks = args.ranks if args.ranks is not None else int(meta["tp_degree"])
    table, failures = walk(header, rank, ranks)
    if args.tsv:
        with open(args.tsv, "w") as f:
            f.write("layer\tkind\tclass\trule\tpayload\tscale\trow_offset"
                    "\tgroup_offset\tscale_stride\tverdict\n")
            for row in table:
                f.write("\t".join(str(c) for c in row) + "\n")
    by_rule = {}
    for row in table:
        key = (row[3], row[2])
        by_rule[key] = by_rule.get(key, 0) + 1
    print("reference %s %s@%s sha256 %s (%s); checkpoint %s@%s" % (
        REFERENCE.relative_to(ROOT), "hyv4.cpp", REFERENCE_COMMIT,
        REFERENCE_SHA256[:16], pin_reference(), CHECKPOINT,
        CHECKPOINT_REVISION[:12]))
    print("rank %d of %d: %d FP8 planes, %d contract failures" % (
        rank, ranks, len(table), failures))
    for (rule, klass), n in sorted(by_rule.items()):
        print("  %-18s %-6s %4d" % (rule, klass, n))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
