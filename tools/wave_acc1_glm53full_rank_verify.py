#!/usr/bin/env python3
"""glm53full rank-pack parity oracle: warm checkpoint vs placed rank pack.

Re-derives the packer's own bytes for anchor entries by RE-RUNNING the lane
packer's plan producers (tools/glm52_resident_stagepack.py Packer.build_plan)
against the live warm checkpoint, then compares those bytes against the
placed rank pack read with tools/glm53full_bf16_tp16_source_verify.py's
PackReader. The packer code is the only statement of the transform; this
tool duplicates no slice/scale math.

Anchor set: embedding, final norm, lm_head (globals) plus, at layers
0 / middle / last: attn_norm, post_attn_norm, q_a, q_b, o_proj, and at
routed layers the router, f32 correction bias and shared experts; the
routed-expert payload + scale planes of the MIDDLE layer are compared for
sampled experts (--expert-samples).

Usage (node-local, next to the warm mount):
  python3 tools/wave_acc1_glm53full_rank_verify.py --arm bf16 \
      --pack ~/sparkdata/glm53full.bf16.tp16/packs/glm53full.bf16.tp16-rank5.glm52sp \
      --checkpoint /mnt/model-warm/glm-5.3-bf16 --tp-degree 16 --tp-rank 5

Exit 0 only on PASS; any mismatch names the entry.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import glm52_resident_stagepack as packer_mod  # noqa: E402
from glm53full_bf16_tp16_source_verify import PackReader  # noqa: E402

import struct  # noqa: E402

ENTRY_STRUCT = struct.Struct("<8I4Q")


class FullEntryReader(PackReader):
    def __init__(self, path: Path):
        super().__init__(path)
        with open(path, "rb") as handle:
            head = handle.read(packer_mod.HEADER_BYTES)
            directory_offset = struct.unpack_from("<2Q", head, 80)[0]
            handle.seek(directory_offset)
            raw = handle.read(self.tensor_count * packer_mod.ENTRY_BYTES)
        self.full = {}
        for i in range(self.tensor_count):
            e = ENTRY_STRUCT.unpack_from(raw, i * packer_mod.ENTRY_BYTES)
            self.full[(e[0], e[1])] = e

GLOBAL_LAYER = packer_mod.GLOBAL_LAYER
CODECS = {"bf16": packer_mod.CODEC_BF16, "fp8": packer_mod.CODEC_FP8,
          "nvfp4": packer_mod.CODEC_NVFP4}
K_EXPERT_UP_GATE = packer_mod.K_EXPERT_UP_GATE
K_EXPERT_DOWN = packer_mod.K_EXPERT_DOWN
EXPERT_KINDS = frozenset({K_EXPERT_UP_GATE, K_EXPERT_DOWN})
ROUTED_ONLY_KINDS = frozenset({
    packer_mod.K_ROUTER, packer_mod.K_ROUTER_CORRECTION,
    packer_mod.K_SHARED_GATE_UP, packer_mod.K_SHARED_DOWN,
})
SPINE_ANCHOR_KINDS = frozenset({
    packer_mod.K_ATTN_NORM, packer_mod.K_POST_ATTN_NORM, packer_mod.K_Q_A,
    packer_mod.K_Q_B, packer_mod.K_ATTN_OUTPUT,
})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=tuple(CODECS), required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--layer-count", type=int, default=78)
    parser.add_argument("--expert-samples", default="0,mid,last")
    parser.add_argument("--expert-layer", type=int, default=-1,
                        help="routed layer for the expert-plane anchor "
                             "(default: the middle layer)")
    args = parser.parse_args()

    source = packer_mod.Fp8SourceReader(args.checkpoint, cache_byte_cap=1 << 30)
    contract = packer_mod.load_contract(args.contract_root)
    last = args.layer_count - 1
    anchor_layers = {0, last // 2, last}
    expert_layer = last // 2 if args.expert_layer < 0 else args.expert_layer
    experts = contract["moe_expert_count"]
    picks = {"0": (0,), "mid": (experts // 2,), "last": (experts - 1,)}
    wanted_experts = sorted({e for key in args.expert_samples.split(",")
                             for e in picks[key]})

    plan_builder = packer_mod.Packer(
        source, contract, (0, last),
        tp_degree=args.tp_degree, tp_rank=args.tp_rank,
        expert_codec=CODECS[args.arm])
    plan_builder.build_plan()

    pack = FullEntryReader(args.pack)
    if pack.tp_degree != args.tp_degree or pack.tp_rank != args.tp_rank:
        print(f"FAIL pack tp {pack.tp_degree}/{pack.tp_rank} != requested "
              f"{args.tp_degree}/{args.tp_rank}", file=sys.stderr)
        return 1
    if pack.expert_codec != CODECS[args.arm]:
        print(f"FAIL pack expert codec {pack.expert_codec} != arm {args.arm}",
              file=sys.stderr)
        return 1

    routed_first = contract["first_routed_layer"]
    checked = 0
    failures = 0
    for item in plan_builder.plan:
        entry = item.entry
        layer = entry.layer
        if entry.kind in EXPERT_KINDS:
            if layer != expert_layer or layer < routed_first:
                continue
        elif entry.kind in ROUTED_ONLY_KINDS:
            if layer not in anchor_layers or layer < routed_first:
                continue
        elif entry.kind in SPINE_ANCHOR_KINDS:
            if layer not in anchor_layers:
                continue
        elif layer != GLOBAL_LAYER:
            continue
        try:
            failures += compare_entry(item, pack, entry.kind in EXPERT_KINDS,
                                      wanted_experts, experts)
            checked += 1
        except (packer_mod.PackFailure, RuntimeError) as error:
            print(f"FAIL kind={entry.kind} layer={hex(layer)}: {error}")
            failures += 1

    print(f"glm53full rank oracle arm={args.arm} rank={args.tp_rank}: "
          f"{checked} anchor entries checked, {failures} failures")
    if failures:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS — packer plan re-derivation matches the placed pack bytes")
    return 0


def compare_entry(item, pack: FullEntryReader, is_expert: bool,
                  wanted_experts: list[int], experts: int) -> int:
    entry = item.entry
    failures = 0
    if is_expert:
        return compare_expert_entry(item, pack, wanted_experts, experts)
    pack_entry = pack.full.get((entry.kind, entry.layer))
    if pack_entry is None:
        print(f"FAIL kind={entry.kind} layer={hex(entry.layer)}: "
              f"missing in placed pack")
        return 1
    payload = pack.mmap[pack_entry[8]:pack_entry[8] + pack_entry[9]].tobytes()
    produced = b"".join(item.produce_payload())
    if payload != produced:
        where = next((i for i, (a, b) in enumerate(zip(payload, produced))
                      if a != b), min(len(payload), len(produced)))
        print(f"FAIL kind={entry.kind} layer={hex(entry.layer)}: payload mismatch "
              f"near byte {where} (pack {len(payload)} vs re-derived {len(produced)})")
        failures += 1
    if entry.scale_bytes:
        scale = pack.mmap[pack_entry[10]:pack_entry[10] + pack_entry[11]].tobytes()
        produced_scale = b"".join(item.produce_scale())
        if scale != produced_scale:
            print(f"FAIL kind={entry.kind} layer={hex(entry.layer)}: "
                  f"scale plane mismatch")
            failures += 1
    return failures


def compare_expert_entry(item, pack: FullEntryReader, wanted_experts: list[int],
                         experts: int) -> int:
    entry = item.entry
    pack_entry = pack.full.get((entry.kind, entry.layer))
    if pack_entry is None:
        print(f"FAIL kind={entry.kind} layer={hex(entry.layer)}: "
              f"missing in placed pack")
        return 1
    projections = 2 if entry.kind == K_EXPERT_UP_GATE else 1
    per_expert_payload = entry.payload_bytes // experts
    per_projection_payload = per_expert_payload // projections
    failures = 0
    for index, blob in enumerate(item.produce_payload()):
        expert, projection = divmod(index, projections)
        if expert not in wanted_experts:
            continue
        offset = (expert * per_expert_payload
                  + projection * per_projection_payload)
        base = pack_entry[8]
        want = pack.mmap[base + offset:base + offset + len(blob)].tobytes()
        if want != blob:
            where = next((i for i, (a, b) in enumerate(zip(want, blob))
                          if a != b), min(len(want), len(blob)))
            print(f"FAIL kind={entry.kind} layer={hex(entry.layer)} expert={expert} "
                  f"projection={projection}: payload mismatch near byte {where}")
            failures += 1
    if not entry.scale_bytes:
        return failures
    per_expert_scale = entry.scale_bytes // experts
    flat = b"".join(item.produce_scale())
    if len(flat) != entry.scale_bytes:
        print(f"FAIL kind={entry.kind} layer={hex(entry.layer)}: re-derived scale "
              f"{len(flat)} bytes != pack plane {entry.scale_bytes}")
        return failures + 1
    base = pack_entry[10]
    for expert in wanted_experts:
        offset = expert * per_expert_scale
        want = pack.mmap[base + offset:base + offset + per_expert_scale].tobytes()
        if want != flat[offset:offset + per_expert_scale]:
            print(f"FAIL kind={entry.kind} layer={hex(entry.layer)} "
                  f"expert={expert}: scale mismatch")
            failures += 1
    return failures


if __name__ == "__main__":
    sys.exit(main())
