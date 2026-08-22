#!/usr/bin/env python3
"""W7 end-to-end case on the REAL DFlash2 weights.

Runs the landed reference forward (tools/qwen36_dspark_e2e_parity.forward, i.e.
qwen36_dspark_reference's own 5-layer conv-wrapped block forward on the real
drafter checkpoint plus the target's shared lm_head/embed_tokens) and dumps

  * the final-normed block hidden WITHOUT the anchor row (the oracle's
    hidden[1:]) as BF16 - exactly the tensor the module hands the selector,
  * the reference's selector expectations: stable top-16 ids, their BF16 unary
    logits, the context gate, the K x K lattice and the walked draft ids,
  * the raw BF16 weight blobs the selector reads: the target lm_head and the
    drafter's predecessor/successor codebooks and hidden projection,

so the CUDA side can load them with fread and run the module's own emit
sequence on the real 248320 x 5120 head. Synthesized taps and the fixed C0 are
the reference's own (default_rng(42), c0 12345, base position 64), so the case
is reproducible without any module run.

usage: qwen36_dflash2_selector_real_case.py OUTPUT_DIRECTORY
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen36_dspark_reference as ref  # noqa: E402
import qwen36_dspark_e2e_parity as rail  # noqa: E402

MAGIC = b"Q6DF2RW1"


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: qwen36_dflash2_selector_real_case.py OUTPUT_DIRECTORY")
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    taps = (rng.standard_normal((ref.TAPS, ref.HIDDEN)).astype(np.float32) * 0.1).astype(np.float32)
    c0 = 12345
    base_position = float(ref.BASE_POS)
    (output / "taps.bin").write_bytes(ref.f32_to_bf16(taps).astype("<u2").tobytes())

    logits, hidden, drafter = rail.forward(ref.bf16(taps), c0, base_position)
    mask_hidden = hidden[1:]
    slots = mask_hidden.shape[0]
    top_k = ref.SELECTOR_TOP_K
    rank = ref.SELECTOR_RANK

    mask_logits = logits[1:]
    top_ids = np.argsort(-mask_logits, axis=-1, kind="stable")[:, :top_k].astype(np.uint32)
    unary = np.take_along_axis(mask_logits, top_ids.astype(np.int64), axis=-1).astype(np.float32)
    gate = ref.bf16(mask_hidden @ drafter["candidate_selector.hidden_projection.weight"].T)
    edges = ref._score_edges(
        drafter["candidate_selector.predecessor_codebook"],
        drafter["candidate_selector.successor_codebook"],
        top_ids[None].astype(np.int64), unary[None], gate[None], np.array([c0]), top_k)[0].astype(np.float32)
    drafts = np.asarray(ref._greedy_walk(edges, top_ids), dtype=np.uint32)
    rail_drafts = rail.selector_drafts(hidden, logits, c0, drafter)
    if not np.array_equal(drafts, rail_drafts):
        raise SystemExit("internal: local selector tail disagrees with the landed rail")

    payload = bytearray()
    payload += MAGIC
    payload += struct.pack("<7I", slots, ref.VOCAB, ref.HIDDEN, rank, top_k, c0, int(base_position))
    payload += ref.f32_to_bf16(mask_hidden).astype("<u2").tobytes()
    payload += top_ids.astype("<u4").tobytes()
    payload += unary.astype("<f4").tobytes()
    payload += ref.f32_to_bf16(gate).astype("<u2").tobytes()
    payload += edges.astype("<f4").tobytes()
    payload += drafts.astype("<u4").tobytes()
    (output / "real_case.bin").write_bytes(bytes(payload))

    # Raw BF16 weight blobs the CUDA side freads. The lm_head comes from the
    # target checkpoint (the drafter shares it); the rest from the drafter.
    lm_head_bf16 = ref.read_safetensors_tensor(
        ref.TARGET / __import__("json").loads((ref.TARGET / "model.safetensors.index.json").read_text())
        ["weight_map"]["lm_head.weight"], "lm_head.weight")
    (output / "lm_head.bf16").write_bytes(np.ascontiguousarray(lm_head_bf16).tobytes())
    for name, filename in (("candidate_selector.predecessor_codebook", "predecessor.bf16"),
                           ("candidate_selector.successor_codebook", "successor.bf16"),
                           ("candidate_selector.hidden_projection.weight", "hidden_projection.bf16")):
        raw = ref.read_safetensors_tensor(ref.DRAFTER / "model.safetensors", name)
        (output / filename).write_bytes(np.ascontiguousarray(raw).tobytes())

    print(f"output          = {output}")
    print(f"slots           = {slots} vocab={ref.VOCAB} hidden={ref.HIDDEN} rank={rank} top_k={top_k} c0={c0}")
    print(f"hidden[0][:4]   = {mask_hidden[0][:4].tolist()}")
    print(f"top_ids[0][:4]  = {top_ids[0][:4].tolist()}")
    print(f"unary[0][:4]    = {unary[0][:4].tolist()}")
    print(f"edges[0,0,:4]   = {edges[0, 0, :4].tolist()}")
    print(f"reference drafts= {drafts.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
