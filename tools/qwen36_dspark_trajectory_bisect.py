#!/usr/bin/env python3
"""Trajectory bisect: per-step drafter dumps vs the reference, and vs the golden.

The acceptance decay and the token-15 divergence need ONE table: for every step of
one sequence, what the target committed, what the drafter proposed, what the
reference forward says it should have proposed, and how degenerate that proposal
is. This builds that table from the module's per-step dumps.

INPUT - a directory of per-step dumps written by the module when
SPARK_QWEN36_DSPARK_DUMP_DIR is set:
    step_<base_position>_taps.bin     5 x 5120 BF16 target tap hidden
    step_<base_position>_c0.bin       the committed token the block anchors on
    step_<base_position>_drafts.bin   the B-1 draft ids the module emitted
optionally a golden token stream (one decimal id per line) from a no-spec run of
the same prompt, to locate the first stream divergence.

OUTPUT - one row per step:
    position, c0, module drafts, reference drafts, PARITY/MISMATCH, distinct draft
    count, whether every draft equals c0 (the "repeat the committed token"
    degeneracy), and the mean top-1 unary logit (the drafter's confidence).
Then the two bisect answers: the first step whose drafts diverge from the
reference (a DFlash2 defect - none so far), and the first step whose tap state is
degenerate (a target-trajectory defect).

The reference weights are loaded ONCE and reused for every step: rail.forward
reloads the 1.9B drafter plus the target's lm_head and embed_tokens on each call,
which is ~18 GB and a minute per step, so this caches those two loaders instead of
duplicating the forward (no logic drift - the same rail.forward runs).

usage: qwen36_dspark_trajectory_bisect.py DUMP_DIRECTORY [GOLDEN_TOKENS_FILE]
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen36_dspark_reference as ref  # noqa: E402
import qwen36_dspark_e2e_parity as rail  # noqa: E402

_WEIGHT_CACHE: dict = {}


def _cached(name, loader):
    def wrapper():
        if name not in _WEIGHT_CACHE:
            _WEIGHT_CACHE[name] = loader()
        return _WEIGHT_CACHE[name]
    return wrapper


ref.load_drafter = _cached("drafter", ref.load_drafter)
ref.load_target_shared = _cached("target", ref.load_target_shared)


def read_steps(directory: Path) -> list[tuple[int, np.ndarray, int, list[int]]]:
    steps = []
    for taps_path in sorted(directory.glob("step_*_taps.bin")):
        match = re.match(r"step_(\d+)_taps\.bin$", taps_path.name)
        if match is None:
            continue
        position = int(match.group(1))
        c0_path = directory / f"step_{position}_c0.bin"
        drafts_path = directory / f"step_{position}_drafts.bin"
        if not c0_path.exists() or not drafts_path.exists():
            continue
        raw = np.fromfile(taps_path, dtype=np.uint16)
        if raw.size != ref.TAPS * ref.HIDDEN:
            continue
        taps = ref.bf16_to_f32(raw).reshape(ref.TAPS, ref.HIDDEN)
        c0 = struct.unpack("<I", c0_path.read_bytes()[:4])[0]
        payload = drafts_path.read_bytes()
        drafts = list(struct.unpack("<%dI" % (len(payload) // 4), payload))
        steps.append((position, taps, c0, drafts))
    return sorted(steps, key=lambda entry: entry[0])


def reference_step(taps: np.ndarray, c0: int, base_position: int):
    """The reference's own forward plus selector tail for one step."""
    forwarded = rail.forward(ref.bf16(taps), c0, float(base_position))
    logits, hidden = forwarded[0], forwarded[1]
    selector = {
        name: ref.bf16_to_f32(ref.read_safetensors_tensor(ref.DRAFTER / "model.safetensors", name)).copy()
        for name in ("candidate_selector.predecessor_codebook",
                     "candidate_selector.successor_codebook",
                     "candidate_selector.hidden_projection.weight")
    }
    mask_logits = logits[1:]
    top_ids = np.argsort(-mask_logits, axis=-1, kind="stable")[:, :ref.SELECTOR_TOP_K]
    unary = np.take_along_axis(mask_logits, top_ids, axis=-1)
    gate = ref.bf16(hidden[1:] @ selector["candidate_selector.hidden_projection.weight"].T)
    edges = ref._score_edges(
        selector["candidate_selector.predecessor_codebook"],
        selector["candidate_selector.successor_codebook"],
        top_ids[None], unary[None], gate[None], np.array([c0]), ref.SELECTOR_TOP_K)[0]
    return ref._greedy_walk(edges, top_ids), top_ids[:, 0].tolist(), float(unary[:, 0].mean())


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    directory = Path(sys.argv[1])
    golden = None
    if len(sys.argv) > 2:
        golden = [int(line) for line in Path(sys.argv[2]).read_text().split() if line.strip()]
    steps = read_steps(directory)
    if not steps:
        raise SystemExit(f"no step_*_taps.bin dumps in {directory} "
                         f"(run the module with SPARK_QWEN36_DSPARK_DUMP_DIR set)")
    print(f"steps            = {len(steps)} from {directory}")
    print(f"{'pos':>6} {'c0':>7} {'module drafts':>34} {'ref drafts':>34} {'parity':>8} "
          f"{'distinct':>8} {'all==c0':>8} {'unary1':>8}")
    first_mismatch = None
    first_degenerate = None
    for position, taps, c0, drafts in steps:
        reference_drafts, reference_top1, unary_mean = reference_step(taps, c0, position)
        parity = "PARITY" if list(drafts) == list(reference_drafts) else "MISMATCH"
        distinct = len(set(drafts))
        repeat_c0 = all(draft == c0 for draft in drafts)
        degenerate = repeat_c0 or distinct <= 2
        if parity != "PARITY" and first_mismatch is None:
            first_mismatch = position
        if degenerate and first_degenerate is None:
            first_degenerate = position
        print(f"{position:>6} {c0:>7} {str(list(drafts)[:4]):>34} {str(list(reference_drafts)[:4]):>34} "
              f"{parity:>8} {distinct:>8} {str(repeat_c0):>8} {unary_mean:>8.2f}")
    print()
    print(f"first step whose drafts diverge from the reference : "
          f"{first_mismatch if first_mismatch is not None else 'none (the DFlash2 path is faithful)'}")
    print(f"first step whose tap state is degenerate           : "
          f"{first_degenerate if first_degenerate is not None else 'none'}")
    if golden is not None:
        print(f"golden tokens    = {len(golden)}")
        print("Compare the golden against the committed stream your run recorded; the bisect reads: tap state "
              "degenerate BEFORE the first token divergence -> a state restore/rollback bug; token divergence "
              "first with the taps following -> the commit/accounting path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
