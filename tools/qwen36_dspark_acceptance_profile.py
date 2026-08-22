#!/usr/bin/env python3
"""Acceptance profile: what the drafter ACTUALLY earns per round, and its ceiling.

The race bar is average acceptance, not oracle parity, and the per-step dumps plus a
no-spec golden stream already contain the answer - no bench harness needed.

Three numbers come out of one dump directory:

  REALIZED   the position ladder IS the acceptance record. Each decode frame dumps its
             base position, and the next frame's base position is this one's plus the
             committed token plus every accepted draft, so
                 accepted(round) = position(next) - position(this) - 1
             with no reliance on any log line.
  PROJECTED  the leading run of the module's drafts that matches the golden stream at
             that position. The run must be lossless for this to be meaningful (it is
             checked: every step's c0 must equal the golden token at its position), and
             then the target's greedy token IS the golden token, so this is exactly what
             the verifier accepted - a cross-check on REALIZED that also survives a
             truncated ladder.
  CEILING    with --oracle, the same projection for the ORACLE's drafts recomputed from
             the same taps. This is what a perfectly faithful drafter forward would have
             earned on this prompt, so the gap to the bar splits in two:
                 CEILING - PROJECTED  what forward exactness is worth
                 bar     - CEILING    what no forward fix can buy (drafter quality,
                                      draft count, prompt mix)

The per-slot accept rate says where the drafter dies (slot 0 rarely, the tail often),
which is what decides whether a wider block or a better checkpoint is the lever.

usage: qwen36_dspark_acceptance_profile.py DUMP_DIR GOLDEN_TOKENS [--first-position N]
                                           [--oracle] [--bar 6.0]
"""
from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen36_dspark_reference as ref  # noqa: E402


def read_ladder(directory: Path):
    """(position, c0, drafts) per dumped decode frame, position-ordered."""
    steps = []
    for path in directory.glob("step_*_drafts.bin"):
        match = re.match(r"step_(\d+)_drafts\.bin$", path.name)
        if match is None:
            continue
        position = int(match.group(1))
        c0_path = directory / f"step_{position}_c0.bin"
        if not c0_path.exists():
            continue
        payload = path.read_bytes()
        steps.append((position,
                      struct.unpack("<I", c0_path.read_bytes()[:4])[0],
                      list(struct.unpack("<%dI" % (len(payload) // 4), payload))))
    return sorted(steps, key=lambda entry: entry[0])


def prefix_match(drafts, golden, index):
    """How many leading drafts match the golden continuation after golden[index]."""
    matched = 0
    for offset, draft in enumerate(drafts):
        position = index + 1 + offset
        if position >= len(golden) or golden[position] != draft:
            break
        matched += 1
    return matched


def oracle_drafts(position: int, directory: Path, c0: int, count: int):
    """The oracle's drafts for one step, recomputed from that step's taps."""
    import qwen36_dspark_e2e_parity as rail
    weights = _oracle_weights()
    raw = np.fromfile(directory / f"step_{position}_taps.bin", dtype=np.uint16)
    taps = ref.bf16_to_f32(raw).reshape(ref.TAPS, ref.HIDDEN)
    logits, hidden = rail.forward(ref.bf16(taps), c0, float(position))[:2]
    rows = slice(1, 1 + count)
    top_ids = np.argsort(-logits[rows], axis=-1, kind="stable")[:, :ref.SELECTOR_TOP_K]
    unary = np.take_along_axis(logits[rows], top_ids, axis=-1).astype(np.float32)
    gate = ref.bf16(hidden[rows] @ weights["projection"].T)
    edges = ref._score_edges(weights["predecessor"], weights["successor"], top_ids[None],
                             unary[None], gate[None], np.array([c0]), ref.SELECTOR_TOP_K)[0]
    return ref._greedy_walk(edges, top_ids)


_CACHE: dict = {}


def _oracle_weights():
    if "selector" not in _CACHE:
        import qwen36_dspark_e2e_parity  # noqa: F401  (shares ref's loaders)
        names = {"predecessor": "candidate_selector.predecessor_codebook",
                 "successor": "candidate_selector.successor_codebook",
                 "projection": "candidate_selector.hidden_projection.weight"}
        _CACHE["selector"] = {
            key: ref.bf16_to_f32(ref.read_safetensors_tensor(
                ref.DRAFTER / "model.safetensors", name)).copy()
            for key, name in names.items()}
        loader = ref.load_drafter
        cache = {}

        def cached():
            if "w" not in cache:
                cache["w"] = loader()
            return cache["w"]
        ref.load_drafter = cached
        target_loader = ref.load_target_shared
        target_cache = {}

        def cached_target():
            if "w" not in target_cache:
                target_cache["w"] = target_loader()
            return target_cache["w"]
        ref.load_target_shared = cached_target
    return _CACHE["selector"]


def main() -> int:
    arguments = sys.argv[1:]
    use_oracle = "--oracle" in arguments
    bar = 6.0
    first_position = None
    positional = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--oracle":
            pass
        elif argument == "--bar":
            index += 1
            bar = float(arguments[index])
        elif argument == "--first-position":
            index += 1
            first_position = int(arguments[index])
        elif argument.startswith("--"):
            raise SystemExit(f"unknown flag {argument}")
        else:
            positional.append(argument)
        index += 1
    if len(positional) < 2:
        raise SystemExit(__doc__)
    directory = Path(positional[0])
    golden = [int(token) for token in Path(positional[1]).read_text().split() if token.strip()]
    steps = read_ladder(directory)
    if not steps:
        raise SystemExit(f"no step_*_drafts.bin dumps in {directory}")
    if first_position is None:
        first_position = steps[0][0]
    print(f"steps            = {len(steps)} from {directory}")
    print(f"golden tokens    = {len(golden)}, golden[0] is absolute position {first_position}")

    # LOSSLESSNESS CHECK. Every dumped step's committed anchor must be the golden token
    # at that position, or the projection below is measuring a diverged stream.
    aligned = mismatched = 0
    for position, c0, _drafts in steps:
        index = position - first_position
        if 0 <= index < len(golden):
            aligned += 1
            mismatched += int(golden[index] != c0)
    print(f"anchor vs golden : {aligned - mismatched}/{aligned} aligned"
          + ("" if not mismatched else f"  WARNING {mismatched} steps diverged - "
             "the projection below is not an acceptance measurement"))

    block = len(steps[0][2]) + 1
    print()
    print(f"{'pos':>6} {'realized':>9} {'projected':>10} {'oracle':>7} {'drafts':>28}")
    realized_all, projected_all, oracle_all = [], [], []
    slot_hits = np.zeros(block - 1, dtype=np.int64)
    slot_total = 0
    for order, (position, c0, drafts) in enumerate(steps):
        index = position - first_position
        if index < 0 or index >= len(golden):
            continue
        realized = None
        if order + 1 < len(steps):
            realized = steps[order + 1][0] - position - 1
        projected = prefix_match(drafts, golden, index)
        oracle = None
        if use_oracle:
            oracle = prefix_match(oracle_drafts(position, directory, c0, len(drafts)),
                                  golden, index)
            oracle_all.append(oracle)
        if realized is not None:
            realized_all.append(realized)
        projected_all.append(projected)
        slot_total += 1
        for slot, draft in enumerate(drafts):
            if index + 1 + slot < len(golden) and golden[index + 1 + slot] == draft:
                slot_hits[slot] += 1
        print(f"{position:>6} {('-' if realized is None else realized):>9} {projected:>10} "
              f"{('-' if oracle is None else oracle):>7} {str(drafts[:4]):>28}")

    def summary(name, values):
        if not values:
            return
        array = np.array(values, dtype=np.float64)
        print(f"{name:>26} : mean {array.mean():.3f}  median {np.median(array):.1f}  "
              f"min {array.min():.0f}  max {array.max():.0f}  "
              f"full-block {int((array == block - 1).sum())}/{len(array)}  "
              f"zero {int((array == 0).sum())}/{len(array)}")

    print()
    print(f"block size {block} -> at most {block - 1} accepted drafts per round")
    summary("REALIZED (position ladder)", realized_all)
    summary("PROJECTED (module drafts)", projected_all)
    summary("CEILING (oracle drafts)", oracle_all)
    if slot_total:
        rates = ", ".join(f"{slot}:{slot_hits[slot] / slot_total:.2f}"
                          for slot in range(block - 1))
        print(f"    per-slot accept rate : {rates}  (unconditional, over {slot_total} steps)")
    # CROSS-CHECK of the two accountings instead of asserting a convention: a round
    # commits this frame's own token, then every accepted draft, then the target's bonus
    # token at the first rejection, so the ladder advance should be projected + 2. How
    # often that identity holds is printed rather than assumed (the acceptance clamp
    # legitimately breaks it at a full-block round).
    if realized_all and len(realized_all) <= len(projected_all):
        agree = sum(1 for realized, projected in zip(realized_all, projected_all)
                    if realized - 2 == projected)
        print(f"    ladder identity advance == projected + 2 : {agree}/{len(realized_all)} rounds "
              f"(own token + accepted drafts + bonus token)")
    # ONSET. A drafter that is merely weak accepts poorly but at a roughly constant rate;
    # a drafter whose CONTEXT gets corrupted mid-stream stops accepting entirely from
    # some position on. Printing the last suffix of consecutive zero-acceptance rounds
    # separates those two, and the position where it starts is directly comparable with
    # the state-divergence position the decode-state dumps report.
    zero_tail = 0
    for projected in reversed(projected_all):
        if projected != 0:
            break
        zero_tail += 1
    if zero_tail:
        tail_start = [position for position, _c0, _drafts in steps][len(projected_all) - zero_tail]
        head = projected_all[:len(projected_all) - zero_tail]
        print(f"    zero-acceptance suffix : {zero_tail}/{len(projected_all)} rounds, "
              f"unbroken from position {tail_start} to the end"
              + (f"; before it the mean is {float(np.mean(head)):.3f}" if head else ""))
    if realized_all:
        realized_mean = float(np.mean(realized_all))
        print()
        accepted_mean = float(np.mean(projected_all))
        print(f"BAR {bar:.2f}: accepted drafts per round {accepted_mean:.3f} "
              f"(ladder advance {realized_mean:.3f} tokens/round), short by "
              f"{bar - accepted_mean:.3f}")
        if oracle_all:
            ceiling_mean = float(np.mean(oracle_all))
            projected_mean = float(np.mean(projected_all))
            print(f"  a perfectly faithful forward is worth "
                  f"{ceiling_mean - projected_mean:+.3f} on this prompt "
                  f"(oracle {ceiling_mean:.3f} vs module {projected_mean:.3f})")
            print(f"  no forward fix can buy the remaining {bar - ceiling_mean:.3f} - that is "
                  f"drafter quality, draft count or prompt mix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
