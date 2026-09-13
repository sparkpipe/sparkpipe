#!/usr/bin/env python3
"""Context probe: is the drafter starved of a clean context, or just weak?

The acceptance profile answers "what does the drafter earn" and proves a perfectly
faithful forward earns the same. That leaves exactly one fork, and the decode-state
dumps already contain the data to settle it:

  CONTEXT   the spec lane's target state is corrupted, so BOTH the module and the
            oracle draft from a poisoned context and both accept nothing.
  QUALITY   the drafter checkpoint simply does not predict this prompt, and no state
            fix will move acceptance.

This tool separates them without a bench run, from two directories of per-position
decode-state dumps (SPARK_QWEN36_DECODE_STATE_DUMP_DIR):

    decode_<pos>_taps.bin     TAPS x H BF16 - the drafter's whole context
    decode_<pos>_hidden.bin   H BF16       - the target's final pre-head hidden

  DIVERGENCE  spec vs no-spec taps at every shared position: how many of the
              TAPS x H BF16 words differ, and the FIRST position where they do. That
              onset is comparable with the position where acceptance collapses.
  STREAM      the no-spec token at each position, recovered as argmax(hidden @
              lm_head^T) from the dumped hidden. No token file needed, and the
              recovered stream is self-consistent with the taps by construction.
  CLEAN       the drafter's acceptance from the NO-SPEC taps - a provably clean
              context - against that recovered stream. This is the drafter's true
              ceiling on this prompt: if it is high, the deployed acceptance loss was
              context corruption; if it is as low as the spec run's, the checkpoint is
              the cap and only draft count or prompt mix can move the bar.

usage: qwen36_dspark_context_probe.py NOSPEC_DIR [SPEC_DIR] [--positions N] [--stride N]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen36_dspark_reference as ref  # noqa: E402
import qwen36_dspark_e2e_parity as rail  # noqa: E402

_CACHE: dict = {}


def _cached(name, loader):
    def wrapper():
        if name not in _CACHE:
            _CACHE[name] = loader()
        return _CACHE[name]
    return wrapper


ref.load_drafter = _cached("drafter", ref.load_drafter)
ref.load_target_shared = _cached("target", ref.load_target_shared)

SELECTOR_NAMES = {"predecessor": "candidate_selector.predecessor_codebook",
                  "successor": "candidate_selector.successor_codebook",
                  "projection": "candidate_selector.hidden_projection.weight"}


def selector_weights():
    if "selector" not in _CACHE:
        _CACHE["selector"] = {
            key: ref.bf16_to_f32(ref.read_safetensors_tensor(
                ref.DRAFTER / "model.safetensors", name)).copy()
            for key, name in SELECTOR_NAMES.items()}
    return _CACHE["selector"]


def positions_of(directory: Path, suffix: str) -> list[int]:
    found = []
    for path in directory.glob(f"decode_*_{suffix}"):
        match = re.match(rf"decode_(\d+)_{re.escape(suffix)}$", path.name)
        if match is not None:
            found.append(int(match.group(1)))
    return sorted(found)


def read_taps(directory: Path, position: int):
    raw = np.fromfile(directory / f"decode_{position}_taps.bin", dtype=np.uint16)
    if raw.size != ref.TAPS * ref.HIDDEN:
        return None
    return raw.reshape(ref.TAPS, ref.HIDDEN)


def read_hidden(directory: Path, position: int):
    raw = np.fromfile(directory / f"decode_{position}_hidden.bin", dtype=np.uint16)
    if raw.size != ref.HIDDEN:
        return None
    return raw


def recovered_token(hidden_words: np.ndarray) -> int:
    """The token this hidden commits: argmax over the BF16-truncated head row, with the
    contract's id-ascending tie rule."""
    lm_head, _embed = ref.load_target_shared()
    logits = ref.bf16(ref.bf16_to_f32(hidden_words) @ lm_head.T)
    return int(np.argmax(logits))


def drafts_from(taps_words: np.ndarray, c0: int, position: int):
    """The oracle drafter's block from one context."""
    weights = selector_weights()
    taps = ref.bf16_to_f32(taps_words).reshape(ref.TAPS, ref.HIDDEN)
    logits, hidden = rail.forward(ref.bf16(taps), c0, float(position))[:2]
    rows = slice(1, ref.BLOCK)
    top_ids = np.argsort(-logits[rows], axis=-1, kind="stable")[:, :ref.SELECTOR_TOP_K]
    unary = np.take_along_axis(logits[rows], top_ids, axis=-1).astype(np.float32)
    gate = ref.bf16(hidden[rows] @ weights["projection"].T)
    edges = ref._score_edges(weights["predecessor"], weights["successor"], top_ids[None],
                             unary[None], gate[None], np.array([c0]), ref.SELECTOR_TOP_K)[0]
    return ref._greedy_walk(edges, top_ids)


def main() -> int:
    arguments = sys.argv[1:]
    sample = 20
    stride = None
    positional = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--positions":
            index += 1
            sample = int(arguments[index])
        elif argument == "--stride":
            index += 1
            stride = int(arguments[index])
        elif argument.startswith("--"):
            raise SystemExit(f"unknown flag {argument}")
        else:
            positional.append(argument)
        index += 1
    if not positional:
        raise SystemExit(__doc__)
    nospec = Path(positional[0])
    spec = Path(positional[1]) if len(positional) > 1 else None
    clean_positions = positions_of(nospec, "taps.bin")
    if not clean_positions:
        raise SystemExit(f"no decode_*_taps.bin in {nospec}")
    print(f"no-spec positions = {len(clean_positions)} "
          f"({clean_positions[0]}..{clean_positions[-1]}) from {nospec}")

    # 1) DIVERGENCE. No forward needed, so this runs over every shared position.
    if spec is not None:
        shared = [position for position in positions_of(spec, "taps.bin")
                  if position in set(clean_positions)]
        print(f"shared with spec  = {len(shared)} from {spec}")
        first_bad = None
        rows = []
        for position in shared:
            a = read_taps(spec, position)
            b = read_taps(nospec, position)
            if a is None or b is None:
                continue
            bad = int(np.count_nonzero(a != b))
            per_layer = [int(np.count_nonzero(a[layer] != b[layer])) for layer in range(ref.TAPS)]
            rows.append((position, bad, per_layer))
            if bad and first_bad is None:
                first_bad = position
        print()
        print(f"{'pos':>6} {'tapdiff':>8} {'of':>7}  per-tap-layer")
        for position, bad, per_layer in rows:
            print(f"{position:>6} {bad:>8} {ref.TAPS * ref.HIDDEN:>7}  {per_layer}")
        clean_count = sum(1 for _p, bad, _l in rows if bad == 0)
        print(f"spec taps identical to no-spec at {clean_count}/{len(rows)} shared positions; "
              f"first divergence at {first_bad if first_bad is not None else 'none'}")

    # 1b) TOKEN CROSS-CHECK. A tap difference at a position whose TOKEN HISTORY is
    # identical cannot be explained by the two runs having drifted apart: same inputs,
    # different state, i.e. corruption. So print where the two streams part company and
    # compare that with where the taps part company.
    if spec is not None:
        spec_stream, clean_stream = {}, {}
        for position in sorted(set(positions_of(spec, "hidden.bin"))
                               & set(positions_of(nospec, "hidden.bin"))):
            a = read_hidden(spec, position)
            b = read_hidden(nospec, position)
            if a is None or b is None:
                continue
            spec_stream[position] = recovered_token(a)
            clean_stream[position] = recovered_token(b)
        divergent = [position for position in sorted(spec_stream)
                     if spec_stream[position] != clean_stream[position]]
        agreeing = [position for position in sorted(spec_stream)
                    if spec_stream[position] == clean_stream[position]]
        print(f"recovered tokens compared at {len(spec_stream)} shared positions: "
              f"{len(agreeing)} agree, first token divergence at "
              f"{divergent[0] if divergent else 'none'}")
        if divergent and first_bad is not None and first_bad <= divergent[0]:
            print(f"  -> the TAPS diverge at {first_bad} while the committed tokens still "
                  f"agree through {divergent[0] - 1}: identical history, different state. "
                  f"That is state CORRUPTION, not two streams drifting apart.")

    # 2) STREAM. The no-spec run's own tokens, from its own hiddens.
    hidden_positions = [position for position in positions_of(nospec, "hidden.bin")]
    stream = {}
    for position in hidden_positions:
        words = read_hidden(nospec, position)
        if words is not None:
            # The hidden at position p commits the token AT p+1.
            # FRAME CONVENTION, and it is load-bearing: the decode frame at base
            # position q consumes the token at q and its final hidden PRODUCES the next
            # token - which is the token the drafter anchors on, exactly the module's
            # own output_token_ids[0]. So stream[q] is "what frame q produced", the
            # anchor for taps(q) is stream[q], and draft slot i predicts stream[q+1+i].
            # Keying this by "the token at position q" is an off-by-one that makes the
            # shipped assembly look 8x worse than an anchor-shifted variant; the
            # contract sweep caught it, which is what the sweep is for.
            stream[position] = recovered_token(words)
    print()
    print(f"recovered no-spec stream: {len(stream)} tokens for positions "
          f"{min(stream) if stream else '-'}..{max(stream) if stream else '-'}")

    # 3) CLEAN acceptance. One forward per sampled position.
    usable = [position for position in clean_positions
              if position in stream and all(position + 1 + slot in stream
                                            for slot in range(ref.BLOCK - 1))]
    if stride:
        chosen = usable[::stride][:sample]
    else:
        step = max(1, len(usable) // sample)
        chosen = usable[::step][:sample]
    print(f"sampling {len(chosen)} of {len(usable)} positions with a full 7-token horizon")
    print()
    print(f"{'pos':>6} {'c0':>7} {'accepted':>9}  drafts vs stream")
    accepted_all = []
    slot_hits = np.zeros(ref.BLOCK - 1, dtype=np.int64)
    for position in chosen:
        c0 = stream[position]
        drafts = drafts_from(read_taps(nospec, position), c0, position)
        accepted = 0
        for slot, draft in enumerate(drafts):
            if stream[position + 1 + slot] == draft:
                slot_hits[slot] += 1
                if accepted == slot:
                    accepted = slot + 1
        accepted_all.append(accepted)
        wanted = [stream[position + 1 + slot] for slot in range(len(drafts))]
        print(f"{position:>6} {c0:>7} {accepted:>9}  {drafts[:4]} vs {wanted[:4]}")
    if accepted_all:
        array = np.array(accepted_all, dtype=np.float64)
        print()
        print(f"CLEAN-CONTEXT acceptance : mean {array.mean():.3f}  median {np.median(array):.1f}  "
              f"max {array.max():.0f}  zero {int((array == 0).sum())}/{len(array)}")
        print("    per-slot accept rate : " +
              ", ".join(f"{slot}:{slot_hits[slot] / len(accepted_all):.2f}"
                        for slot in range(ref.BLOCK - 1)))
        print()
        print("VERDICT: compare this with the spec run's acceptance profile. A clean-context "
              "mean well above the deployed number means the loss was CONTEXT corruption "
              "(fix the state, not the drafter); a clean-context mean as low as the deployed "
              "one means the checkpoint is the cap and only draft count or prompt mix moves "
              "the bar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
