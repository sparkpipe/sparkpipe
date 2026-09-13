#!/usr/bin/env python3
"""Contract sweep: score plausible readings of the DFlash2 contract by ACCEPTANCE.

Three independent measurements put this drafter at ~0.8 accepted drafts per round:
the deployed module (0.853), a perfectly faithful numpy forward on the same taps
(0.853), and the same forward on a PROVABLY CLEAN no-spec context (0.800). The
published figure for this checkpoint is 4.80, and every constant in
tools/qwen36_dspark_reference.py matches the checkpoint's own dflash_config
(block_size 8, mask_token_id 248070, target_layer_ids 5/19/33/47/61, rank 256,
top_k 16). So the remaining candidate is not a constant and not a kernel: it is how
the context is ASSEMBLED - and a misreading there is invisible to every parity test,
because the numpy oracle and the CUDA module share it.

This tool scores variants of that assembly against the no-spec run's own token
stream, on a clean context, using acceptance as the only judge:

    baseline        as shipped: positions p..p+7, pos_ctx p-1, taps in dump order,
                    block row 0 = embed(committed token), rows 1..7 = embed(mask)
    taps-reversed   the five tap layers concatenated in the opposite order
    taps-rolled     the tap order rotated by one
    ctx-zero        NO context at all - the control that says whether the tap
                    pathway contributes anything
    pos-shift-up    draft positions p+1..p+8 with pos_ctx p
    ctx-pos-current pos_ctx = p instead of p-1
    anchor-mask     block row 0 masked too (no committed-token anchor)
    anchor-next     the anchor is the NEXT token instead of the committed one

Every variant is also scored at three draft/stream alignments (p+slot, p+1+slot,
p+2+slot), which costs nothing extra and catches an off-by-one in what a draft slot
is supposed to predict.

A variant that beats the baseline by more than the noise is the acceptance bug. A
sweep where nothing beats the baseline says the assembly is right and the gap is the
checkpoint or the prompt - which is a different escalation entirely.

usage: qwen36_dspark_contract_sweep.py NOSPEC_DIR [--positions N] [--variants a,b,c]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen36_dspark_reference as ref  # noqa: E402

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


def read_words(path: Path, count: int):
    raw = np.fromfile(path, dtype=np.uint16)
    return raw if raw.size == count else None


def recover_stream(directory: Path):
    """position -> committed token, from each dumped final hidden."""
    lm_head, _embed = ref.load_target_shared()
    stream = {}
    for position in positions_of(directory, "hidden.bin"):
        words = read_words(directory / f"decode_{position}_hidden.bin", ref.HIDDEN)
        if words is None:
            continue
        logits = ref.bf16(ref.bf16_to_f32(words) @ lm_head.T)
        # FRAME convention: stream[q] is the token frame q PRODUCED, which is what the
        # drafter anchors on for taps(q); draft slot i then predicts stream[q+1+i]. The
        # "token at position q" keying is an off-by-one worth 8x in apparent acceptance.
        stream[position] = int(np.argmax(logits))
    return stream


def forward_variant(taps: np.ndarray, c0: int, position: float, variant: str):
    """The reference forward with one assembly choice changed. Returns (logits, hidden)."""
    drafter = ref.load_drafter()
    _lm_head, embed_tokens = ref.load_target_shared()
    base = float(position)
    positions_q = np.arange(base, base + ref.BLOCK, dtype=np.float32)
    pos_ctx = base - 1.0
    ordered = taps
    if variant == "taps-reversed":
        ordered = taps[::-1].copy()
    elif variant == "taps-rolled":
        ordered = np.roll(taps, 1, axis=0).copy()
    elif variant == "pos-shift-up":
        positions_q = np.arange(base + 1.0, base + 1.0 + ref.BLOCK, dtype=np.float32)
        pos_ctx = base
    elif variant == "ctx-pos-current":
        pos_ctx = base
    ctx = ref.bf16(drafter["fc.weight"] @ ordered.reshape(-1))
    ctx = ref.bf16(ref.rms_norm(ctx, drafter["hidden_norm.weight"]))
    if variant == "ctx-zero":
        ctx = np.zeros_like(ctx)
    block = np.empty((ref.BLOCK, ref.HIDDEN), dtype=np.float32)
    block[0] = embed_tokens[ref.MASK_TOKEN_ID] if variant == "anchor-mask" else embed_tokens[c0]
    block[1:] = embed_tokens[ref.MASK_TOKEN_ID]
    x = block
    for layer in range(ref.N_LAYERS):
        lw = {name.split(f"layers.{layer}.")[1]: drafter[name]
              for name in drafter if f"layers.{layer}." in name}
        x = ref.forward_layer(x, ctx, lw, positions_q, pos_ctx)
    hidden = ref.bf16(ref.rms_norm(x, drafter["norm.weight"]))
    logits = (hidden @ ref.load_target_shared()[0].T).astype(np.float32)
    return ref.bf16_to_f32(ref.f32_to_bf16(logits)), hidden


def drafts_of(taps: np.ndarray, c0: int, position: int, variant: str):
    weights = selector_weights()
    logits, hidden = forward_variant(taps, c0, position, variant)
    rows = slice(1, ref.BLOCK)
    top_ids = np.argsort(-logits[rows], axis=-1, kind="stable")[:, :ref.SELECTOR_TOP_K]
    unary = np.take_along_axis(logits[rows], top_ids, axis=-1).astype(np.float32)
    gate = ref.bf16(hidden[rows] @ weights["projection"].T)
    edges = ref._score_edges(weights["predecessor"], weights["successor"], top_ids[None],
                             unary[None], gate[None], np.array([c0]), ref.SELECTOR_TOP_K)[0]
    return ref._greedy_walk(edges, top_ids)


def accepted_at(drafts, stream, position, offset):
    """Leading run of drafts matching the stream starting at position + offset."""
    matched = 0
    for slot, draft in enumerate(drafts):
        wanted = stream.get(position + offset + slot)
        if wanted is None or wanted != draft:
            break
        matched += 1
    return matched


VARIANTS = ("baseline", "taps-reversed", "taps-rolled", "ctx-zero", "pos-shift-up",
            "ctx-pos-current", "anchor-mask", "anchor-next")


def main() -> int:
    arguments = sys.argv[1:]
    sample = 8
    variants = list(VARIANTS)
    positional = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--positions":
            index += 1
            sample = int(arguments[index])
        elif argument == "--variants":
            index += 1
            variants = arguments[index].split(",")
        elif argument.startswith("--"):
            raise SystemExit(f"unknown flag {argument}")
        else:
            positional.append(argument)
        index += 1
    if not positional:
        raise SystemExit(__doc__)
    directory = Path(positional[0])
    stream = recover_stream(directory)
    available = [position for position in positions_of(directory, "taps.bin")
                 if position in stream
                 and all(position + 1 + slot in stream for slot in range(ref.BLOCK - 1))]
    if not available:
        raise SystemExit(f"no usable positions in {directory}")
    step = max(1, len(available) // sample)
    chosen = available[::step][:sample]
    print(f"stream tokens = {len(stream)}; scoring {len(chosen)} positions "
          f"{chosen[0]}..{chosen[-1]} from {directory}")
    print(f"variants      = {', '.join(variants)}")
    print()
    results = {variant: {offset: [] for offset in (0, 1, 2)} for variant in variants}
    for variant in variants:
        for position in chosen:
            words = read_words(directory / f"decode_{position}_taps.bin",
                               ref.TAPS * ref.HIDDEN)
            if words is None:
                continue
            taps = ref.bf16_to_f32(words).reshape(ref.TAPS, ref.HIDDEN)
            anchor = stream[position + 1] if variant == "anchor-next" else stream[position]
            drafts = drafts_of(taps, anchor, position, variant)
            for offset in (0, 1, 2):
                results[variant][offset].append(accepted_at(drafts, stream, position, offset))
        means = {offset: float(np.mean(results[variant][offset])) for offset in (0, 1, 2)}
        best = max(means, key=lambda offset: means[offset])
        print(f"{variant:>16} : acceptance at slot->position offset "
              f"+0 {means[0]:.3f}  +1 {means[1]:.3f}  +2 {means[2]:.3f}   "
              f"best +{best} = {means[best]:.3f}")
    print()
    baseline = float(np.mean(results[variants[0]][1])) if variants else 0.0
    ranked = sorted(((float(np.mean(results[variant][1])), variant) for variant in variants),
                    reverse=True)
    print(f"baseline (as shipped, offset +1) = {baseline:.3f}")
    for score, variant in ranked:
        marker = "  <-- BEATS BASELINE" if score > baseline + 1e-9 else ""
        print(f"  {variant:>16} {score:.3f}{marker}")
    if ranked and ranked[0][0] <= baseline + 1e-9:
        print("no variant beats the shipped assembly: the contract reading is right and the "
              "acceptance gap is the checkpoint or the prompt, not the wiring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
