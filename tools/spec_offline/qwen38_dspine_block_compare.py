#!/usr/bin/env python3
"""Compare the Qwen3.8-27B DSpine drafter module with the Python drafter block_draftd.py serves.

    qwen38_dspine_block_compare.py extract --recording FILE --rows N --out DIR [--sequence ID]
    qwen38_dspine_block_compare.py synthetic --first F --rows N --out DIR [--seed S]
    qwen38_dspine_block_compare.py compare --checkpoint DIR --target SHARD --out DIR [--precise]

`extract` cuts an unbroken run of positions F..N of one sequence out of a tap recording, as
dsv41_dspark_block_compare.py does, with the taps o1, o16, o31, o46 and o61. `synthetic` writes the same files from
random taps and tokens. `build/qwen38_dspine_block_probe` ingests them in frames of up to 2,048 rows and writes each
layer's key and value cache, the block's rows after the final norm (y.bin and the bridge's normed.bin), each row's
candidates and the drafts under a greedy rule and two sampling rules.
`compare` replays the frames through block_draftd.py's Context, checks the cache, then runs the Python drafter's
forward (DSpineDrafter.pre_attention, attend_window and post_attention, as DraftGraph does) on the module's own cache
and compares y, the normed rows, the candidates and the drafts. --precise lets cuBLAS reduce in fp32 only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from block_drafter import rms  # noqa: E402
from tapstream import frames  # noqa: E402

TAPS = ["o1", "o16", "o31", "o46", "o61"]
NAMES = {"embed": "model.language_model.embed_tokens.weight", "head": "lm_head.weight", "norm": "model.language_model.norm.weight"}
HIDDEN, BLOCK, DRAFTS, LANES, WINDOW, LAYERS, KV_HEADS, HEAD_DIM, COLUMNS = 5120, 16, 15, 16, 8176, 5, 8, 128, 8192
FRAME_ROWS = 2048
RULES = ((0, 0.0, 0.0, 0), (12345, 1.0, 0.95, 0), (777, np.float32(1.0) / np.float32(0.6), 1.0, 20))


def bf16_file(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int16)).view(torch.bfloat16).view(*shape)


def frame_key(frame):
    return frame["control_generation"], frame["request_id"], frame["sequence_id"]


def unbroken_owner(recording, wanted, sequence):
    runs = {}
    for _, frame in frames(recording):
        key, first = frame_key(frame), frame["first_position"]
        if sequence is not None and key[2] != sequence:
            continue
        start, end = runs.get(key, (first, first))
        if first > end:
            start = first
        runs[key] = (start, max(end, first + frame["rows"]) if first >= start else first + frame["rows"])
        if runs[key][0] < wanted - 1 and runs[key][1] >= wanted:
            return key, runs[key][0]
    return None, None


def extract(options):
    wanted = options.rows + 1
    owner, start = unbroken_owner(options.recording, wanted, options.sequence)
    if owner is None:
        raise SystemExit(f"{options.recording}: no sequence holds an unbroken run of positions up to {options.rows}")
    taps, tokens = None, np.zeros(wanted, dtype=np.uint32)
    with open(options.recording, "rb") as handle:
        for hello, frame in frames(options.recording):
            first = frame["first_position"]
            if frame_key(frame) != owner or first < start or first >= wanted:
                continue
            columns = [hello["entries"].index(name) for name in TAPS]
            if taps is None:
                taps = np.zeros((wanted, len(TAPS) * hello["hidden"]), dtype=np.uint16)
            count = min(frame["rows"], wanted - first)
            handle.seek(frame["offset"])
            data = np.frombuffer(handle.read(count * frame["row_bytes"]), dtype=np.uint16).reshape(count, len(hello["entries"]), hello["hidden"])
            taps[first:first + count] = data[:, columns].reshape(count, -1)
            tokens[first:first + count] = frame["tokens"][:count]
    write_case(options.out, start, taps[start:options.rows], tokens)
    print(f"wrote {options.rows - start} tap rows from position {start} of sequence {owner[2]} to {options.out}")
    return 0


def write_case(out, start, taps, tokens):
    out.mkdir(parents=True, exist_ok=True)
    np.array([start], dtype=np.uint64).tofile(out / "first.bin")
    taps.tofile(out / "taps.bin")
    tokens.tofile(out / "tokens.bin")


def synthetic(options):
    generator = torch.Generator().manual_seed(options.seed)
    taps = (torch.randn(options.rows - options.first, len(TAPS) * HIDDEN, generator=generator) * 2.0).to(torch.bfloat16)
    tokens = torch.randint(0, 200000, (options.rows + 1,), generator=generator).numpy().astype(np.uint32)
    write_case(options.out, options.first, taps.view(torch.int16).numpy().view(np.uint16), tokens)
    print(f"wrote {options.rows - options.first} random tap rows from position {options.first} to {options.out}")
    return 0


def relative(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def compare(options):
    from block_draftd import Context, dspark_rule  # noqa: F401
    from block_drafter import open_drafter
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not options.precise
    out = options.out
    tokens = np.fromfile(out / "tokens.bin", dtype=np.uint32)
    rows = tokens.shape[0] - 1
    start = int(np.fromfile(out / "first.bin", dtype=np.uint64)[0])
    taps = bf16_file(out / "taps.bin", (rows - start, len(TAPS) * HIDDEN))
    d = open_drafter(options.checkpoint, options.target, NAMES, "cuda")
    failures = []
    with torch.inference_mode():
        context = Context(d, WINDOW)
        for first in range(start, rows, FRAME_ROWS):
            count = min(FRAME_ROWS, rows - first)
            context.write(first, tokens[first:first + count], taps[first - start:first - start + count].to(d.device))
        module_positions = np.fromfile(out / "cache_positions.bin", dtype=np.int64)
        held = module_positions >= 0
        python_held = context.positions >= 0
        if not np.array_equal(held, python_held) or not np.array_equal(module_positions[held], context.positions[held]):
            failures.append("cache positions differ")
        slots = torch.from_numpy(np.flatnonzero(held)).to(d.device)
        for layer in range(LAYERS):
            keys = bf16_file(out / f"keys{layer}.bin", (KV_HEADS, COLUMNS, HEAD_DIM)).to(d.device)
            values = bf16_file(out / f"values{layer}.bin", (KV_HEADS, COLUMNS, HEAD_DIM)).to(d.device)
            key_equal = (keys[:, slots] == context.keys[layer][:, slots]).float().mean().item()
            value_equal = (values[:, slots] == context.values[layer][:, slots]).float().mean().item()
            key_gap, value_gap = relative(keys[:, slots], context.keys[layer][:, slots]), relative(values[:, slots], context.values[layer][:, slots])
            print(f"layer {layer} cache: keys equal {100 * key_equal:.3f}% (relative {relative(keys[:, slots], context.keys[layer][:, slots]):.5f}),"
                  f" values equal {100 * value_equal:.3f}% (relative {relative(values[:, slots], context.values[layer][:, slots]):.5f})")
            if options.precise and max(key_gap, value_gap) > 1e-3:
                failures.append(f"layer {layer} cache differs")
            context.keys[layer].copy_(keys)
            context.values[layer].copy_(values)
        positions_window = np.arange(max(0, rows - WINDOW), rows)
        held_window = context.held(positions_window)
        if not held_window.all():
            positions_window = positions_window[int(np.flatnonzero(~held_window)[-1]) + 1:]
        begin, block_at, valid = context.span(positions_window)
        mask = None if valid is None else torch.from_numpy(valid).to(d.device).view(1, 1, 1, -1)
        anchor = int(tokens[rows])
        d.anchor = torch.tensor(anchor, device=d.device)
        block_tokens = torch.full((BLOCK,), d.mask, dtype=torch.long, device=d.device)
        block_tokens[0] = anchor
        block_positions = torch.arange(rows, rows + BLOCK, device=d.device)
        h = d.embed.weight[block_tokens].float()
        for index, layer in enumerate(d.layers):
            q, kb, vb = d.pre_attention(layer, h, block_positions)
            after = d.post_attention(layer, h, d.attend_window(index, q, kb, vb, context.keys, context.values, begin, block_at, mask))
            module_after = bf16_file(out / f"hidden{index + 1}.bin", (BLOCK, HIDDEN)).to(d.device)
            gap = relative(module_after, after)
            print(f"layer {index} from the module's input: output relative {gap:.5f}, equal {100 * (module_after == after).float().mean().item():.2f}%")
            if options.precise and gap > 0.01:
                failures.append(f"layer {index} differs from the Python layer on the same input")
            h = module_after
        y = rms(h, d.norm, d.eps).to(torch.bfloat16)
        base = (y @ d.head.t()).float()
        module_y = bf16_file(out / "y.bin", (BLOCK, HIDDEN)).to(d.device)
        module_normed = bf16_file(out / "normed.bin", (BLOCK, HIDDEN)).to(d.device)
        module_candidates = torch.from_numpy(np.fromfile(out / "candidates.bin", dtype=np.uint32).astype(np.int64).reshape(DRAFTS, LANES)).to(d.device)
        candidates = d.candidates(base)
        same_candidates = int((candidates == module_candidates).all(-1).sum())
        print(f"forward at {rows}: y relative {relative(module_y, y):.5f}, normed relative {relative(module_normed, d.stash['normed']):.5f};"
              f" candidate rows equal {same_candidates} of {DRAFTS}")
        drafted = np.fromfile(out / "drafts.bin", dtype=np.uint32).reshape(len(RULES), 2 * DRAFTS + 1)
        for index, rule in enumerate(RULES):
            sampling = None
            if rule[1] > 0.0:
                seed = rule[0] - (1 << 64) if rule[0] >= (1 << 63) else rule[0]
                sampling = (torch.tensor(seed, device=d.device), torch.tensor(np.float32(rule[1]), device=d.device),
                            torch.tensor(np.float32(rule[2] if rule[2] < 1.0 else 0.0), device=d.device), torch.tensor(rule[3], device=d.device))
            python_tokens, python_scores = d.walk(y, base, torch.tensor(anchor, device=d.device), BLOCK, sampling,
                                                  torch.tensor(rows, device=d.device), torch.tensor(sampling is not None, device=d.device))
            module_tokens = drafted[index, 1:1 + DRAFTS].tolist()
            same = sum(1 for a, b in zip(module_tokens, python_tokens.tolist()) if a == b)
            print(f"rule {index}: module drafts {module_tokens} python {python_tokens.tolist()} ({same} of {DRAFTS} equal)")
    print("PASS" if not failures else "FAIL " + "; ".join(failures))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    cut = commands.add_parser("extract")
    cut.add_argument("--recording", required=True)
    cut.add_argument("--rows", type=int, required=True)
    cut.add_argument("--out", type=Path, required=True)
    cut.add_argument("--sequence", type=int)
    random_case = commands.add_parser("synthetic")
    random_case.add_argument("--first", type=int, required=True)
    random_case.add_argument("--rows", type=int, required=True)
    random_case.add_argument("--out", type=Path, required=True)
    random_case.add_argument("--seed", type=int, default=7)
    check = commands.add_parser("compare")
    check.add_argument("--checkpoint", required=True)
    check.add_argument("--target", required=True)
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--precise", action="store_true")
    options = parser.parse_args()
    return {"extract": extract, "synthetic": synthetic, "compare": compare}[options.command](options)


if __name__ == "__main__":
    sys.exit(main())
