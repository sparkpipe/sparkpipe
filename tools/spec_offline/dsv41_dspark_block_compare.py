#!/usr/bin/env python3
"""Compare the DeepSeek V4.1 Flash DSpark drafter module with the Python drafter block_draftd.py serves.

    dsv41_dspark_block_compare.py extract --recording FILE --rows N --out DIR [--sequence ID]
    dsv41_dspark_block_compare.py compare --checkpoint DIR --out DIR

`extract` finds the first sequence of a tap stream recording (or the one --sequence names), from one engine session and
request, whose frames hold an unbroken run of positions F..N, and writes first.bin (F, u64), taps.bin (the N - F rows of the taps o36, o37 and
o38 from F on, bf16) and tokens.bin (N + 1 tokens by position, zero below F, u32). Recordings start past a cached
prefix, so F is rarely 0. `build/dsv41_dspark_block_probe` then ingests the taps from F in frames of up to 2,048 rows
and writes ring.bin and
ring_positions.bin (the module's window keys and their positions), hidden.bin and logits.bin (the block forward at
position N with token N as the anchor) and drafts.bin (`propose` under a greedy rule and two sampling rules).
`compare` checks, with --precise (cuBLAS reductions in fp32, as the module's kernels and the publisher's accumulate;
block_draftd.py runs with PyTorch's default, which lets cuBLAS reduce some bf16 products in bf16):
  - the window positions equal block_draftd.py's RingContext fed the same frames (dspark_fast.DsparkFast), and at least
    99% of the key elements are equal;
  - each stage, run by DsparkFast from the module's own input streams and mix (streams0..3.bin, pre0..3.bin), lands
    within 2% (relative) of the module's output streams;
  - the drafts and confidences equal DsparkFast's heads run on the module's hidden rows and logits, for each rule.
It also reports the whole forward on the module's keys against DsparkFast's and the eager reference's: the hidden
rows' relative distance and each row's top token. The drafter is chaotic (a routing choice that flips on a near tie
moves a row by percent), so the whole forward is reported, not gated.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from block_drafter import coupled_choice  # noqa: E402
from dspark_drafter import rms_norm  # noqa: E402
from tapstream import frames  # noqa: E402

TAPS = ["o36", "o37", "o38"]
BLOCK, HIDDEN, VOCAB, WINDOW, STAGES, HEAD_DIM = 5, 5120, 129280, 128, 3, 512
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
        raise SystemExit(f"{options.recording}: no sequence holds an unbroken run of positions up to {options.rows} from one session and request")
    taps = None
    tokens = np.zeros(wanted, dtype=np.uint32)
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
    options.out.mkdir(parents=True, exist_ok=True)
    np.array([start], dtype=np.uint64).tofile(options.out / "first.bin")
    taps[start:options.rows].tofile(options.out / "taps.bin")
    tokens.tofile(options.out / "tokens.bin")
    print(f"wrote {options.rows - start} tap rows from position {start} and {wanted} tokens of sequence {owner[2]} to {options.out}")
    return 0


def relative(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def walk(fast, hidden, logits, anchor, position, rule):
    previous = torch.tensor(anchor, device=fast.device)
    tokens, embeddings = [], []
    for slot in range(BLOCK):
        embedding = torch.index_select(fast.markov_embed, 0, previous.view(1)).view(-1)
        embeddings.append(embedding)
        row = logits[slot] + torch.mm(embedding.view(1, -1), fast.markov16.t(), out_dtype=torch.float32).view(-1)
        previous = coupled_choice(row, rule, position + slot) if rule is not None else torch.argmax(row)
        tokens.append(int(previous))
    features = torch.cat((hidden, torch.stack(embeddings)), -1).to(torch.float32)
    return tokens, torch.sigmoid(F.linear(features, fast.confidence).squeeze(-1)).tolist()


def compare(options):
    from block_draftd import RingContext, dspark_rule
    from dspark_fast import DsparkFast
    out = options.out
    tokens = np.fromfile(out / "tokens.bin", dtype=np.uint32)
    rows = tokens.shape[0] - 1
    start = int(np.fromfile(out / "first.bin", dtype=np.uint64)[0])
    taps = bf16_file(out / "taps.bin", (rows - start, len(TAPS) * HIDDEN))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not options.precise
    fast = DsparkFast(options.checkpoint, "cuda", keep_reference=True, compiled=False)
    device = fast.device
    failures = []
    with torch.inference_mode():
        context = RingContext(fast)
        for first in range(start, rows, FRAME_ROWS):
            count = min(FRAME_ROWS, rows - first)
            context.write(first, tokens[first:first + count], taps[first - start:first - start + count].to(device))
        module_keys = bf16_file(out / "ring.bin", (STAGES, WINDOW, HEAD_DIM)).to(device)
        module_positions = np.fromfile(out / "ring_positions.bin", dtype=np.int64)
        if not np.array_equal(module_positions, context.positions):
            failures.append("window positions differ")
        held = torch.from_numpy(module_positions >= 0).to(device)
        python_keys = context.ring.keys
        equal = (module_keys == python_keys)[:, held].float().mean().item() if bool(held.any()) else 1.0
        if options.precise and equal < 0.99:
            failures.append("fewer than 99% of the window key elements are equal")
        print(f"window keys: {int(held.sum())} positions held; module and RingContext equal on {100 * equal:.3f}% of key elements;"
              f" relative distance per stage {[round(relative(module_keys[s][held], python_keys[s][held]), 6) for s in range(STAGES)] if bool(held.any()) else []}")
        anchor = int(tokens[rows])
        module_hidden = bf16_file(out / "hidden.bin", (BLOCK, HIDDEN)).to(device)
        module_logits = torch.from_numpy(np.fromfile(out / "logits.bin", dtype=np.float32).reshape(BLOCK, VOCAB)).to(device)
        fast.static_keys.copy_(module_keys)
        fast.static_positions.copy_(torch.from_numpy(module_positions).to(device))
        fast.static_token.fill_(anchor)
        fast.static_position.fill_(rows)
        fast_hidden, fast_logits = fast.forward_fast()
        eager_context = fast.context()
        eager_context.keys.copy_(module_keys)
        eager_context.positions.copy_(torch.from_numpy(module_positions).to(device))
        eager_hidden, eager_logits = fast.forward_reference(eager_context, anchor, rows)
        positions = torch.arange(rows, rows + BLOCK, device=device)
        known = fast.static_positions
        valid = (known >= 0) & (known < rows) & (known >= rows - WINDOW)
        for stage, s in enumerate(fast.stages):
            h = bf16_file(out / f"streams{stage}.bin", (BLOCK, fast.hc, HIDDEN)).to(device)
            pre = torch.from_numpy(np.fromfile(out / f"pre{stage}.bin", dtype=np.float32).reshape(BLOCK, fast.hc)).to(device)
            following = bf16_file(out / f"streams{stage + 1}.bin", (BLOCK, fast.hc, HIDDEN)).to(device)
            following_pre = torch.from_numpy(np.fromfile(out / f"pre{stage + 1}.bin", dtype=np.float32).reshape(BLOCK, fast.hc)).to(device)
            residual = h
            attn_pre, attn_post, attn_comb = fast.hc_mixes_fast(h, s, "attn")
            x = rms_norm(fast.hc_pre(h, pre), s["attn_norm"], fast.eps)
            x = fast.attention_fast(s, x, positions, fast.static_keys[stage], valid)
            h = fast.hc_post(x, residual, attn_post, attn_comb)
            residual = h
            ffn_pre, ffn_post, ffn_comb = fast.hc_mixes_fast(h, s, "ffn")
            x = rms_norm(fast.hc_pre(h, attn_pre), s["ffn_norm"], fast.eps)
            x = fast.moe_fast(s, x)
            h = fast.hc_post(x, residual, ffn_post, ffn_comb)
            stage_gap = relative(following, h)
            print(f"stage {stage} from the module's input: output streams relative distance {stage_gap:.5f}, next pre largest difference"
                  f" {float((following_pre - ffn_pre).abs().max()):.2e}")
            if options.precise and stage_gap > 0.02:
                failures.append(f"stage {stage} output differs from DsparkFast's on the same input")
        top = [module_logits.argmax(-1).tolist(), fast_logits.argmax(-1).tolist(), eager_logits.argmax(-1).tolist()]
        print(f"forward at {rows} on the module's keys: hidden relative distance module-fast {relative(module_hidden, fast_hidden):.5f}"
              f" module-eager {relative(module_hidden, eager_hidden):.5f} fast-eager {relative(fast_hidden, eager_hidden):.5f};"
              f" top tokens module {top[0]} fast {top[1]} eager {top[2]}")
        drafted = np.fromfile(out / "drafts.bin", dtype=np.uint32).reshape(len(RULES), 2 * BLOCK + 1)
        for index, rule in enumerate(RULES):
            python_tokens, python_scores = walk(fast, module_hidden, module_logits, anchor, rows, dspark_rule(fast, rule))
            module_tokens = drafted[index, 1:1 + BLOCK].tolist()
            module_scores = drafted[index, 1 + BLOCK:].view(np.float32).tolist()
            gap = max(abs(a - b) for a, b in zip(module_scores, python_scores))
            same = sum(1 for a, b in zip(module_tokens, python_tokens) if a == b)
            print(f"rule {index} {rule}: module drafts {module_tokens} python {python_tokens} ({same} of {BLOCK} equal), confidence gap {gap:.2e}")
            if module_tokens != python_tokens or gap > 1e-4:
                failures.append(f"rule {index} drafts or confidences differ")
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
    check = commands.add_parser("compare")
    check.add_argument("--checkpoint", required=True)
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--precise", action="store_true")
    options = parser.parse_args()
    return extract(options) if options.command == "extract" else compare(options)


if __name__ == "__main__":
    sys.exit(main())
