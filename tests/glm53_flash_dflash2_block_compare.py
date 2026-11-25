#!/usr/bin/env python3
"""PyTorch reference and parity comparator for GLM-5.3-Flash DFlash2 drafter module.

    glm53_flash_dflash2_block_compare.py synthetic --first F --rows N --out DIR [--seed S]
    glm53_flash_dflash2_block_compare.py compare --checkpoint DIR --embed-head-shard SHARD --norm-shard SHARD --out DIR [--precise]

Generates synthetic taps and tokens (including sliding-window boundary and wrap-around cases
past 2048), then compares build/modules/glm53_flash_dflash2_draft_stage.so outputs dumped by
build/glm53_flash_dflash2_block_probe directly against tools/spec_offline/selector_block_drafter.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "spec_offline"))
import block_drafter  # noqa: E402
import selector_block_drafter  # noqa: E402

HIDDEN = 4096
TAPS = 5
LAYERS = 5
HEADS = 32
KV_HEADS = 8
GROUP = HEADS // KV_HEADS
HEAD_DIM = 128
ATTN_SCALE = HEAD_DIM ** -0.5
BLOCK = 8
DRAFTS = BLOCK - 1
SLIDING = 2048
WINDOW = 2048
COLUMNS = WINDOW + BLOCK
FRAME_ROWS = 512
RULES = ((0, 0.0, 0.0, 0), (12345, 1.0, 0.95, 0), (777, np.float32(1.0) / np.float32(0.6), 1.0, 8))


class MultiSafeTensors:
    def __init__(self, paths):
        self.shards = [block_drafter.SafeTensors(p) for p in (paths if isinstance(paths, (list, tuple)) else [paths])]

    def load(self, name, device):
        for shard in self.shards:
            if name in shard.header:
                return shard.load(name, device)
        raise KeyError(name)


def open_selector(checkpoint_dir, embed_head_shard, norm_shard, device="cuda"):
    orig = selector_block_drafter.SafeTensors
    selector_block_drafter.SafeTensors = MultiSafeTensors
    try:
        return selector_block_drafter.SelectorBlockDrafter(
            checkpoint_dir,
            (embed_head_shard, norm_shard),
            {
                "embed": "model.language_model.embed_tokens.weight",
                "head": "lm_head.weight",
                "norm": "model.language_model.norm.weight",
            },
            device,
        )
    finally:
        selector_block_drafter.SafeTensors = orig


def bf16_file(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).astype(np.int16)).view(torch.bfloat16).view(*shape)


def f32_file(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).copy()).view(*shape)


def relative(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def write_case(out, start, taps, tokens):
    out.mkdir(parents=True, exist_ok=True)
    np.array([start], dtype=np.uint64).tofile(out / "first.bin")
    taps.tofile(out / "taps.bin")
    tokens.tofile(out / "tokens.bin")


def synthetic(options):
    gen = torch.Generator().manual_seed(options.seed)
    taps = (torch.randn(options.rows - options.first, TAPS * HIDDEN, generator=gen) * 1.5).to(torch.bfloat16)
    tokens = torch.randint(0, 150000, (options.rows + 1,), generator=gen).numpy().astype(np.uint32)
    write_case(options.out, options.first, taps.view(torch.int16).numpy().view(np.uint16), tokens)
    print(f"wrote {options.rows - options.first} random tap rows from position {options.first} to {options.out}")
    return 0


def exact_gqa_attention(q, k, v, visible):
    k_all = k.repeat_interleave(GROUP, dim=0).float()
    v_all = v.repeat_interleave(GROUP, dim=0).float()
    scores = torch.matmul(q.float(), k_all.transpose(1, 2)) * ATTN_SCALE
    scores = scores.masked_fill(~visible[None, :, :], float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v_all).transpose(0, 1).reshape(BLOCK, -1).to(torch.bfloat16)


def compare(options):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not options.precise
    out = options.out
    tokens = np.fromfile(out / "tokens.bin", dtype=np.uint32)
    rows = tokens.shape[0] - 1
    start = int(np.fromfile(out / "first.bin", dtype=np.uint64)[0])
    taps = bf16_file(out / "taps.bin", (rows - start, TAPS * HIDDEN))
    ref = open_selector(options.checkpoint, options.embed_head_shard, options.norm_shard, "cuda")
    failures = []

    with torch.inference_mode():
        ref_keys = [torch.zeros((KV_HEADS, COLUMNS, HEAD_DIM), dtype=torch.bfloat16, device=ref.device) for _ in range(LAYERS)]
        ref_vals = [torch.zeros((KV_HEADS, COLUMNS, HEAD_DIM), dtype=torch.bfloat16, device=ref.device) for _ in range(LAYERS)]
        ref_pos = np.full(WINDOW, -1, dtype=np.int64)
        for first in range(start, rows, FRAME_ROWS):
            count = min(FRAME_ROWS, rows - first)
            skip = max(0, count - WINDOW)
            f0 = first + skip
            c0 = count - skip
            pos = np.arange(f0, f0 + c0)
            slots = pos % WINDOW
            t_chunk = taps[first - start + skip:first - start + count].to(ref.device)
            p_chunk = torch.from_numpy(pos).to(ref.device)
            ks, vs = ref.context(t_chunk, p_chunk)
            slot_idx = torch.from_numpy(slots).to(ref.device)
            for layer in range(LAYERS):
                ref_keys[layer].index_copy_(1, slot_idx, ks[layer].transpose(0, 1))
                ref_vals[layer].index_copy_(1, slot_idx, vs[layer].transpose(0, 1))
            ref_pos[ref_pos >= f0] = -1
            ref_pos[slots] = pos

        mod_pos = np.fromfile(out / "cache_positions.bin", dtype=np.int64)
        held = mod_pos >= 0
        if not np.array_equal(held, ref_pos >= 0) or not np.array_equal(mod_pos[held], ref_pos[held]):
            failures.append("cache positions differ")

        held_slots = torch.from_numpy(np.flatnonzero(held)).to(ref.device)
        mod_keys, mod_vals = [], []
        for layer in range(LAYERS):
            mk = bf16_file(out / f"keys{layer}.bin", (KV_HEADS, COLUMNS, HEAD_DIM)).to(ref.device)
            mv = bf16_file(out / f"values{layer}.bin", (KV_HEADS, COLUMNS, HEAD_DIM)).to(ref.device)
            k_gap = relative(mk[:, held_slots], ref_keys[layer][:, held_slots])
            v_gap = relative(mv[:, held_slots], ref_vals[layer][:, held_slots])
            k_eq = (mk[:, held_slots] == ref_keys[layer][:, held_slots]).float().mean().item()
            v_eq = (mv[:, held_slots] == ref_vals[layer][:, held_slots]).float().mean().item()
            print(f"layer {layer} cache: keys eq {100 * k_eq:.2f}% (rel {k_gap:.5f}), values eq {100 * v_eq:.2f}% (rel {v_gap:.5f})")
            if max(k_gap, v_gap) > 2e-3:
                failures.append(f"layer {layer} cache rel diff too large: k={k_gap:.5f} v={v_gap:.5f}")
            mod_keys.append(mk)
            mod_vals.append(mv)

        win_first = max(start, rows - (SLIDING - 1))
        win_positions = np.arange(win_first, rows)
        held_mask = ref_pos[win_positions % WINDOW] == win_positions
        if not held_mask.all():
            win_positions = win_positions[int(np.flatnonzero(~held_mask)[-1]) + 1:]
        win_slots = torch.from_numpy((win_positions % WINDOW).astype(np.int64)).to(ref.device)
        ctx_pos_t = torch.from_numpy(win_positions.astype(np.int64)).to(ref.device)
        ctx_len = len(win_positions)
        visible = torch.ones((BLOCK, ctx_len + BLOCK), dtype=torch.bool, device=ref.device)
        row_ids = torch.arange(BLOCK, device=ref.device)[:, None]
        visible[:, :ctx_len] = ctx_pos_t[None, :] >= rows + row_ids - (SLIDING - 1)

        anchor = int(tokens[rows])
        block_tokens = torch.full((BLOCK,), ref.mask, dtype=torch.long, device=ref.device)
        block_tokens[0] = anchor
        block_pos = torch.arange(rows, rows + BLOCK, device=ref.device)

        h_ref = ref.embed[block_tokens].float()
        h_step = h_ref.clone()
        for layer_idx, layer in enumerate(ref.layers):
            q_s, kb_s, vb_s = ref.pre_attention(layer, h_step, block_pos)
            k_s = torch.cat((mod_keys[layer_idx].index_select(1, win_slots), kb_s), dim=1)
            v_s = torch.cat((mod_vals[layer_idx].index_select(1, win_slots), vb_s), dim=1)
            att_s = exact_gqa_attention(q_s, k_s, v_s, visible)
            h_step = ref.post_attention(layer, h_step, att_s)

            q_r, kb_r, vb_r = ref.pre_attention(layer, h_ref, block_pos)
            k_r = torch.cat((ref_keys[layer_idx].index_select(1, win_slots), kb_r), dim=1)
            v_r = torch.cat((ref_vals[layer_idx].index_select(1, win_slots), vb_r), dim=1)
            att_r = exact_gqa_attention(q_r, k_r, v_r, visible)
            h_ref = ref.post_attention(layer, h_ref, att_r)

            mod_h = f32_file(out / f"hidden{layer_idx + 1}.bin", (BLOCK, HIDDEN)).to(ref.device)
            gap_step = relative(mod_h, h_step)
            gap_e2e = relative(mod_h, h_ref)
            print(f"layer {layer_idx}: step-matched rel {gap_step:.5f}, end-to-end rel {gap_e2e:.5f}")
            if gap_step > 5e-3:
                failures.append(f"layer {layer_idx} step-matched rel diff {gap_step:.5f} > 5e-3")
            h_step = mod_h

        y_ref = block_drafter.rms(h_ref, ref.norm, ref.eps).to(torch.bfloat16)
        mod_y = bf16_file(out / "y.bin", (BLOCK, HIDDEN)).to(ref.device)
        y_gap = relative(mod_y, y_ref)
        base_mod = (mod_y @ ref.head.t()).float()
        base_ref = (y_ref @ ref.head.t()).float()
        logit_rel = relative(base_mod, base_ref)
        top_mod = base_mod.argmax(-1).tolist()
        top_ref = base_ref.argmax(-1).tolist()
        top_matches = sum(int(a == b) for a, b in zip(top_mod, top_ref))
        print(f"final y rel {y_gap:.5f}, base logits rel {logit_rel:.5f}, base top-1 match {top_matches}/{BLOCK}: mod={top_mod} ref={top_ref}")
        if top_matches != BLOCK:
            failures.append(f"base top-1 matched {top_matches}/{BLOCK}")

        drafted = np.fromfile(out / "drafts.bin", dtype=np.uint32).reshape(len(RULES), 2 * DRAFTS + 1)
        for idx, rule in enumerate(RULES):
            sampling = None
            sampled = None
            pos_t = None
            if rule[1] > 0.0:
                seed = rule[0] - (1 << 64) if rule[0] >= (1 << 63) else rule[0]
                sampling = (
                    torch.tensor(seed, device=ref.device),
                    torch.tensor(np.float32(rule[1]), device=ref.device),
                    torch.tensor(np.float32(rule[2] if rule[2] < 1.0 else 0.0), device=ref.device),
                    torch.tensor(rule[3], device=ref.device),
                )
                pos_t = torch.tensor(rows, device=ref.device)
                sampled = torch.tensor(True, device=ref.device)
            ref_toks_m, ref_scores_m, _, _ = ref.walk(mod_y, None, anchor, DRAFTS, sampling, pos_t, sampled)
            ref_toks_e, ref_scores_e, _, _ = ref.walk(y_ref, None, anchor, DRAFTS, sampling, pos_t, sampled)
            ref_toks_m = ref_toks_m.tolist()
            ref_scores_m = ref_scores_m.tolist()
            ref_toks_e = ref_toks_e.tolist()
            ref_scores_e = ref_scores_e.tolist()
            mod_toks = drafted[idx, 1:1 + DRAFTS].tolist()
            mod_scores = drafted[idx, 1 + DRAFTS:].view(np.float32).tolist()
            match_head = sum(int(a == b) for a, b in zip(mod_toks, ref_toks_m))
            match_e2e = sum(int(a == b) for a, b in zip(mod_toks, ref_toks_e))
            score_diff_head = max(abs(a - b) for a, b in zip(mod_scores, ref_scores_m))
            score_diff_e2e = max(abs(a - b) for a, b in zip(mod_scores, ref_scores_e))
            print(f"rule {idx}: head-exact {match_head}/{DRAFTS} (score max diff {score_diff_head:.2e}), "
                  f"e2e {match_e2e}/{DRAFTS} (score max diff {score_diff_e2e:.2e}); mod={mod_toks} ref={ref_toks_e}")
            if match_head != DRAFTS or score_diff_head > 2e-3:
                failures.append(f"rule {idx} selector mismatch: {match_head}/{DRAFTS}, score_diff={score_diff_head:.2e}")

    print("PASS" if not failures else "FAIL " + "; ".join(failures))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    rand_cmd = commands.add_parser("synthetic")
    rand_cmd.add_argument("--first", type=int, required=True)
    rand_cmd.add_argument("--rows", type=int, required=True)
    rand_cmd.add_argument("--out", type=Path, required=True)
    rand_cmd.add_argument("--seed", type=int, default=7)
    cmp_cmd = commands.add_parser("compare")
    cmp_cmd.add_argument("--checkpoint", required=True)
    cmp_cmd.add_argument("--embed-head-shard", required=True)
    cmp_cmd.add_argument("--norm-shard", required=True)
    cmp_cmd.add_argument("--out", type=Path, required=True)
    cmp_cmd.add_argument("--precise", action="store_true")
    options = parser.parse_args()
    return synthetic(options) if options.command == "synthetic" else compare(options)


if __name__ == "__main__":
    sys.exit(main())
