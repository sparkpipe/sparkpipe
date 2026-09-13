#!/usr/bin/env python3
"""DFlash2 W4/W3 parity oracle: generate smoke inputs, dump the EXACT expected
outputs from the numpy oracles in tools/qwen36_dspark_reference.py.

W4 (top-K over the vocabulary) mirrors qwen36_dspark_reference.main() lines
"mask_logits = (hidden @ lm_head.T)" -> f32_to_bf16 -> top-K.
W3 (selector) calls the landed oracles _score_edges and _greedy_walk verbatim,
with hproj = bf16(hidden @ hidden_projection.T) exactly as main() computes it.

TIE RULE (pinned here, matched by the kernels): the top-K order is the total
order (score DESCENDING, id ASCENDING). main()'s np.argsort(-x) is quicksort,
i.e. UNSTABLE, so it leaves ties undefined; after the BF16 truncation of a
248320-wide logit row ties are the common case, not the corner case, so the
contract has to name one. Ascending-id is the rule the landed top-1 head
reduction already uses (candidate < warp_best_candidate) and the rule a STABLE
descending argsort produces.

EXACT regime: the "exact" cases draw every value from the lattice k/16 with
|k| small enough that every product and every partial sum is an exact fp32
value, so fp32 accumulation is order independent and the numpy oracle and the
CUDA kernel must agree BIT for BIT. The harness asserts that regime holds by
recomputing in fp64 and requiring bitwise equality. The "random" cases use
full-mantissa BF16 draws and are informational (fp32 accumulation order is
then observable).

Usage:
  python3 tools/qwen36_dflash2_w34_oracle.py --out /tmp/dflash2_w34
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "qwen36_dspark_reference", HERE / "qwen36_dspark_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = load_reference()


def lattice(rng, shape, limit):
    """Values k/16, |k| <= limit: exactly BF16-representable for limit <= 127."""
    k = rng.integers(-limit, limit + 1, shape).astype(np.float32)
    return (k / np.float32(16.0)).astype(np.float32)


def bf16_round(x):
    return REF.bf16_to_f32(REF.f32_to_bf16(np.ascontiguousarray(x, dtype=np.float32)))


def bf16_bits(x):
    return REF.f32_to_bf16(np.ascontiguousarray(x, dtype=np.float32))


def head_topk(logits, top_k):
    """The W4 oracle tail: the BF16-truncated logits, then the top-K under the
    pinned total order. Returns (ids [rows,K] u32, scores [rows,K] f32)."""
    rows = logits.shape[0]
    ids = np.empty((rows, top_k), dtype=np.uint32)
    scores = np.empty((rows, top_k), dtype=np.float32)
    for row in range(rows):
        order = np.lexsort((np.arange(logits.shape[1]), -logits[row]))[:top_k]
        ids[row] = order.astype(np.uint32)
        scores[row] = logits[row][order]
    return ids, scores


def assert_exact_regime(hidden, head, f32):
    """Prove the lattice case is inside the exactly-representable regime: the
    fp64 dot and the fp32 dot must be bitwise identical, so ANY fp32
    accumulation order (numpy's or the kernel's) lands on the same value."""
    f64 = (hidden.astype(np.float64) @ head.astype(np.float64).T).astype(np.float32)
    if not np.array_equal(np.ascontiguousarray(f32).view(np.uint32),
                          np.ascontiguousarray(f64).view(np.uint32)):
        raise AssertionError("case is NOT in the exact fp32 regime - shrink the lattice")


def write(path, array, dtype):
    array = np.ascontiguousarray(array, dtype=dtype)
    path.write_bytes(array.tobytes())
    return array


def dims_file(path, pairs):
    path.write_text("".join(f"{k} {v}\n" for k, v in pairs))


def emit_w4(out, name, rows, vocab, hidden_dim, top_k, seed, exact):
    """The head is generated and consumed in row blocks: the production shape
    (248320 x 5120) is 2.54 GiB of BF16 and 5.1 GiB as fp32, and every output
    logit is an independent dot, so blocking cannot change a single result."""
    rng = np.random.default_rng(seed)
    if exact:
        hidden = lattice(rng, (rows, hidden_dim), 15)
    else:
        hidden = bf16_round(rng.standard_normal((rows, hidden_dim)) * 0.05)
    block_rows = max(1, 33554432 // hidden_dim)
    logits = np.empty((rows, vocab), dtype=np.float32)
    with open(out / f"{name}_head.bf16", "wb") as handle:
        for start in range(0, vocab, block_rows):
            stop = min(vocab, start + block_rows)
            if exact:
                head = lattice(rng, (stop - start, hidden_dim), 15)
            else:
                head = bf16_round(rng.standard_normal((stop - start, hidden_dim)) * 0.05)
            handle.write(bf16_bits(head).tobytes())
            part = (hidden.astype(np.float32) @ head.astype(np.float32).T).astype(np.float32)
            if exact:
                assert_exact_regime(hidden, head, part)
            logits[:, start:stop] = part
            del head, part
    ids, scores = head_topk(bf16_round(logits), top_k)
    dims_file(out / f"{name}_dims.txt", [
        ("rows", rows), ("candidates", vocab), ("hidden", hidden_dim), ("top_k", top_k),
        ("exact", 1 if exact else 0)])
    write(out / f"{name}_hidden.bf16", bf16_bits(hidden), np.uint16)
    write(out / f"{name}_expect_ids.u32", ids, np.uint32)
    write(out / f"{name}_expect_scores.f32", scores, np.float32)
    ties = int(sum(len(np.unique(scores[r])) != top_k for r in range(rows)))
    print(f"{name}: rows={rows} vocab={vocab} hidden={hidden_dim} top_k={top_k} "
          f"exact={int(exact)} rows_with_ties_in_topk={ties} top1={ids[0][0]}/{scores[0][0]:.6f}")


def emit_w3(out, name, batch, slots, top_k, rank, hidden_dim, vocab, seed, exact, coarse=0, truncating=0):
    """coarse != 0 builds an EXACT tie into the max of every walk row, which is
    what exercises _greedy_walk's argmax-FIRST-max rule against the kernel's
    strict-greater scan; random edges never tie, so without this the walk's tie
    rule is untested. The tie is physically realizable (no duplicate candidate
    ids): two candidates per slot get identical SUCCESSOR rows in the codebook
    and an identical, dominating unary, so their two edge columns are bitwise
    equal in every predecessor row and both are the row max. A kernel that
    scanned with >= instead of > would take the LATER index and fail here."""
    rng = np.random.default_rng(seed)
    if exact:
        # H must stay on the k/16 lattice AND be BF16-exact so every downstream
        # product/sum is exact: a sparse 0/1 hidden with 16 live channels and
        # small projection weights keeps |16*H| <= 64.
        if truncating:
            # A DENSE lattice hidden makes the pre-truncation H a multiple of
            # 1/256 with ~11 significant bits, so hproj's BF16 truncation
            # actually ROUNDS (the sparse variant below lands H on the 1/16
            # lattice, where the truncation is a no-op and a kernel that
            # skipped it would still pass). The products and sums downstream
            # of the truncated H stay exact - the fp64 asserts below prove it.
            hidden = lattice(rng, (batch * slots, hidden_dim), 15)
        else:
            hidden = np.zeros((batch * slots, hidden_dim), dtype=np.float32)
            for row in range(batch * slots):
                hidden[row, rng.choice(hidden_dim, 16, replace=False)] = np.float32(1.0)
        projection = lattice(rng, (rank, hidden_dim), 4)
        predecessor = lattice(rng, (vocab, rank), 5)
        successor = lattice(rng, (vocab, rank), 5)
        unary = lattice(rng, (batch, slots, top_k), 15)
    else:
        hidden = bf16_round(rng.standard_normal((batch * slots, hidden_dim)) * 0.05)
        projection = bf16_round(rng.standard_normal((rank, hidden_dim)) * 0.05)
        predecessor = bf16_round(rng.standard_normal((vocab, rank)) * 0.5)
        successor = bf16_round(rng.standard_normal((vocab, rank)) * 0.5)
        unary = bf16_round(rng.standard_normal((batch, slots, top_k)) * 4.0)
    candidate_ids = rng.integers(0, vocab, (batch, slots, top_k)).astype(np.uint32)
    anchors = rng.integers(0, vocab, (batch,)).astype(np.uint32)
    if coarse:
        for b in range(batch):
            for s in range(slots):
                low, high = sorted(rng.choice(top_k, 2, replace=False).tolist())
                successor[candidate_ids[b, s, high]] = successor[candidate_ids[b, s, low]]
                unary[b, s, low] = np.float32(8.0)
                unary[b, s, high] = np.float32(8.0)

    raw_hproj = (hidden.astype(np.float32) @ projection.astype(np.float32).T).astype(np.float32)
    hproj = bf16_round(raw_hproj).reshape(batch, slots, rank)
    truncated = int((raw_hproj.reshape(-1) != hproj.reshape(-1)).sum())
    if exact:
        exact_hproj = bf16_round(hidden.astype(np.float64) @ projection.astype(np.float64).T)
        if not np.array_equal(hproj.reshape(-1).view(np.uint32),
                              exact_hproj.reshape(-1).view(np.uint32)):
            raise AssertionError("hproj is NOT in the exact fp32 regime")
    edges = REF._score_edges(predecessor.astype(np.float32), successor.astype(np.float32),
                             candidate_ids.astype(np.int64), unary.astype(np.float32),
                             hproj.astype(np.float32), anchors.astype(np.int64), top_k)
    if exact:
        edges64 = REF._score_edges(predecessor.astype(np.float64), successor.astype(np.float64),
                                   candidate_ids.astype(np.int64), unary.astype(np.float64),
                                   hproj.astype(np.float64), anchors.astype(np.int64), top_k)
        if not np.array_equal(edges.astype(np.float32).reshape(-1).view(np.uint32),
                              edges64.astype(np.float32).reshape(-1).view(np.uint32)):
            raise AssertionError("edges are NOT in the exact fp32 regime - shrink the lattice")
    drafts = np.empty((batch, slots), dtype=np.uint32)
    tied_walk_rows = 0
    for b in range(batch):
        drafts[b] = np.asarray(REF._greedy_walk(edges[b].astype(np.float32),
                                                candidate_ids[b]), dtype=np.uint32)
        previous = 0
        for step in range(slots):
            row = edges[b, step, previous].astype(np.float32)
            tied_walk_rows += int((row == row.max()).sum() > 1)
            previous = int(np.min(np.where(row == row.max())[0]))

    dims_file(out / f"{name}_dims.txt", [
        ("batch", batch), ("slots", slots), ("top_k", top_k), ("rank", rank),
        ("hidden", hidden_dim), ("vocab", vocab), ("exact", 1 if exact else 0)])
    write(out / f"{name}_hidden.bf16", bf16_bits(hidden), np.uint16)
    write(out / f"{name}_proj.bf16", bf16_bits(projection), np.uint16)
    write(out / f"{name}_pred.bf16", bf16_bits(predecessor), np.uint16)
    write(out / f"{name}_succ.bf16", bf16_bits(successor), np.uint16)
    write(out / f"{name}_candidates.u32", candidate_ids, np.uint32)
    write(out / f"{name}_unary.f32", unary, np.float32)
    write(out / f"{name}_anchor.u32", anchors, np.uint32)
    write(out / f"{name}_expect_hproj.f32", hproj, np.float32)
    write(out / f"{name}_expect_edges.f32", edges, np.float32)
    write(out / f"{name}_expect_drafts.u32", drafts, np.uint32)
    print(f"{name}: batch={batch} slots={slots} top_k={top_k} rank={rank} hidden={hidden_dim} "
          f"vocab={vocab} exact={int(exact)} tied_walk_rows={tied_walk_rows}/{batch * slots} "
          f"hproj_rounded_by_bf16={truncated}/{hproj.size} drafts[0]={drafts[0].tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/dflash2_w34")
    parser.add_argument("--cases", default="all")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = args.cases.split(",") if args.cases != "all" else None

    def wanted(name):
        return want is None or name in want

    K = REF.SELECTOR_TOP_K
    if wanted("w4_exact_vocab"):
        # the production vocabulary, the axis W4 exists for
        emit_w4(out, "w4_exact_vocab", REF.BLOCK - 1, REF.VOCAB, 256, K, 4001, True)
    if wanted("w4_exact_hidden"):
        # the production hidden width
        emit_w4(out, "w4_exact_hidden", REF.BLOCK - 1, 8192, REF.HIDDEN, K, 4002, True)
    if wanted("w4_random"):
        emit_w4(out, "w4_random", REF.BLOCK - 1, 32768, REF.HIDDEN, K, 4003, False)
    if wanted("w4_production"):
        # the real DFlash2 geometry: 7 mask slots x 248320 vocab x 5120 hidden
        emit_w4(out, "w4_production", REF.BLOCK - 1, REF.VOCAB, REF.HIDDEN, K, 4004, False)
    if wanted("w3_exact"):
        emit_w3(out, "w3_exact", 2, REF.BLOCK - 1, K, REF.SELECTOR_RANK, REF.HIDDEN,
                REF.VOCAB, 3001, True)
    if wanted("w3_random"):
        emit_w3(out, "w3_random", 2, REF.BLOCK - 1, K, REF.SELECTOR_RANK, REF.HIDDEN,
                4096, 3002, False)
    if wanted("w3_exact_trunc"):
        # the same exact regime, but with hproj's BF16 truncation load-bearing
        emit_w3(out, "w3_exact_trunc", 2, REF.BLOCK - 1, K, REF.SELECTOR_RANK, REF.HIDDEN,
                REF.VOCAB, 3004, True, truncating=1)
    if wanted("w3_tie"):
        # the walk's first-max tie rule, forced (see emit_w3's coarse lattice)
        emit_w3(out, "w3_tie", 4, REF.BLOCK - 1, K, REF.SELECTOR_RANK, REF.HIDDEN,
                REF.VOCAB, 3003, True, coarse=1)
    print(f"dumped to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
