#!/usr/bin/env python3
"""Run the head's chunked top-k on a CPU and score it against float32.

tests/host_cuda/head_host.cu includes head.cuh unmodified and runs the
candidate/commit pair at THREADS == 1 - the sequential schedule a correct
kernel must also be valid under.

What is checked, beyond the set: the order (descending score), the tie rule
(lower token id, the same rule the argmax commit uses - the harness plants
two identical weight rows at tokens 4 and 13), the scores themselves in
float32, and a restricted set that is unsorted and excludes the vocabulary's
best, so following row order instead of the id map fails loudly.
"""
import struct
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "head_host.cu"
BINARY = Path("/tmp") / "lm_head_host"
RESTRICTED_IDS = [20, 3, 14, 1, 22, 7, 9, 0, 17]


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def build():
    result = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O0", f"-I{ROOT}/tests/host_cuda/shim",
         f"-I{ROOT}/tests/host_cuda", f"-I{ROOT}",
         "-x", "c++", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True)
    if result.returncode != 0:
        errors = [l for l in result.stderr.split("\n") if "error" in l]
        print("FAIL host build:", (errors or [result.stderr])[0][:200])
        return False
    return True


def topk(row, vocab_ids, normed, weight, hidden, k):
    scores = []
    for token in vocab_ids:
        total = 0.0
        for h in range(hidden):
            total = f32(total + f32(normed[row * hidden + h]
                                    * weight[token * hidden + h]))
        scores.append(total)
    order = sorted(range(len(vocab_ids)),
                   key=lambda i: (-scores[i], vocab_ids[i]))
    return [(vocab_ids[i], scores[i]) for i in order[:k]]


def main():
    if not build():
        return 1
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0:
        print("FAIL host run:", (run.stdout + run.stderr).strip()[:200])
        return 1
    lines = run.stdout.strip().split("\n")
    header = lines[0].split()
    vocab, hidden, top_k, rows = (int(header[1]), int(header[3]),
                                  int(header[5]), int(header[7]))
    normed, weight, full, restricted = [], [], [], []
    for line in lines[1:]:
        tag, rest = line.split(" ", 1)
        parts = rest.split()
        if tag == "normed":
            normed.append(float(parts[0]))
        elif tag == "weight":
            weight.append(float(parts[0]))
        elif tag == "full":
            full.append((int(parts[0]), int(parts[1]), float(parts[2])))
        elif tag == "restricted":
            restricted.append((int(parts[0]), int(parts[1]), float(parts[2])))

    failures = 0
    tie_rows = 0
    for got, ids in ((full, list(range(vocab))), (restricted, RESTRICTED_IDS)):
        reference = [topk(r, ids, normed, weight, hidden, top_k)
                     for r in range(rows)]
        for index, (row, token, score) in enumerate(got):
            r, k = divmod(index, top_k)
            exp_token, exp_score = reference[r][k]
            if row != r or token != exp_token or abs(score - exp_score) > 1e-6:
                print(f"  FAIL row {r} k {k}: got ({token}, {score:.9g}), "
                      f"reference ({exp_token}, {exp_score:.9g})")
                failures += 1
    # The planted tie (identical weight rows at tokens 4 and 13) must order
    # the lower id first wherever both make the shortlist - and it must have
    # shown up at least once, or the test proves nothing about ties.
    for r in range(rows):
        row_tokens = [t for _, t, _ in full[r * top_k:(r + 1) * top_k]]
        if 4 in row_tokens and 13 in row_tokens:
            tie_rows += 1
            if row_tokens.index(4) > row_tokens.index(13):
                print(f"  FAIL row {r}: tie ordered 13 before 4")
                failures += 1

    print(f"vocab {vocab}  hidden {hidden}  top_k {top_k}  rows {rows}")
    print(f"rows where the planted tie reached the shortlist: {tie_rows}")
    if tie_rows == 0:
        print("\nFAIL the planted tie never reached a shortlist; the tie "
              "rule is untested")
        return 1
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe real head top-k kernels, run on a CPU, match float32 "
          "reference including ties")
    return 0


if __name__ == "__main__":
    sys.exit(main())
