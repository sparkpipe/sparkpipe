#!/usr/bin/env python3
"""Run K3's MLA path on a CPU: store into a paged cache, then attend over it.

The last of K3's forward that had never executed. It needs more setup than the
other harnesses because the cache is paged, and that addressing is the point:
LmKvSlot returns null for an unmapped page rather than an address into page
zero, and its comment says why - the alternative is reading another sequence's
keys and getting output that is fluent and wrong.

TWO SEQUENCES WITH INTERLEAVED PAGES. One sequence cannot tell correct page
addressing from ignoring the page table, because every mapping is the identity.
Sequence 0 holds physical pages 0 and 2, sequence 1 holds 1 and 3, so a kernel
that indexed by position alone would read the other sequence's tokens - and the
test checks that reading them would have produced a different answer.

The reference is absorbed MLA: the cached latent row IS the key, and its latent
half is also the value.
"""
import math
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "mla_host.cu"
BINARY = Path("/tmp") / "lm_mla_host"
PAGE_SLOTS = 2
PAGES_PER_SEQUENCE = 2
PAGE_TABLE = [0, 2, 1, 3]
SCALE = 0.25


def build_and_run():
    result = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O0", f"-I{ROOT}", f"-I{ROOT}/tests/host_cuda",
         "-x", "c++", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True)
    if result.returncode != 0:
        errors = [l for l in result.stderr.split("\n") if "error" in l]
        print("FAIL host build:", (errors or [result.stderr])[0][:200])
        return None
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0:
        print("FAIL host run:", run.stderr.strip()[:200])
        return None
    return run.stdout


def attend(query, keys, latent, scale):
    scores = [sum(q * k for q, k in zip(query, key)) * scale for key in keys]
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    return [sum(w * key[e] for w, key in zip(weights, keys)) / total
            for e in range(latent)]


def main():
    text = build_and_run()
    if text is None:
        return 1
    lines = text.strip().split("\n")
    header = lines[0].split()
    latent, rope = int(header[1]), int(header[3])
    heads, sequences, context = int(header[5]), int(header[7]), int(header[9])
    v = {}
    for line in lines[1:]:
        tag, _, number = line.partition(" ")
        v.setdefault(tag, []).append(float(number))

    width = latent + rope
    # the rows as stored, indexed the way the page table maps them
    def key_of(sequence, position):
        row = sequence * context + position
        return v["slot"][row * width:(row + 1) * width]

    failures = 0
    got = v["out"]
    index = 0
    wrong_page_would_differ = 0
    for sequence in range(sequences):
        keys = [key_of(sequence, p) for p in range(context)]
        # what a kernel that ignored the page table would have read: the same
        # physical slots, but resolved as if sequence 0 owned pages 0 and 1
        other = [key_of(1 - sequence, p) for p in range(context)]
        for head in range(heads):
            base = (sequence * heads + head) * width
            query = v["query"][base:base + width]
            want = attend(query, keys, latent, SCALE)
            wrong = attend(query, other, latent, SCALE)
            magnitude = max(abs(x) for x in want) or 1.0
            worst = max(abs(a - b) for a, b in zip(want, got[index:index + latent]))
            if worst / magnitude > 6e-3:
                print(f"  FAIL sequence {sequence} head {head}: "
                      f"relative {worst / magnitude:.3e}")
                failures += 1
            if max(abs(a - b) for a, b in zip(want, wrong)) / magnitude > 1e-2:
                wrong_page_would_differ += 1
            index += latent

    print(f"latent {latent}  rope {rope}  heads {heads}  "
          f"sequences {sequences}  context {context}")
    print(f"page table {PAGE_TABLE}, interleaved across sequences")
    print(f"head-sequence pairs where reading the other sequence's pages "
          f"would have differed: {wrong_page_would_differ} of {sequences * heads}")
    if wrong_page_would_differ < sequences * heads:
        print("\nFAIL the two sequences are not distinguishable; this cannot "
              "tell correct page addressing from ignoring the page table")
        return 1
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nthe real MLA store and attention kernels match the reference, "
          "and address the right pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
