#!/usr/bin/env python3
"""TP4 sharding equivalence check, run OFFLINE (no ring needed).

Five single-spark dumps of the same stage-0 token (the runner's --dump
hidden): the FULL stage pack at tp_degree 1, and the four rank packs at
tp_degree 1. The rank packs compute full-width PARTIAL hiddens (their
out-projections take the rank's slices), so the all-reduce the TP4 run
performs is, element-wise, the sum of the four rank dumps - the exact
contract the deployment ships. This script checks it:

    full[k] ~= rank0[k] + rank1[k] + rank2[k] + rank3[k]

in fp32 with a BF16-rounding tolerance (the tree sums four BF16 partials
where the full GEMM emits one). PASS prints the max relative deviation;
FAIL names the worst columns. Usage:

    k3_tp4_equivalence_check.py FULL_DUMP RANK0 RANK1 RANK2 RANK3
"""
import struct
import sys


def load(path):
    raw = open(path, "rb").read()
    values = struct.unpack("<%dH" % (len(raw) // 2), raw)
    return [v if v < 0x8000 else v - 0x10000 for v in values]


def bf16_to_f32(bits):
    # sign-extended int16 -> fp32
    return float(bits) / 256.0


def main():
    if len(sys.argv) != 6:
        print("usage: k3_tp4_equivalence_check.py FULL R0 R1 R2 R3")
        return 2
    full = load(sys.argv[1])
    ranks = [load(p) for p in sys.argv[2:]]
    if not (len(full) == len(ranks[0]) == len(ranks[1]) == len(ranks[2])
            == len(ranks[3])):
        print("FAIL: dump lengths differ")
        return 1
    failures = 0
    worst = 0.0
    worst_col = -1
    for k in range(len(full)):
        got = sum(bf16_to_f32(r[k]) for r in ranks)
        expect = bf16_to_f32(full[k])
        delta = abs(got - expect)
        rel = delta / max(abs(expect), 0.25)
        if rel > worst:
            worst, worst_col = rel, k
        if rel > 0.03:
            failures += 1
            if failures <= 8:
                print(f"  mismatch[{k}] full={expect:.6g} sum={got:.6g} "
                      f"rel={rel:.4f}")
    print(f"checked {len(full)} columns, worst rel {worst:.5f} at "
          f"column {worst_col}, {failures} beyond 0.03")
    if failures:
        print("TP4 EQUIVALENCE FAIL")
        return 1
    print("TP4 EQUIVALENCE PASS (the rank partials sum to the full stage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
