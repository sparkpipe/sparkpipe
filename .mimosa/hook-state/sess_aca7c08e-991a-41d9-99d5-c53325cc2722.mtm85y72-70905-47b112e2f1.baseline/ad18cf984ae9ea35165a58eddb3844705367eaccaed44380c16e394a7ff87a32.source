#!/usr/bin/env python3
"""Run the GQA store and decode kernels on a CPU and score them against a reference.

tests/host_cuda/gqa_host.cu includes inference/kernels/gqa.cuh UNMODIFIED and
gives it a grid: store K and V rows into a paged cache through the page table,
then attend over it, once over the whole context and once over a window. The
kernels compiled there are the kernels qwen_3_6 and mimo_2_5 launch for
sm_121a. A reimplementation inside the test would only prove the test agrees
with itself, so the reference below is written from the math, not the code:

    out[h] = sum_p softmax_p(q[h] . k[h / (heads/kv_heads), p] * scale)
             * v[h / (heads/kv_heads), p]

What this catches, because each was a live defect class in these drivers: the
store reading a buffer nothing wrote (the output is then uncorrelated with the
reference), the value read from the key's region or at the key's width (MiMo's
value is narrower than its key), GQA grouping collapsed to one KV head per
query head, and page-table addressing ignored (two sequences, interleaved
pages).
"""
import math
import subprocess
import sys
from pathlib import Path

from host_cuda_compiler import host_cuda_cxx

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tests" / "host_cuda" / "gqa_host.cu"
BINARY = Path("/tmp") / "lm_gqa_host"


def build():
    result = subprocess.run(
        [host_cuda_cxx(), "-std=c++17", "-O0", f"-I{ROOT}", f"-I{ROOT}/tests/host_cuda",
         "-x", "c++", str(SOURCE), "-o", str(BINARY)],
        capture_output=True, text=True)
    if result.returncode != 0:
        errors = [l for l in result.stderr.split("\n") if "error" in l]
        print("FAIL host build:", (errors or [result.stderr])[0][:200])
        return False
    return True


def parse(text):
    lines = text.strip().split("\n")
    header = lines[0].split()
    shape = {header[i].lower(): int(header[i + 1]) if header[i] != "SCALE"
             else float(header[i + 1])
             for i in range(0, len(header) - 1, 2)}
    values = {}
    for line in lines[1:]:
        tag, _, number = line.partition(" ")
        values.setdefault(tag, []).append(float(number.split()[0]))
    return shape, values


def reference(shape, values, positions_per_row):
    """softmax attention per query head, reading K and V from the ORIGINAL
    dense rows - never from the pool - so the store's layout is validated by
    the decode agreeing, not assumed."""
    kv_heads, head_dim = shape["kv_heads"], shape["head_dim"]
    value_dim, heads = shape["value_dim"], shape["heads"]
    sequences, scale = shape["sequences"], shape["scale"]
    group = heads // kv_heads
    context = shape["context"]
    out = []
    for sequence in range(sequences):
        for head in range(heads):
            kv_head = head // group
            q_base = (sequence * heads + head) * head_dim
            scores = []
            for position in positions_per_row[sequence]:
                row = sequence * context + position
                k_base = (row * kv_heads + kv_head) * head_dim
                score = sum(
                    values["query"][q_base + i] * values["key"][k_base + i]
                    for i in range(head_dim))
                scores.append(score * scale)
            top = max(scores)
            weights = [math.exp(s - top) for s in scores]
            total = sum(weights)
            for element in range(value_dim):
                acc = 0.0
                for weight, position in zip(weights, positions_per_row[sequence]):
                    row = sequence * context + position
                    v_base = (row * kv_heads + kv_head) * value_dim
                    acc += weight * values["value"][v_base + element]
                out.append(acc / max(total, 1.0e-20))
    return out


def compare(tag, want, got):
    if len(want) != len(got):
        print(f"FAIL {tag}: reference produced {len(want)}, kernels {len(got)}")
        return False
    worst = max(abs(a - b) for a, b in zip(want, got))
    magnitude = max(abs(x) for x in want) or 1.0
    relative = worst / magnitude
    print(f"{tag}: outputs compared {len(got)}  "
          f"max |kernel - reference| = {worst:.3e}  relative {relative:.3e}")
    if magnitude < 1e-4:
        print(f"FAIL {tag}: outputs all near zero; the test cannot see anything")
        return False
    # Every input is a printed bf16 value, so the only slack is the kernel's
    # own bf16 output rounding: half a ulp of an 8-bit significand is 2^-9
    # (1.95e-3), and the observed error sits exactly at that bound. The bar is
    # 4e-3 - just past rounding, and orders of magnitude below what a wrong
    # head mapping, a wrong value region or an ignored page table moves.
    if relative > 4e-3:
        print(f"FAIL {tag}: the kernels do not compute the reference attention")
        return False
    return True


def main():
    if not build():
        return 1
    run = subprocess.run([str(BINARY)], capture_output=True, text=True)
    if run.returncode != 0:
        print("FAIL host run:", run.stderr.strip()[:200])
        return 1
    shape, values = parse(run.stdout)
    sequences, context, window = shape["sequences"], shape["context"], shape["window"]
    full = [[p for p in range(context)] for _ in range(sequences)]
    windowed = [[p for p in range(context - window, context)] for _ in range(sequences)]
    ok = compare("full context",
                 reference(shape, values, full), values["out_full"])
    ok = compare("windowed",
                 reference(shape, values, windowed), values["out_window"]) and ok
    if not ok:
        return 1
    print("\nthe GQA store and decode kernels, run on a CPU, match the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
