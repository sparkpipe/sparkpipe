import json, sys
import numpy as np
batch = json.load(open(sys.argv[1]))
report = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[2] != "-" else None
if report is not None and "prompts" in report:
    report = report["prompts"]
runs = batch["runs"]
alone = runs["B1"]
def stats(values):
    if not values:
        return "n=0"
    a = np.abs(np.array(values))
    return f"n={len(a)} median={np.median(a):.2e} p99={np.percentile(a, 99):.3f} max={a.max():.3f} exact0={np.mean(a == 0):.2f}"
for label in [k for k in runs if k != "B1"]:
    other = runs[label]
    deltas, deltas_ref_prompts, flips = [], [], []
    for name, a in alone.items():
        b = other[name]
        for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
            la, lb = a["logprobs"][i], b["logprobs"][i]
            if x != y:
                margin = la[1][1] - la[2][1] if len(la) > 2 else None
                ob = lb[1][1] - lb[2][1] if len(lb) > 2 else None
                flips.append((name, i, round(margin, 3) if margin is not None else None, round(ob, 3) if ob is not None else None))
                break
            d = la[0][1] - lb[0][1]
            deltas.append(d)
            if report is not None and name in report:
                deltas_ref_prompts.append(d)
    print(f"B1 vs {label}: served-token |lp delta| {stats(deltas)}")
    if report is not None:
        print(f"B1 vs {label} (reference prompts only): {stats(deltas_ref_prompts)}")
    print(f"B1 vs {label}: top-1 flips {len(flips)}: " + "; ".join(f"{n}@{i} B1 top1-top2 margin {m} other margin {o}" for n, i, m, o in flips))
if report is not None:
    ref_d, ref_d32, ties, nonties = [], [], [], []
    for name, e in report.items():
        for i, row in enumerate(e["rows"]):
            served_lp = row.get("served_logprob", row["served_top"][0][1] if "served_top" in row else None)
            d = served_lp - row["reference_logprob"]
            ref_d.append(d)
            if i < 32:
                ref_d32.append(d)
            if row["top"][0][0] != row["served"]:
                (ties if row["margin"] < 0.1 else nonties).append((name, i, round(row["margin"], 3)))
    print(f"served vs reference: |lp delta| {stats(ref_d)}")
    print(f"served vs reference (first 32 tokens): {stats(ref_d32)}")
    print(f"served vs reference: top-1 disagreements near-tie {len(ties)} non-tie {len(nonties)} {nonties}")
