import json, sys
import numpy as np
base_cap, base_rep, new_cap = (json.load(open(p)) for p in sys.argv[1:4])
new_rep = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[4] != "-" else {}
base_rep = base_rep.get("prompts", base_rep)
new_rep = new_rep.get("prompts", new_rep)
tie = 0.1
def score(rows, served_tokens, served_lps):
    agree, nonties, ties, deltas = 0, [], [], []
    for i, row in enumerate(rows[:len(served_tokens)]):
        if row["top"][0][0] == served_tokens[i]:
            agree += 1
        elif row["margin"] < tie:
            ties.append(i)
        else:
            nonties.append(i)
        deltas.append(abs(served_lps[i] - row["reference_logprob"]))
    return agree, ties, nonties, deltas
all_b, all_n = [], []
print("| prompt | tokens identical | base top-1 agree | base non-tie | base lp delta median / max | new top-1 agree | new non-tie | new lp delta median / max | ref source |")
print("|---|---|---|---|---|---|---|---|---|")
for name in base_rep:
    bt, nt = base_cap[name]["tokens"], new_cap[name]["tokens"]
    blp = [x[0][1] for x in base_cap[name]["logprobs"]]
    nlp = [x[0][1] for x in new_cap[name]["logprobs"]]
    div = next((i for i, (x, y) in enumerate(zip(bt, nt)) if x != y), None if len(bt) == len(nt) else min(len(bt), len(nt)))
    ba, bties, bnon, bd = score(base_rep[name]["rows"], bt, blp)
    if div is None:
        rows, src = base_rep[name]["rows"], "base ref (same tokens)"
    elif name in new_rep:
        rows, src = new_rep[name]["rows"], "new ref"
        assert [r["served"] for r in rows] == nt[:len(rows)], name
    else:
        rows, src = base_rep[name]["rows"][:div + 1], f"base ref rows 0..{div} (new ref pending)"
    na, nties, nnon, nd = score(rows, nt[:len(rows)], nlp[:len(rows)])
    all_b += bd; all_n += nd
    print(f"| {name} | {'yes' if div is None else 'no, from ' + str(div)} | {ba}/{len(bt)} | {bnon} | {np.median(bd):.1e} / {max(bd):.3f} | {na}/{len(rows)} | {nnon} | {np.median(nd):.1e} / {max(nd):.3f} | {src} |")
def st(v):
    a = np.array(v)
    return f"n={len(a)} median={np.median(a):.2e} mean={a.mean():.4f} p99={np.percentile(a, 99):.3f} max={a.max():.3f}"
print("all base:", st(all_b))
print("all new:", st(all_n))
same = [(name, i) for name in base_rep for i, (x, y) in enumerate(zip(base_cap[name]["tokens"], new_cap[name]["tokens"])) if x == y and all(a == b for a, b in zip(base_cap[name]["tokens"][:i], new_cap[name]["tokens"][:i]))]
sb = [abs(base_cap[n]["logprobs"][i][0][1] - base_rep[n]["rows"][i]["reference_logprob"]) for n, i in same]
sn = [abs(new_cap[n]["logprobs"][i][0][1] - base_rep[n]["rows"][i]["reference_logprob"]) for n, i in same]
signed_b = [base_cap[n]["logprobs"][i][0][1] - base_rep[n]["rows"][i]["reference_logprob"] for n, i in same]
signed_n = [new_cap[n]["logprobs"][i][0][1] - base_rep[n]["rows"][i]["reference_logprob"] for n, i in same]
print("shared-context positions (identical prefix, same reference):", len(same))
print("  base |served-ref|:", st(sb), f"signed mean={np.mean(signed_b):+.4f}")
print("  new  |served-ref|:", st(sn), f"signed mean={np.mean(signed_n):+.4f}")
print("  new closer at", sum(a < b for a, b in zip(sn, sb)), "farther at", sum(a > b for a, b in zip(sn, sb)), "equal at", sum(a == b for a, b in zip(sn, sb)))
