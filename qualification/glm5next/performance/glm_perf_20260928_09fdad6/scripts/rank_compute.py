import sys, statistics, collections, importlib.util, glob, os, pathlib
spec = importlib.util.spec_from_file_location("tcb", pathlib.Path(__file__).resolve().parents[5] / "tools/tp_chain_budget.py")
tcb = importlib.util.module_from_spec(spec); spec.loader.exec_module(tcb)
d = sys.argv[1]
ranks = {}
for f in glob.glob(d + "/r*.log"):
    r = int(os.path.basename(f).split(".")[0][1:]); ranks[r] = tcb.parse(f)
epochs = set.intersection(*[set(c) for c in ranks.values()])
lo, hi = float(sys.argv[2]), float(sys.argv[3])
per = collections.defaultdict(list); rows = []
for e in sorted(epochs):
    rec = {r: ranks[r][e] for r in ranks}
    if not all("peer" in x and x["path"] == "graph" and x["replays"] for x in rec.values()): continue
    if len({x["steps"] for x in rec.values()}) != 1: continue
    st = rec[0]["steps"]
    run = statistics.mean(sum(x["replays"]) for x in rec.values()) / st
    if not (lo <= run < hi): continue
    comp = {r: (sum(x["replays"]) - x["peer"] - x["source"] - x["copy"] - x["combine"]) / st for r, x in rec.items()}
    peer = {r: x["peer"] / st for r, x in rec.items()}
    other = {r: (x["source"] + x["copy"] + x["combine"]) / st for r, x in rec.items()}
    host = {r: (x["total"] - sum(x["replays"])) / st for r, x in rec.items()}
    for r in rec: per[r].append((comp[r], peer[r], other[r], host[r]))
    rows.append((run, min(peer.values()), statistics.mean(peer.values()), max(comp.values()) - min(comp.values())))
print("chains", len(rows), "run/step median %.2f floor %.2f peer-mean %.2f compute-spread %.2f" % tuple(statistics.median(x[i] for x in rows) for i in range(4)))
print("rank  compute  peer  other  host  (per-step medians, ms)")
for r in sorted(per):
    v = per[r]
    print("%4d %8.2f %5.2f %6.2f %5.2f" % (r, *(statistics.median(x[i] for x in v) for i in range(4))))
