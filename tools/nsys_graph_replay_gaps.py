import csv, sys, collections, statistics, re
rows = list(csv.DictReader(open(sys.argv[1])))
g = collections.defaultdict(list)
for r in rows:
    if r["Name"].startswith("["): continue
    g[r["CorrId"]].append((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"]))
steps = [sorted(v) for v in g.values() if len(v) > 1000]
print("graph replays", len(steps), "kernels/step", statistics.median(len(s) for s in steps))
res = []
for s in steps:
    span = s[-1][0] + s[-1][1] - s[0][0]
    busy = sum(d for _, d, _ in s)
    gaps = [max(0, s[i+1][0] - (s[i][0] + s[i][1])) for i in range(len(s) - 1)]
    res.append((span / 1e6, busy / 1e6, sum(gaps) / 1e6, statistics.median(gaps) / 1e3))
res.sort()
print("span_ms busy_ms gap_ms median_gap_us  (sorted by span)")
for x in res: print("%.2f %.2f %.2f %.2f" % x)
best = min(steps, key=lambda s: s[-1][0] + s[-1][1] - s[0][0])
cat = collections.Counter()
def short(n):
    n = re.sub(r"\(.*", "", n); return n.replace("void ", "")[:70]
for _, d, n in best: cat[short(n)] += d
print("fastest replay kernel time by kernel (ms):")
for n, d in cat.most_common(22): print("%7.3f %s" % (d / 1e6, n))
