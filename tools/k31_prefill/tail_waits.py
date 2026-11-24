import re
import sys
import statistics
from collections import defaultdict

events = []
with open(sys.argv[1], errors="replace") as handle:
    for line in handle:
        if line.startswith(("CONCURRENT_KERNEL", "KERNEL")):
            m = re.match(r'^(?:CONCURRENT_KERNEL|KERNEL)\s*\[\s*(\d+),\s*(\d+)\s*\][^"]*"([^"]*)"', line)
            if m:
                events.append((int(m.group(1)), int(m.group(2)), m.group(3)))
events.sort()
waits = []
gemm_before = []
last_conv = None
last_gemm = None
index = 0
for i, (s, e, n) in enumerate(events):
    if "K3StreamedConvTailKernel" in n:
        last_conv = i
    elif "LmStreamGemmKernel" in n:
        last_gemm = (s, e)
    elif last_conv is not None and "SparkTpMeshHardwareArrivalKernel" in n and i - last_conv <= 3:
        waits.append((s, (e - s) / 1e6, last_gemm[1] - last_gemm[0] if last_gemm else 0))
        last_conv = None
print("conv-tail arrival waits:", len(waits))
d = [w[1] for w in waits]
print("ms min %.2f median %.2f mean %.2f p90 %.2f max %.2f" % (min(d), statistics.median(d), statistics.mean(d), sorted(d)[int(len(d) * 0.9)], max(d)))
by_shard = defaultdict(list)
for k, w in enumerate(waits):
    by_shard[k % 16].append(w[1])
print("mean wait by shard index within a layer:", " ".join("%d:%.1f" % (s, statistics.mean(v)) for s, v in sorted(by_shard.items())))
g = [w[2] / 1e6 for w in waits if w[2]]
print("the GEMM right before: ms median %.2f mean %.2f" % (statistics.median(g), statistics.mean(g)))
print("first 40 waits (ms):", " ".join("%.1f" % w[1] for w in waits[:40]))
