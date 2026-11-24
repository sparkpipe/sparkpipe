import re
import sys
from collections import defaultdict

line_re = re.compile(r'^(CONCURRENT_KERNEL|KERNEL)\s*\[\s*(\d+),\s*(\d+)\s*\][^"]*"([^"]*)".*?streamId (\d+)')
events = []
with open(sys.argv[1], errors="replace") as handle:
    pending = None
    for line in handle:
        if line.startswith(("CONCURRENT_KERNEL", "KERNEL")):
            m = re.match(r'^(CONCURRENT_KERNEL|KERNEL)\s*\[\s*(\d+),\s*(\d+)\s*\][^"]*"([^"]*)"', line)
            if m:
                pending = [int(m.group(2)), int(m.group(3)), m.group(4), None]
                sm = re.search(r'streamId (\d+)', line)
                if sm:
                    pending[3] = int(sm.group(1))
                    events.append(tuple(pending))
                    pending = None
        elif pending is not None:
            sm = re.search(r'streamId (\d+)', line)
            if sm:
                pending[3] = int(sm.group(1))
                events.append(tuple(pending))
                pending = None
events.sort()
streams = defaultdict(lambda: [0, 0.0])
for s, e, n, st in events:
    streams[st][0] += 1
    streams[st][1] += (e - s) / 1e9
print("stream\tlaunches\tseconds")
for st, (c, t) in sorted(streams.items(), key=lambda kv: -kv[1][1])[:8]:
    print(f"{st}\t{c}\t{t:.1f}")
def short(name):
    m = re.match(r'_Z\d+(\w+?)(I|E|P|v)', name)
    base = re.sub(r'^_Z\d+', '', name)
    m = re.match(r'([A-Za-z0-9_]+?)(?:I|Ev|Pv|PK|P\d|EvP)', base)
    return (m.group(1) if m else base)[:40]
attr = defaultdict(lambda: [0, 0.0])
last_compute = "start"
for s, e, n, st in events:
    sn = short(n)
    if "SparkTpMesh" in sn or "LmPassSum" in sn:
        attr[(last_compute, sn)][0] += 1
        attr[(last_compute, sn)][1] += (e - s) / 1e9
    else:
        last_compute = sn
print("\npreceding compute kernel\tmesh kernel\tlaunches\tseconds")
for (pre, mk), (c, t) in sorted(attr.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"{pre}\t{mk}\t{c}\t{t:.2f}")
