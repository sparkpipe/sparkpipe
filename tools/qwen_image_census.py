import json, re, sys

index_path = sys.argv[1]
wm = json.load(open(index_path))["weight_map"]
kinds = {}
shapes = {}
for name in wm:
    m = re.sub(r"\.\d+\.", ".N.", name)
    kinds[m] = kinds.get(m, 0) + 1
for pattern in sorted(kinds):
    print(f"{kinds[pattern]:4d}  {pattern}")
print(f"total tensors: {len(wm)}")
