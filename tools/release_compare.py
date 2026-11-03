#!/usr/bin/env python3
import json
import sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
exact = equal = 0
for name in a:
    ta, tb = a[name].get("tokens") or [], b.get(name, {}).get("tokens") or []
    la, lb = a[name].get("logprobs") or [], b.get(name, {}).get("logprobs") or []
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    same = first is None and json.dumps(la) == json.dumps(lb)
    exact += same
    equal += first is None and name in b
    dl = 0.0
    for i, (x, y) in enumerate(zip(la, lb)):
        if first is not None and i >= first:
            break
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            dl = max(dl, abs(x - y))
        elif isinstance(x, dict) and isinstance(y, dict):
            for k in set(x) & set(y):
                dl = max(dl, abs(x[k] - y[k]))
        elif isinstance(x, list) and isinstance(y, list) and x and y:
            served = [x[0], y[0]]
            if all(isinstance(entry, list) and len(entry) == 2 for entry in served):
                dl = max(dl, abs(served[0][1] - served[1][1]))
            tops = [{entry[0]: entry[1] for entry in side[1:] if isinstance(entry, list) and len(entry) == 2} for side in (x, y)]
            for k in set(tops[0]) & set(tops[1]):
                dl = max(dl, abs(tops[0][k] - tops[1][k]))
    print(f"{name}: {'exact' if same else 'tokens-equal' if first is None else 'first-diff@' + str(first)} n={len(ta)} served_max_dlp={dl:.4f}")
print(f"EXACT {exact}/{len(a)}")
sys.exit(0 if a and equal == len(a) else 1)
