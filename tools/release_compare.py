#!/usr/bin/env python3
import json
import sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
exact = 0
for name in a:
    ta, tb = a[name].get("tokens") or [], b.get(name, {}).get("tokens") or []
    la, lb = a[name].get("logprobs") or [], b.get(name, {}).get("logprobs") or []
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    same = first is None and json.dumps(la) == json.dumps(lb)
    exact += same
    dl = 0.0
    for i, (x, y) in enumerate(zip(la, lb)):
        if first is not None and i >= first:
            break
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            dl = max(dl, abs(x - y))
        elif isinstance(x, dict) and isinstance(y, dict):
            for k in set(x) & set(y):
                dl = max(dl, abs(x[k] - y[k]))
    print(f"{name}: {'exact' if same else 'tokens-equal' if first is None else 'first-diff@' + str(first)} n={len(ta)} served_max_dlp={dl:.4f}")
print(f"EXACT {exact}/{len(a)}")
