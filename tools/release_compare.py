#!/usr/bin/env python3
import json
import sys

PATHS = ("chain_tokens", "speculate_tokens", "batch_tokens", "sampled_tokens", "sampled_batch_tokens")
SAME_AS_SINGLE = ("chain_tokens", "speculate_tokens", "batch_tokens")


def first_difference(x, y):
    first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), None)
    if first is None and len(x) != len(y):
        first = min(len(x), len(y))
    return first


def logprob_delta(la, lb, first):
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
    return dl


def describe(first):
    return "equal" if first is None else "first-diff@" + str(first)


a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
exact = equal = 0
failures = []
for name in a:
    ta, tb = a[name].get("tokens") or [], b.get(name, {}).get("tokens") or []
    la, lb = a[name].get("logprobs") or [], b.get(name, {}).get("logprobs") or []
    first = first_difference(ta, tb)
    same = first is None and json.dumps(la) == json.dumps(lb)
    exact += same
    equal += first is None and name in b
    paths = []
    for path in PATHS:
        if path not in a[name]:
            continue
        if path not in b.get(name, {}):
            failures.append(f"{name}: the new capture has no {path}")
            continue
        path_first = first_difference(a[name][path] or [], b[name][path] or [])
        paths.append(f"{path.replace('_tokens', '')}={describe(path_first)}")
        if path_first is not None:
            failures.append(f"{name}: {path} differs from the reference at token {path_first}")
    for path in SAME_AS_SINGLE:
        if path in b.get(name, {}) and first_difference(tb, b[name][path] or []) is not None:
            failures.append(f"{name}: the new capture's {path} differ from its one-step tokens at token {first_difference(tb, b[name][path] or [])}")
    speculation = b.get(name, {}).get("speculation")
    print(f"{name}: {'exact' if same else 'tokens-equal' if first is None else 'first-diff@' + str(first)} n={len(ta)} "
          f"served_max_dlp={logprob_delta(la, lb, first):.4f}" + (" " + " ".join(paths) if paths else "") + (f" speculation={speculation}" if speculation else ""))
for failure in failures:
    print("FAIL", failure)
print(f"EXACT {exact}/{len(a)}")
sys.exit(0 if a and equal == len(a) and not failures else 1)
