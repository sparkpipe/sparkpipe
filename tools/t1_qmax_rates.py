#!/usr/bin/env python3
"""Per-array-kind divergence rates between the reference and a T1R1 candidate.

The compare tool convicts on the first offender; the gate owner asked for the
exact rates behind a verdict, so this reports every array without stopping.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import read_fixture


def rate(name, want, got, rel, absolute):
    if want.dtype.kind in ("i", "u"):
        equal = want.shape == got.shape and np.array_equal(want, got)
        return {"name": name, "kind": "ids", "exact": bool(equal)}
    a = want.astype(np.float32)
    b = got.astype(np.float32)
    if want.shape != got.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"name": name, "kind": "shape", "exact": False}
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    rel_err = np.where(denom > 0, diff / denom, diff)
    bad = (rel_err > rel) & (diff > absolute)
    return {"name": name, "kind": "banded", "exact": not bool(bad.any()),
            "worst_rel": float(rel_err.max()), "offending": int(bad.sum()),
            "elements": int(bad.size)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--rel", type=float, default=0.02)
    parser.add_argument("--abs", type=float, default=1e-3)
    args = parser.parse_args()
    _, reference = read_fixture(args.reference)
    _, candidate = read_fixture(args.candidate)
    rows = [rate(name, reference[name], candidate[name], args.rel, args.abs)
            for name in sorted(reference)]
    groups = {}
    for row in rows:
        kind = row["kind"]
        group = groups.setdefault(kind, {"arrays": 0, "exact": 0, "worst_rel": 0.0})
        group["arrays"] += 1
        group["exact"] += 1 if row["exact"] else 0
        group["worst_rel"] = max(group["worst_rel"], row.get("worst_rel", 0.0))
    ids_exact = sum(1 for row in rows if row["kind"] == "ids" and row["exact"])
    generated = next((row["exact"] for row in rows
                      if row["name"] == "generated_token_ids"), False)
    print(json.dumps({"totals": groups,
                      "ids_exact_arrays": ids_exact,
                      "ids_arrays": groups.get("ids", {}).get("arrays", 0),
                      "generated_exact": bool(generated)},
                     indent=2))
    detail_path = os.path.splitext(args.candidate)[0] + ".rates.json"
    with open(detail_path, "w") as handle:
        json.dump(rows, handle, indent=1)
    print(f"detail: {detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
