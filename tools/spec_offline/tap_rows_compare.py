#!/usr/bin/env python3
"""Compare two tap recordings of the same requests, made by different adapters or tap plans.

Requests are paired in the order they first appear (the second recording may hold fewer). For every
position both recordings carry, the input token and the rows of the entries both name must be the
same bits. Prints one TAPROWS line: rows compared, rows mismatched, and positions only the first
recording carries (a tail-limited recording must carry no prefill row the full one lacks).

--tolerance r compares recordings whose arithmetic differs (other wave shapes): a row matches when
every shared entry's relative L2 distance is at most r, positions after a request's first token
difference are skipped (counted as diverged), and a TAPERR line per entry gives the median, the
99th percentile and the largest relative distance. A misplaced or stale row is far above r; a
different summation order is far below it.
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tapstream import frames  # noqa: E402


def load(path):
    hello, requests = None, OrderedDict()
    for hello, frame in frames(path):
        key = (frame["control_generation"], frame["request_id"], frame["sequence_id"])
        item = requests.setdefault(key, {})
        for k in range(frame["rows"]):
            item[frame["first_position"] + k] = (frame["tokens"][k], frame["offset"] + k * frame["row_bytes"], frame["flags"])
    return hello, requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--tolerance", type=float)
    args = parser.parse_args()
    first_path, second_path = args.first, args.second
    first_hello, first = load(first_path)
    second_hello, second = load(second_path)
    if first_hello["hidden"] != second_hello["hidden"]:
        raise SystemExit("recordings differ in hidden size")
    shared = [name for name in first_hello["entries"] if name in second_hello["entries"]]
    hidden = first_hello["hidden"]
    first_columns = [first_hello["entries"].index(name) for name in shared]
    second_columns = [second_hello["entries"].index(name) for name in shared]
    first_data = np.memmap(first_path, dtype=np.uint16, mode="r")
    second_data = np.memmap(second_path, dtype=np.uint16, mode="r")
    first_count, second_count = len(first_hello["entries"]), len(second_hello["entries"])
    compared = mismatched = token_mismatched = extra = diverged = 0
    errors = [[] for _ in shared]
    for (first_key, first_rows), (second_key, second_rows) in zip(first.items(), second.items()):
        split = None
        if args.tolerance is not None:
            split = min((p for p, (token, _, _) in first_rows.items() if p in second_rows and second_rows[p][0] != token), default=None)
        for position, (token, offset, flags) in first_rows.items():
            other = second_rows.get(position)
            if other is None:
                extra += 1
                continue
            if split is not None and position >= split:
                diverged += 1
                continue
            if other[0] != token:
                token_mismatched += 1
            a = first_data[offset // 2: offset // 2 + first_count * hidden].reshape(first_count, hidden)[first_columns]
            b = second_data[other[1] // 2: other[1] // 2 + second_count * hidden].reshape(second_count, hidden)[second_columns]
            compared += 1
            if args.tolerance is None:
                mismatched += int(not np.array_equal(a, b))
                continue
            x = (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            y = (b.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            distance = np.linalg.norm(x - y, axis=1) / np.maximum(np.linalg.norm(y, axis=1), 1e-30)
            for column, value in enumerate(distance):
                errors[column].append(float(value))
            mismatched += int(not np.all(distance <= args.tolerance))
    tolerance = f" tolerance {args.tolerance} diverged {diverged}" if args.tolerance is not None else ""
    print(f"TAPROWS compared {compared} rows over {min(len(first), len(second))} paired requests of {len(first)} and {len(second)}; mismatched {mismatched} "
          f"token_mismatched {token_mismatched} only_in_first {extra} entries {shared}{tolerance}")
    if args.tolerance is not None:
        for name, values in zip(shared, errors):
            if values:
                v = np.array(values)
                print(f"TAPERR {name} rows {len(v)} median {np.median(v):.3g} p99 {np.percentile(v, 99):.3g} max {v.max():.3g}")


if __name__ == "__main__":
    main()
