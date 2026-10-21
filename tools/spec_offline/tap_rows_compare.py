#!/usr/bin/env python3
"""Compare two tap recordings of the same requests, made by different adapters or tap plans.

Requests are paired in the order they first appear (the second recording may hold fewer). For every
position both recordings carry, the input token and the rows of the entries both name must be the
same bits. Prints one TAPROWS line: rows compared, rows mismatched, and positions only the first
recording carries (a tail-limited recording must carry no prefill row the full one lacks).
"""
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
    first_path, second_path = sys.argv[1], sys.argv[2]
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
    compared = mismatched = token_mismatched = extra = 0
    for (first_key, first_rows), (second_key, second_rows) in zip(first.items(), second.items()):
        for position, (token, offset, flags) in first_rows.items():
            other = second_rows.get(position)
            if other is None:
                extra += 1
                continue
            if other[0] != token:
                token_mismatched += 1
            a = first_data[offset // 2: offset // 2 + first_count * hidden].reshape(first_count, hidden)[first_columns]
            b = second_data[other[1] // 2: other[1] // 2 + second_count * hidden].reshape(second_count, hidden)[second_columns]
            compared += 1
            mismatched += int(not np.array_equal(a, b))
    print(f"TAPROWS compared {compared} rows over {min(len(first), len(second))} paired requests of {len(first)} and {len(second)}; mismatched {mismatched} "
          f"token_mismatched {token_mismatched} only_in_first {extra} entries {shared}")


if __name__ == "__main__":
    main()
