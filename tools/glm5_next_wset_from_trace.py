#!/usr/bin/env python3
"""Build a GLM 5.3 Flash expert working set (.wset) from engine route traces.

Input: the per-rank files written by SPARK_GLM5_NEXT_ROUTE_TRACE=<prefix>
(<prefix>.stageNN.rankNN.trace, one G5N-ROUTE line per routed layer of every
lazy eager step). Every rank of a TP stage routes the same keys, so the ranks
must agree; a disagreement fails the run.

Output: a .wset of little-endian (layer u32, expert u32) pairs, deduplicated
and sorted, plus a JSON coverage report: the route union per layer, the
per-rank route digest (compare two runs with --expect-route-sha256), and,
with --cap-keys, an LFU selection that keeps at least one key per routed
layer and the fraction of recorded steps it would serve with every key held.
"""
import argparse
import hashlib
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

EXPERTS = 288
LINE = re.compile(r"^G5N-ROUTE rank=(\d+) rows=(\d+) pos=(\d+) layer=(\d+) n=(\d+) e=([0-9,]*)$")


def fail(message):
    raise SystemExit(f"glm5_next_wset_from_trace: FAIL: {message}")


def parse_line(path, number, text):
    match = LINE.match(text)
    if match is None:
        fail(f"{path}:{number}: not a G5N-ROUTE record")
    rank, rows, pos, layer, count = (int(match.group(index)) for index in range(1, 6))
    experts = tuple(int(value) for value in match.group(6).split(",")) if match.group(6) else ()
    if len(experts) != count or count == 0:
        fail(f"{path}:{number}: n={count} but {len(experts)} experts")
    if any(expert >= EXPERTS for expert in experts) or list(experts) != sorted(set(experts)):
        fail(f"{path}:{number}: experts must be unique, ascending and below {EXPERTS}")
    return rank, (rows, pos, layer, experts)


def load_traces(paths):
    ranks = {}
    for path in paths:
        for number, text in enumerate(Path(path).read_text().splitlines(), 1):
            rank, record = parse_line(path, number, text)
            ranks.setdefault(rank, []).append(record)
    if not ranks:
        fail("the traces hold no route records")
    return ranks


def route_digest(records):
    lines = sorted(f"{rows} {pos} {layer} {','.join(map(str, experts))}" for rows, pos, layer, experts in records)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def split_steps(records):
    steps, current, key = [], set(), None
    for rows, pos, layer, experts in records:
        if key is not None and (rows, pos) != key:
            steps.append(current)
            current = set()
        key = (rows, pos)
        current.update((layer, expert) for expert in experts)
    steps.append(current)
    return steps


def select_keys(frequency, layers, cap):
    ranked = sorted(frequency, key=lambda key: (-frequency[key], key))
    anchors = {min((key for key in ranked if key[0] == layer), key=lambda key: (-frequency[key], key)) for layer in layers}
    if cap < len(anchors):
        fail(f"cap {cap} is below the {len(anchors)} routed layers; every layer needs a held expert")
    selected = set(anchors)
    for key in ranked:
        if len(selected) >= cap:
            break
        selected.add(key)
    return selected


def build(arguments):
    ranks = load_traces(arguments.traces)
    digests = {rank: route_digest(records) for rank, records in sorted(ranks.items())}
    if len(set(digests.values())) != 1:
        fail(f"ranks route differently: {json.dumps(digests)}")
    digest = next(iter(digests.values()))
    if arguments.expect_route_sha256 and digest != arguments.expect_route_sha256:
        fail(f"route digest {digest} differs from the expected {arguments.expect_route_sha256}")
    records = ranks[min(ranks)]
    frequency = Counter((layer, expert) for _, _, layer, experts in records for expert in experts)
    layers = sorted({layer for layer, _ in frequency})
    selected = set(frequency) if arguments.cap_keys is None else select_keys(frequency, layers, arguments.cap_keys)
    steps = split_steps(records)
    payload = b"".join(struct.pack("<II", layer, expert) for layer, expert in sorted(selected))
    Path(arguments.output).write_bytes(payload)
    return {
        "traces": [str(path) for path in arguments.traces],
        "ranks": sorted(ranks),
        "records": len(records),
        "steps": len(steps),
        "route_sha256": digest,
        "union_keys": len(frequency),
        "union_per_layer": {str(layer): sum(1 for key in frequency if key[0] == layer) for layer in layers},
        "cap_keys": arguments.cap_keys,
        "wset": str(arguments.output),
        "wset_keys": len(selected),
        "wset_sha256": hashlib.sha256(payload).hexdigest(),
        "all_hit_steps": sum(1 for step in steps if step <= selected),
        "all_hit_fraction": sum(1 for step in steps if step <= selected) / len(steps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cap-keys", type=int)
    parser.add_argument("--expect-route-sha256")
    arguments = parser.parse_args()
    if arguments.cap_keys is not None and arguments.cap_keys <= 0:
        fail("--cap-keys must be positive")
    report = json.dumps(build(arguments), indent=2, sort_keys=True)
    if arguments.report:
        arguments.report.write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
