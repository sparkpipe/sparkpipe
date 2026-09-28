#!/usr/bin/env python3
"""Compare two per-row hash traces and report the first divergence.

A trace is a tab-separated file whose first line names the columns
wave, step, layer, site, row, sequence, position and hash. A row of a trace is
the 64-bit hash of one site buffer for one execution row. Two traces are joined
on (step, layer, site, sequence, position): the wave composition and the row
index inside the wave are what may differ, and the hash may not.

Exit status: 0 when every joined entry is equal, 1 when at least one entry
diverges, 2 when the traces are malformed or do not cover the same entries.

usage:
  row_hash_compare.py REFERENCE.tsv CANDIDATE.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

COLUMNS = ("wave", "step", "layer", "site", "row", "sequence", "position", "hash")


class TraceError(Exception):
    pass


def load(path: Path):
    lines = path.read_text().splitlines()
    if not lines or tuple(lines[0].split("\t")) != COLUMNS:
        raise TraceError(f"{path}: the first line must be {' '.join(COLUMNS)}")
    entries = {}
    order = {}
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(COLUMNS):
            raise TraceError(f"{path}:{number}: expected {len(COLUMNS)} fields, found {len(fields)}")
        wave, step, layer, site, row, sequence, position, value = fields
        try:
            key = (int(step), int(layer), site, int(sequence), int(position))
            entry = (wave, int(row), int(value, 16))
        except ValueError as error:
            raise TraceError(f"{path}:{number}: {error}") from error
        if key in entries:
            raise TraceError(f"{path}:{number}: duplicate entry step={key[0]} layer={key[1]} site={key[2]} sequence={key[3]} position={key[4]}")
        entries[key] = entry
        order.setdefault(site, len(order))
    return entries, order


def compare(reference: Path, candidate: Path, out=sys.stdout) -> int:
    try:
        left, order = load(reference)
        right, _ = load(candidate)
    except (OSError, TraceError) as error:
        print(f"ROWHASH-ERROR {error}", file=out)
        return 2
    missing = sorted(set(left) ^ set(right), key=lambda key: (key[0], key[1], order.get(key[2], len(order)), key[3]))
    if missing:
        key = missing[0]
        side = "candidate" if key in left else "reference"
        print(f"ROWHASH-INCOMPLETE entries_unmatched={len(missing)} first_missing_in={side} step={key[0]} layer={key[1]} site={key[2]} sequence={key[3]} position={key[4]}", file=out)
        return 2
    diverged = sorted((key for key in left if left[key][2] != right[key][2]), key=lambda key: (key[0], key[1], order[key[2]], key[3]))
    sites = {}
    for key in left:
        count = sites.setdefault(key[2], [0, 0, None])
        count[0 if left[key][2] == right[key][2] else 1] += 1
    for key in diverged:
        if sites[key[2]][2] is None:
            sites[key[2]][2] = key[1]
    for site in sorted(sites, key=lambda site: order[site]):
        equal, differ, first = sites[site]
        print(f"ROWHASH-SITE site={site} equal={equal} diverged={differ} first_layer={'-' if first is None else first}", file=out)
    if not diverged:
        print(f"ROWHASH-EQUAL entries={len(left)} sites={len(sites)}", file=out)
        return 0
    key = diverged[0]
    print(f"ROWHASH-FIRST-DIVERGENCE step={key[0]} layer={key[1]} site={key[2]} sequence={key[3]} position={key[4]} reference_wave={left[key][0]} reference_row={left[key][1]} candidate_wave={right[key][0]} candidate_row={right[key][1]} diverged={len(diverged)} entries={len(left)}", file=out)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    arguments = parser.parse_args()
    return compare(arguments.reference, arguments.candidate)


if __name__ == "__main__":
    sys.exit(main())
