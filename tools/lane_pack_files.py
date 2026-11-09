#!/usr/bin/env python3
"""Print the weight files every production lane names, host by host.

Each record in deployment/lanes.json renders, through tools/stage_lane.py,
one config/adapter.json per rank whose stage_pack_path names the rank pack
under /home/<host>/sparkdata directly. Beside that pack the engine opens its
.experts manifest, its .sha256 digest and the files the model's MODELS entry
lists as pack_siblings. Every other /home/<host>/sparkdata path a rendered
node file names is printed as well. One line per file:

    <host> <record> <role> <path>

with role pack, experts, digest, sibling or named. The tool renders the
records locally and reads nothing on the fleet, so pruning and audits work
from names and never follow a link.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_lane

ROLES = ("pack", "experts", "digest")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def lane_files(record_id: str) -> list[tuple[str, str, str, str]]:
    args = stage_lane.record_arguments(record_id)
    model = stage_lane.MODELS[args.model]
    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        stage_lane.write_lane(args, directory)
        deployment = json.loads((directory / "deployment.json").read_text())
        for rank, host in enumerate(stage_lane.HOSTS):
            adapter = json.loads((directory / f"adapter.{host}.json").read_text())
            files = stage_lane.rank_weight_files(model, rank)
            if adapter["stage_pack_path"] != files[0]:
                raise SystemExit(f"{record_id}: {host} adapter.json names {adapter['stage_pack_path']}, not the rank pack {files[0]}")
            rows += [(host, record_id, ROLES[index] if index < len(ROLES) else "sibling", path) for index, path in enumerate(files)]
            prefix = f"/home/{host}/sparkdata/"
            named = sorted({text for text in strings([adapter, deployment]) if text.startswith(prefix)} - set(files))
            rows += [(host, record_id, "named", path) for path in named]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("records", nargs="*", help="lane records to list; every record by default")
    parser.add_argument("--host", choices=stage_lane.HOSTS, help="print only this host's files")
    args = parser.parse_args()
    known = stage_lane.lane_records()["lanes"]
    unknown = sorted(set(args.records) - set(known))
    if unknown:
        raise SystemExit(f"{stage_lane.LANES} has no lane record {', '.join(unknown)}; known: {', '.join(sorted(known))}")
    rows = [row for record_id in (args.records or sorted(known)) for row in lane_files(record_id)]
    for row in sorted(rows):
        if args.host in (None, row[0]):
            print(" ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
