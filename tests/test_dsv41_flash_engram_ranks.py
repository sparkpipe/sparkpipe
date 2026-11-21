#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "dsv41_flash_engram_shard.py"
FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)


def load():
    spec = importlib.util.spec_from_file_location("dsv41_flash_engram_shard", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    tool = load()
    entries = dict(tool.ENTRIES)
    for ranks, sizes in ((16, None), (4, (51004553472, 51004553472, 51004553472, 51004552960))):
        tool.RANKS = ranks
        for layer, rows in entries.items():
            part = tool.rows_per_rank(rows)
            owned = [tool.rank_rows(rows, rank) for rank in range(ranks)]
            check(part == (rows + ranks - 1) // ranks, f"{ranks} ranks: a rank's part of layer {layer} is the table's rows over the ranks, rounded up")
            check(sum(owned) == rows and all(count == part for count in owned[:-1]) and 0 < owned[-1] <= part,
                  f"{ranks} ranks: the shards of layer {layer} hold every row once, the last one the remainder")
        if sizes is not None:
            got = tuple(tool.derive_layout(entries, rank)[1] for rank in range(ranks))
            check(got == sizes, f"{ranks} ranks: shard sizes {got} are not those cut on 2026-11-12 {sizes}")
        check(tool.contract_sha(entries) != "", f"{ranks} ranks: the shard contract has a digest")
    tool.RANKS = 16
    sixteen = tool.contract_sha(entries)
    tool.RANKS = 4
    check(tool.contract_sha(entries) != sixteen, "a four-rank cut carries its own contract digest")
    refused = subprocess.run([sys.executable, str(TOOL), "--ranks", "5", "selftest"], capture_output=True, text=True)
    check(refused.returncode != 0 and "16, 8 or 4" in refused.stderr, "a group of five ranks is refused with the sizes that exist")
    usage = subprocess.run([sys.executable, str(TOOL), "--ranks", "4"], capture_output=True, text=True)
    check(usage.returncode == 2 and "--ranks N" in usage.stdout, "the usage text names the rank option")
    if FAILURES:
        for failure in FAILURES:
            print("FAIL dsv41 engram ranks:", failure)
        return 1
    print("PASS dsv41 engram ranks: the shard tool cuts for 16 or 4 ranks with the stage's row partition, a four-rank shard has the size cut on the hub, and other rank counts are refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
