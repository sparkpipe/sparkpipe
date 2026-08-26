#!/usr/bin/env python3
"""Require room for a 32K prompt plus the benchmark output budget."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_POSITIONS = 32768 + 256


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    for rank in range(4):
        path = f"examples/deployments/qwen36_tp4_rank{rank}.json"
        assert load(path)["max_sequence_positions"] >= REQUIRED_POSITIONS, path
    dsv4 = load("tools/devcycle/templates/dsv4_flash_tp4_stage.template.json")
    assert dsv4["max_sequence_positions"] >= REQUIRED_POSITIONS
    makefile = (ROOT / "modules/dsv4_resident_decode_stage/Makefile").read_text()
    match = re.search(r"^MAX_SEQUENCE_POSITIONS\s*\?=\s*(\d+)\s*$", makefile, re.MULTILINE)
    assert match and int(match.group(1)) >= REQUIRED_POSITIONS
    print(f"PASS: Qwen TP4 and DSV4 admit {REQUIRED_POSITIONS} positions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
