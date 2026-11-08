#!/usr/bin/env python3
"""The DSV4.1 engram row-shard cutter round-trips a fabricated checkpoint at 16 ranks (the TP16 lane) and
at 4 ranks (a TP4 x PP4 prefill stage): cut, then re-derive every shard's layout, boundary rows, planes,
manifest and digest against the checkpoint."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/dsv41_flash_engram_shard.py"


def main() -> int:
    for prefix in ([], ["--ranks", "4"]):
        result = subprocess.run([sys.executable, str(TOOL), *prefix, "selftest"], capture_output=True, text=True, timeout=600)
        print(result.stdout + result.stderr, end="")
        if result.returncode != 0 or "selftest PASS" not in result.stdout:
            print(f"FAIL dsv41 engram shard selftest {' '.join(prefix) or '--ranks 16'}")
            return 1
    refused = subprocess.run([sys.executable, str(TOOL), "--ranks", "3", "selftest"], capture_output=True, text=True, timeout=60)
    if refused.returncode == 0:
        print("FAIL dsv41 engram shard accepted 3 ranks")
        return 1
    print("PASS dsv41 engram shard: 16- and 4-rank cuts verify, 3 ranks refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
