#!/usr/bin/env python3
"""A bare make in a decode-stage module runs its contract check.

Every module Makefile that includes resident_decode_stage_rules.mk must
leave `all` (the contract check) as its default goal. A rule defined
before the rules file, such as a wrapper's adapter target, would
otherwise become the default and a bare make would build and link
something else.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZERO_SHA = "0" * 64
PROBE = "spark-default-goal: ; @echo SPARK_DEFAULT_GOAL=$(.DEFAULT_GOAL)"


def default_goal(directory: Path) -> str | None:
    for codec in ("fp8", "bf16", "nvfp4"):
        result = subprocess.run(
            ["make", "-s", "-C", str(directory), "--eval", PROBE, "--eval", ".DEFAULT_GOAL :=", "spark-default-goal",
             f"EXPERT_CODEC={codec}", "MODEL_REVISION=test", f"CONTRACT_SHA256={ZERO_SHA}", "CUDA_HOME=/nonexistent"],
            capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if line.startswith("SPARK_DEFAULT_GOAL="):
                return line.split("=", 1)[1]
    return None


def main() -> int:
    failures = []
    modules = sorted(path.parent for path in (ROOT / "modules").glob("*/Makefile") if "resident_decode_stage_rules.mk" in path.read_text(encoding="utf-8"))
    for directory in modules:
        goal = default_goal(directory)
        if goal != "all":
            failures.append(f"{directory.name}: default goal is {goal!r}, not 'all'")
        else:
            print(f"  ok {directory.name}")
    if not modules:
        failures.append("no module Makefile includes resident_decode_stage_rules.mk")
    if failures:
        print("\n".join("FAIL " + failure for failure in failures))
        return 1
    print(f"PASS a bare make runs the contract check in all {len(modules)} decode-stage modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
