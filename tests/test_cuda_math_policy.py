#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_POLICY_FILES = (
    ROOT / "Makefile",
    ROOT / "modules" / "resident_decode_stage_rules.mk",
    ROOT / "modules" / "glm52_resident_decode_stage" / "Makefile",
    ROOT / "tools" / "build.sh",
)


def main() -> int:
    failures = []
    for path in BUILD_POLICY_FILES:
        text = path.read_text(encoding="utf-8")
        if "--use_fast_math" in text:
            failures.append(path.relative_to(ROOT).as_posix())
    if failures:
        for path in failures:
            print(f"  FAIL global fast-math policy remains in {path}")
        return 1
    print("CUDA build policy does not globally force approximate math")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
