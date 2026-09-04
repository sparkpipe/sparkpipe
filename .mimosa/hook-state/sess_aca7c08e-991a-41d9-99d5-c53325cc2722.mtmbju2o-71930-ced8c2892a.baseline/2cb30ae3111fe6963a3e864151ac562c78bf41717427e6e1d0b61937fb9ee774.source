#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for root_name in ("tests", "tools", "runtime"):
        root = repository / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
                compile(source, str(path), "exec")
            except Exception as error:
                failures.append(f"{path.relative_to(repository)}: {error}")
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print("PASS Python syntax inventory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
