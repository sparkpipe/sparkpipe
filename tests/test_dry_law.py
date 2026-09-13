"""Enforce model-neutral shared infrastructure.

A model token belongs only in its model family, model module, model tool, or
model test. Shared runtime, transport, serving, cache, and deployment code may
not name any model. There is no debt budget and therefore no place for a new
model-specific fallback to hide.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON = (
    "include/sparkpipe",
    "node",
    "ring",
    "serving",
    "api",
    "cache",
    "scheduler",
    "text",
    "src",
    "runtime",
    "deployment",
    "inference/stage",
    "inference/kernels",
)
MODEL_TOKEN = re.compile(
    r"glm(?:5[_-]?2|52)|kimi|(?:^|[^a-z0-9])k3(?:[^a-z0-9]|$)|"
    r"qwen|dsv4|deepseek|mimo25",
    re.IGNORECASE,
)


def main():
    failures = 0
    checked = 0
    for root in COMMON:
        for path in sorted((ROOT / root).rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            checked += 1
            relative = str(path.relative_to(ROOT))
            text = path.read_text(errors="surrogateescape")
            if MODEL_TOKEN.search(relative) or MODEL_TOKEN.search(text):
                print(f"  FAIL {relative}: model token in shared code")
                failures += 1
    print(f"common files checked {checked}")
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("\nPASS shared infrastructure is model-neutral")
    return 0


if __name__ == "__main__":
    sys.exit(main())
