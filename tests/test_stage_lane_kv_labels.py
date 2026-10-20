#!/usr/bin/env python3
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import stage_lane  # noqa: E402


def main():
    tags = {}
    for source in ROOT.glob("modules/*/source/*serving*.c"):
        for tag in re.findall(r'\.module_tag\s*=\s*"([^"]+)"', source.read_text()):
            tags[tag] = source.parent.parent.name
    failures = []
    for name, model in stage_lane.MODELS.items():
        if model["kv_label"] not in tags:
            failures.append(f"{name}: kv_label {model['kv_label']!r} is no adapter's module_tag, so the swap controller would never release its KV pool")
    for line in failures:
        print("FAILED: " + line)
    if failures:
        return 1
    print(f"PASS {len(stage_lane.MODELS)} lane KV labels name an adapter module_tag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
