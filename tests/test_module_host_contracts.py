#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINNED = ["EXPERT_CODEC=fp8", "MODEL_REVISION=0", "CONTRACT_SHA256=0"]
ARGUMENTS = {"dsv41_flash": PINNED, "glm52": PINNED, "glm5_next": PINNED, "laguna": PINNED, "ling": PINNED, "qwen38_max": PINNED}


def main():
    modules = sorted(path.parent for path in ROOT.glob("modules/*_resident_decode_stage/Makefile"))
    failures = []
    for module in modules:
        family = module.name[:-len("_resident_decode_stage")]
        result = subprocess.run(["make", "-s", "-C", str(module), "contract"] + ARGUMENTS.get(family, []), capture_output=True, text=True)
        if result.returncode != 0:
            failures.append(family)
            print("FAIL %s module host sources do not compile:\n%s" % (family, (result.stdout + result.stderr).strip()[-2000:]))
    if len(modules) < 10:
        failures.append("discovery")
        print("FAIL found only %d module Makefiles" % len(modules))
    if failures:
        return 1
    print("PASS module host contracts: %d resident decode stage modules compile their host sources" % len(modules))
    return 0


if __name__ == "__main__":
    sys.exit(main())
