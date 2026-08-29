#!/usr/bin/env python3
"""Production code is C/CUDA; Python lives in tests/, tools/, docs/, examples/.

The production tree ships to CUDA-only targets, so a new .py outside the
support directories is either dead weight or a build step nobody can run.
Three files predate this rule and are still load-bearing - the stage packers
under runtime/pack/, which tests load by path, and the ds4 eval comparator -
so they are whitelisted by exact path. Anything else fails.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWED_DIRS = {"tests", "tools", "docs", "examples"}
# generated or local-only directories, not production
SKIP_DIRS = {".git", ".audit", "build", "__pycache__", ".pytest_cache",
             "diagnostics"}
WHITELIST = {
    "qualification/ds4_eval/compare_runs.py",
    "runtime/pack/fp8_resident_pack.py",
    "runtime/pack/stage_pack.py",
    # pinned publisher modeling references: vendored semantics ground
    # truth for kernel ports (provenance + sha in the file's header
    # dir README) — not executed, never linked into serving.
    "model_contracts/references/modeling_qwen4_exp.py",
}


def main():
    failures = 0
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.parts[0] in ALLOWED_DIRS or str(rel) in WHITELIST:
            continue
        print(f"  FAIL {rel}: Python outside tests/, tools/, docs/, "
              f"examples/ and not whitelisted")
        failures += 1
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print("no Python in the production tree beyond the whitelisted packers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
