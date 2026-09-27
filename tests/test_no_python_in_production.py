#!/usr/bin/env python3
"""Production code is C/CUDA; Python lives in tests/, tools/, docs/, examples/.

The production tree ships to CUDA-only targets, so a new .py outside the
support directories is either dead weight or a build step nobody can run.
The ds4 eval comparator predates this rule, and the vendored publisher
modeling files under model_contracts/references/ are pinned by path and
sha256 from the contracts and name maps, so they are whitelisted by exact
path. Anything else fails.
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
    "model_contracts/references/modeling_ling_bailing_moe_v3.py",
    "model_contracts/references/modeling_muse_glimmer.py",
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
    print("no Python in the production tree beyond the whitelist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
