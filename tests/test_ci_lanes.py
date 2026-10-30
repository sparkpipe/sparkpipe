#!/usr/bin/env python3
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ci_lanes

LANE_TOKENS = re.compile(r"""(?<![\w.-])site(?:/|["']\s*[,)/])|sparkpipe_pay|site_terms|site_kr_networks|brand_logo|hyor-io\.conf|hyor\.io|sparkpipe-ai-ssl|sparkpipe-lang\.conf|sparkpipe-pay\.|rspamd-dkim_signing|PAYMENTS\.md|LITELLM_FRONTEND|docs/brand""")
WHOLE_TREE = re.compile(r"""\bROOT\s*\.rglob\(|os\.walk\(\s*ROOT\b|["']ls-files["']|\(\s*["']tests["']\s*,\s*["']tools["']""")


def check_classification(failures):
    cases = [
        (["site/buy.html", "site/terms/hourly-en.md", "tools/sparkpipe_pay.py", "tests/test_pay.py", "docs/PAYMENTS.md"], False),
        (["deployment/vps/hyor.io/index.html", "deployment/vps/sparkpipe-pay.service", "docs/brand/hyor-ai/README.md"], False),
        ([], False),
        (["site/buy.html", "inference/kernels/x.cu"], True),
        (["README.md"], True),
        (["TECHDEBT.md"], True),
        (["docs/archive/K3_PACK_FORMAT_V2.md"], True),
        (["deployment/vps/chat-local/index.html"], True),
        (["deployment/vps/sparkpipe-public-api.conf"], True),
        (["tools/ci_lanes.py"], True),
        ([".github/workflows/cuda13-sm121a-compile.yml"], True),
        (["website/site/x.html"], True),
        (None, True),
    ]
    if "tests/test_site.py" not in ci_lanes.fast_tests() or "tests/test_ci_lanes.py" not in ci_lanes.fast_tests():
        failures.append(f"the fast lane does not run the site tests and this test: {ci_lanes.fast_tests()}")
    for paths, engine in cases:
        if ci_lanes.engine_needed(paths) != engine:
            failures.append(f"{paths}: engine lane {'skipped' if engine else 'run'}")
    if ci_lanes.changed("0" * 40, "HEAD") is not None or ci_lanes.changed("", "HEAD") is not None:
        failures.append("a push without a previous commit does not fall back to the engine lane")


def check_fast_tests_exist(failures):
    for name in ci_lanes.FAST_TESTS:
        if not (ROOT / name).is_file():
            failures.append(f"{name}: in the fast lane but missing")


def check_engine_tests_ignore_site(failures):
    fast = set(ci_lanes.fast_tests())
    for path in sorted(list((ROOT / "tests").glob("*.py")) + list((ROOT / "tests").glob("*.sh"))):
        name = path.relative_to(ROOT).as_posix()
        if name in fast:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        found = LANE_TOKENS.search(source)
        if found:
            failures.append(f"{name} reads site-lane file {found.group(0)!r}: add it to FAST_TESTS or take the file out of SITE_PATHS")
        found = WHOLE_TREE.search(source)
        if found:
            failures.append(f"{name} scans the whole tree ({found.group(0)!r}): add it to FAST_TESTS")


def check_build_ignores_site(failures):
    build = [ROOT / "Makefile"] + sorted(ROOT.glob("*.mk")) + sorted((ROOT / "tools").glob("*.sh")) + sorted((ROOT / "tools" / "ci").glob("*"))
    for path in build:
        if not path.is_file():
            continue
        found = LANE_TOKENS.search(path.read_text(encoding="utf-8", errors="replace"))
        if found:
            failures.append(f"{path.relative_to(ROOT)} builds from site-lane file {found.group(0)!r}")


def main():
    failures = []
    check_classification(failures)
    check_fast_tests_exist(failures)
    check_engine_tests_ignore_site(failures)
    check_build_ignores_site(failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print(f"ci lanes: {len(ci_lanes.SITE_PATHS)} site paths skip the engine lane; no engine test or build rule reads them; {len(ci_lanes.fast_tests())} fast tests cover the site and every whole-tree scan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
