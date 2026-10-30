#!/usr/bin/env python3
import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SITE_PATHS = (
    "site/*",
    "tools/sparkpipe_pay.py",
    "tools/site_terms.py",
    "tools/site_kr_networks.py",
    "tools/brand_logo.py",
    "tests/test_site.py",
    "tests/test_pay.py",
    "deployment/vps/hyor.io/*",
    "deployment/vps/hyor-io.conf",
    "deployment/vps/sparkpipe-ai-ssl.conf",
    "deployment/vps/sparkpipe-lang.conf",
    "deployment/vps/sparkpipe-pay.conf",
    "deployment/vps/sparkpipe-pay.env.example",
    "deployment/vps/sparkpipe-pay.service",
    "deployment/vps/rspamd-dkim_signing.conf",
    "docs/PAYMENTS.md",
    "docs/LITELLM_FRONTEND.md",
    "docs/brand/*",
)

FAST_TESTS = (
    "tests/test_ci_lanes.py",
    "tests/test_code_size.py",
    "tests/test_complexity_ceiling.py",
    "tests/test_memory_contracts.py",
    "tests/test_no_build_defaults.py",
    "tests/test_no_python_in_production.py",
    "tests/test_no_source_comments.py",
    "tests/test_python_syntax.py",
    "tests/test_third_party_notices.py",
    "tests/test_hardware_probe_coverage.py",
    "tests/test_dry_law.py",
    "tests/test_device_neutral.py",
    "tests/test_unlinked_components.py",
    "tests/test_sources_exist.py",
    "tests/test_package_manifest.py",
)


def site_path(path):
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in SITE_PATHS)


def fast_tests():
    site_tests = [path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "tests").iterdir()) if path.is_file() and site_path(path.relative_to(ROOT).as_posix())]
    return site_tests + list(FAST_TESTS)


def engine_needed(paths):
    return paths is None or not all(site_path(path) for path in paths)


def changed(base, head):
    if not base or set(base) == {"0"}:
        return None
    run = subprocess.run(["git", "diff", "--name-only", f"{base}...{head}"], capture_output=True, text=True)
    if run.returncode != 0:
        return None
    return [line for line in run.stdout.splitlines() if line]


def main(argv):
    if argv[1:] == ["fast-tests"]:
        print("\n".join(fast_tests()))
        return 0
    if len(argv) != 3:
        print("usage: ci_lanes.py BASE HEAD | fast-tests", file=sys.stderr)
        return 2
    paths = changed(argv[1], argv[2])
    if paths is not None:
        for path in paths:
            print(f"{'site  ' if site_path(path) else 'engine'} {path}", file=sys.stderr)
    print(f"engine={'true' if engine_needed(paths) else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
