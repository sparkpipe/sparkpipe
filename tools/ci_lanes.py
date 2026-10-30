#!/usr/bin/env python3
import fnmatch
import re
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

MARKDOWN = re.compile(r"""["'/]docs["'/]|[A-Za-z0-9_./-]+\.md\b|\bREADME\b|\bTECHDEBT\b""")


def site_path(path):
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in SITE_PATHS)


def markdown_path(path):
    return path.endswith(".md") and not site_path(path)


def engine_needed(paths):
    return paths is None or not all(site_path(path) or markdown_path(path) for path in paths)


def tests_in(directory):
    return sorted(path for path in (ROOT / directory).iterdir() if path.is_file() and path.suffix in (".py", ".sh"))


def uses(text, module):
    return re.search(rf"\b(?:import|from)\s+{re.escape(module)}\b|\b{re.escape(module)}\.py\b", text)


def markdown_tests():
    tools = {path.stem: path.read_text(encoding="utf-8", errors="replace") for path in tests_in("tools")}
    readers = {name for name, text in tools.items() if MARKDOWN.search(text)}
    while True:
        more = {name for name, text in tools.items() if name not in readers and any(uses(text, module) for module in readers)}
        if not more:
            break
        readers |= more
    found = []
    for path in tests_in("tests"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if MARKDOWN.search(text) or any(uses(text, module) for module in readers):
            found.append(path.relative_to(ROOT).as_posix())
    return found


def light_tests(paths):
    chosen = [path.relative_to(ROOT).as_posix() for path in tests_in("tests") if site_path(path.relative_to(ROOT).as_posix())]
    chosen += list(FAST_TESTS)
    if paths and any(markdown_path(path) for path in paths):
        chosen += markdown_tests()
    return list(dict.fromkeys(chosen))


def changed(base, head):
    if not base or set(base) == {"0"}:
        return None
    run = subprocess.run(["git", "diff", "--name-only", f"{base}...{head}"], capture_output=True, text=True)
    if run.returncode != 0:
        return None
    return [line for line in run.stdout.splitlines() if line]


def main(argv):
    if argv[1:] == ["full"]:
        paths = None
    elif len(argv) == 3:
        paths = changed(argv[1], argv[2])
    else:
        print("usage: ci_lanes.py BASE HEAD | full", file=sys.stderr)
        return 2
    for path in paths or []:
        lane = "site" if site_path(path) else "markdown" if markdown_path(path) else "engine"
        print(f"{lane:8} {path}", file=sys.stderr)
    print(f"engine={'true' if engine_needed(paths) else 'false'}")
    print(f"build={'true' if paths and any(markdown_path(path) for path in paths) else 'false'}")
    print(f"tests={' '.join(light_tests(paths))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
