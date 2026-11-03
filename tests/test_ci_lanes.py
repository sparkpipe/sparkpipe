#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ci_lanes

LANE_TOKENS = re.compile(r"""(?<![\w.-])site(?:/|["']\s*[,)/])|sparkpipe_pay|site_terms|site_kr_networks|brand_logo|hyor-io\.conf|hyor\.io|sparkpipe-ai-ssl|sparkpipe-lang\.conf|sparkpipe-pay\.|rspamd-(?:dkim_signing|actions|redis)|postfix-ma(?:in|ster)\.cf|dovecot-(?:10-auth|99-sparkpipe)|certbot-reload-mail|PAYMENTS\.md|LITELLM_FRONTEND|docs/brand""")
WHOLE_TREE = re.compile(r"""\bROOT\s*\.rglob\(|os\.walk\(\s*ROOT\b|["']ls-files["']|\(\s*["']tests["']\s*,\s*["']tools["']""")


def check_classification(failures):
    cases = [
        (["site/buy.html", "site/terms/agreement-en.md", "tools/sparkpipe_pay.py", "tests/test_pay.py", "docs/PAYMENTS.md"], False),
        (["deployment/vps/hyor.io/index.html", "deployment/vps/sparkpipe-pay.service", "docs/brand/hyor-ai/README.md"], False),
        (["deployment/vps/postfix-main.cf", "deployment/vps/dovecot-99-sparkpipe.conf", "deployment/vps/certbot-reload-mail-services.sh", "docs/VPS_MAIL.md"], False),
        ([], False),
        (["site/buy.html", "inference/kernels/x.cu"], True),
        (["README.md"], False),
        (["TECHDEBT.md", "docs/archive/K3_PACK_FORMAT_V2.md", "site/buy.html"], False),
        (["README.md", "inference/kernels/x.cu"], True),
        (["Makefile"], True),
        (["deployment/vps/chat-local/index.html"], True),
        (["deployment/vps/sparkpipe-public-api.conf"], True),
        (["tools/ci_lanes.py"], False),
        (["tests/test_ci_lanes.py", "tools/ci_lanes.py"], False),
        (["LICENSE", "LICENSE-APACHE-2.0", "LICENSE-PERMISSION.md", "NOTICE", "CLA.md", "CONTRIBUTING.md"], False),
        ([".gitignore", ".gitattributes", ".updaterepo-policy", ".github/pull_request_template.md", ".github/ISSUE_TEMPLATE/bug.md", ".github/CODEOWNERS"], False),
        (["docs/images/fleet.svg", "docs/WORKING_STATE.md", "SPEC.md"], False),
        (["LICENSE", "site/index.html", "README.md"], False),
        (["LICENSE", "Makefile"], True),
        (["tools/ci_lanes.py", "inference/kernels/x.cu"], True),
        (["tools/serving/chat_frontend.py"], True),
        (["model-families/k3/README.md"], False),
        (["newdir/x.c"], True),
        ([".github/workflows/cuda13-sm121a-compile.yml"], True),
        (["website/site/x.html"], True),
        (None, True),
    ]
    site = ci_lanes.light_tests(["site/buy.html"])
    if "tests/test_site.py" not in site or "tests/test_ci_lanes.py" not in site or "tests/test_status_truth.py" in site:
        failures.append(f"a site change runs the wrong light tests: {site}")
    repo = ci_lanes.light_tests(["LICENSE", "NOTICE", "tools/ci_lanes.py"])
    for name in ("tests/test_ci_lanes.py", "tests/test_third_party_notices.py", "tests/test_package_manifest.py"):
        if name not in repo:
            failures.append(f"a repository-file change does not run {name}")
    if ci_lanes.light_tests(["docs/images/fleet.svg"]) != ci_lanes.light_tests(["docs/WORKING_STATE.md"]):
        failures.append("a non-Markdown file under docs/ does not run the Markdown readers")
    readme = ci_lanes.light_tests(["README.md"])
    for name in ("tests/test_status_truth.py", "tests/test_lane_records.py", "tests/test_open_webui_github_tool.py", "tests/test_site.py"):
        if name not in readme:
            failures.append(f"a README change does not run {name}")
    for paths, engine in cases:
        if ci_lanes.engine_needed(paths) != engine:
            failures.append(f"{paths}: engine lane {'skipped' if engine else 'run'}")
    if ci_lanes.changed("0" * 40, "HEAD") is not None or ci_lanes.changed("", "HEAD") is not None:
        failures.append("a push without a previous commit does not fall back to the engine lane")


def check_every_file_has_a_lane(failures):
    run = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True)
    if run.returncode != 0:
        failures.append(f"git ls-files exited {run.returncode}")
        return 0
    files = [entry for entry in run.stdout.decode("utf-8", "surrogateescape").split("\0") if entry]
    loose = sorted({path.split("/")[0] for path in files if ci_lanes.lane(path) == "unclassified"})
    for top in loose:
        failures.append(f"{top}: no lane; add it to ENGINE_PATHS if the engine build or an engine test reads it, otherwise to LIGHT_PATHS")
    return len(files)


def check_fast_tests_exist(failures):
    for name in ci_lanes.FAST_TESTS:
        if not (ROOT / name).is_file():
            failures.append(f"{name}: in the fast lane but missing")


def check_engine_tests_ignore_site(failures):
    fast = set(ci_lanes.light_tests([]))
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
        text = path.read_text(encoding="utf-8", errors="replace")
        found = LANE_TOKENS.search(text)
        if found:
            failures.append(f"{path.relative_to(ROOT)} builds from site-lane file {found.group(0)!r}")
        if path.name == "Makefile" or path.suffix == ".mk":
            found = re.search(r"[A-Za-z0-9_./-]+\.md\b", text)
            if found:
                failures.append(f"{path.relative_to(ROOT)} builds from Markdown {found.group(0)!r}: Markdown changes skip the engine lane")


def main():
    failures = []
    check_classification(failures)
    tracked = check_every_file_has_a_lane(failures)
    check_fast_tests_exist(failures)
    check_engine_tests_ignore_site(failures)
    check_build_ignores_site(failures)
    for failure in failures:
        print(f"  FAIL {failure}")
    if failures:
        return 1
    print(f"ci lanes: only the {len(ci_lanes.ENGINE_PATHS)} engine paths run the engine lane, and each of the {tracked} tracked files has a lane; {len(ci_lanes.SITE_PATHS)} site paths, {len(ci_lanes.LIGHT_PATHS)} repository paths and Markdown skip it; no engine test or build rule reads the site paths; {len(ci_lanes.light_tests([]))} light tests cover the site and every whole-tree scan, and Markdown changes add the {len(ci_lanes.markdown_tests())} tests that can read Markdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
