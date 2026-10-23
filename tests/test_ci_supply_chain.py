#!/usr/bin/env python3
"""CI may only run third-party code that is pinned to an exact, verifiable artifact.

Every GitHub workflow must:
- reference each action by a full 40-hex commit SHA, never a tag or branch, because tags can be moved
  to attacker code;
- reference each container image by sha256 digest;
- install Python packages only with --require-hashes and --no-deps from a requirements file in which
  every line names one exact version and its sha256, so no transitive package resolves to whatever
  is newest on the index at run time;
- grant the job token read-only repository access at the top level and never write-all.

Each rule is also checked against a workflow that breaks it, so a checker that stopped detecting a
violation would fail here too.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
USES = re.compile(r"^\s*(?:-\s+)?uses:\s*([^\s#]+)")
IMAGE = re.compile(r"^\s*(?:-\s+)?image:\s*([^\s#]+)")
PINNED_ACTION = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}$")
PINNED_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
PIP_INSTALL = re.compile(r"\bpip3?\s+install\b")
REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s=;]+)\s+--hash=sha256:[0-9a-f]{64}$")


def logical_lines(text):
    merged, pending = [], ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        merged.append(pending + line)
        pending = ""
    if pending:
        merged.append(pending)
    return merged


def requirement_violations(path):
    if not path.is_file():
        return ["requirements file %s does not exist" % path.relative_to(ROOT)]
    problems, seen = [], 0
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        seen += 1
        if REQUIREMENT.match(line) is None:
            problems.append("%s:%d is not 'name==version --hash=sha256:<64 hex>': %s" % (path.relative_to(ROOT), number, line))
    if seen == 0:
        problems.append("requirements file %s is empty" % path.relative_to(ROOT))
    return problems


def workflow_violations(name, text, root=ROOT):
    problems = []
    lines = text.splitlines()
    top = [line for line in lines if line and not line[0].isspace()]
    if "permissions:" not in top:
        problems.append("%s: no top-level permissions block" % name)
    else:
        start = lines.index("permissions:")
        block = []
        for line in lines[start + 1:]:
            if line and not line[0].isspace():
                break
            block.append(line.strip())
        if "contents: read" not in block:
            problems.append("%s: the top-level permissions do not grant contents: read" % name)
        if any(entry.endswith(": write") for entry in block):
            problems.append("%s: the top-level permissions grant write access" % name)
    if re.search(r"permissions:\s*write-all", text):
        problems.append("%s: write-all permissions" % name)
    for number, line in enumerate(lines, 1):
        match = USES.match(line)
        if match and not match.group(1).startswith("./") and PINNED_ACTION.match(match.group(1)) is None:
            problems.append("%s:%d: action %s is not pinned to a 40-hex commit SHA" % (name, number, match.group(1)))
        match = IMAGE.match(line)
        if match and PINNED_IMAGE.match(match.group(1)) is None:
            problems.append("%s:%d: image %s is not pinned by sha256 digest" % (name, number, match.group(1)))
    for line in logical_lines(text):
        if PIP_INSTALL.search(line) is None:
            continue
        words = line.split()
        if "--require-hashes" not in words or "--no-deps" not in words:
            problems.append("%s: pip install without --require-hashes and --no-deps: %s" % (name, line.strip()))
        files = [words[i + 1] for i, word in enumerate(words[:-1]) if word in ("-r", "--requirement")]
        if not files:
            problems.append("%s: pip install without a hash-locked requirements file: %s" % (name, line.strip()))
        for listed in files:
            problems += requirement_violations(root / listed)
        named = [word for word in words if re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*(==|>=|<=|~=)", word)]
        if named:
            problems.append("%s: pip install names packages outside the hash lock: %s" % (name, " ".join(named)))
    return problems


BAD_WORKFLOWS = {
    "tag-pinned action": "permissions:\n  contents: read\njobs:\n  a:\n    steps:\n      - uses: actions/checkout@v4\n",
    "branch-pinned action": "permissions:\n  contents: read\njobs:\n  a:\n    steps:\n      - uses: actions/checkout@main\n",
    "tag-pinned image": "permissions:\n  contents: read\njobs:\n  a:\n    container:\n      image: nvidia/cuda:13.3.1-devel-ubuntu24.04\n",
    "unpinned pip": "permissions:\n  contents: read\njobs:\n  a:\n    steps:\n      - run: |\n          python3 -m pip install aiohttp==3.14.3\n",
    "pip without no-deps": "permissions:\n  contents: read\njobs:\n  a:\n    steps:\n      - run: python3 -m pip install --require-hashes -r tools/ci/host-tests-requirements.txt\n",
    "missing lock file": "permissions:\n  contents: read\njobs:\n  a:\n    steps:\n      - run: python3 -m pip install --require-hashes --no-deps -r tools/ci/absent.txt\n",
    "no permissions block": "jobs:\n  a:\n    steps:\n      - run: make\n",
    "write permission": "permissions:\n  contents: write\njobs:\n  a:\n    steps:\n      - run: make\n",
}


def main():
    failures = []
    for label, text in BAD_WORKFLOWS.items():
        if not workflow_violations(label, text):
            failures.append("the checker accepted a workflow with a %s" % label)
    good = ("permissions:\n  contents: read\njobs:\n  a:\n    container:\n      image: x/y@sha256:" + "a" * 64 +
            "\n    steps:\n      - uses: actions/checkout@" + "b" * 40 +
            "\n      - run: python3 -m pip install --no-deps --require-hashes -r tools/ci/host-tests-requirements.txt\n")
    accepted = workflow_violations("good", good)
    if accepted:
        failures.append("the checker rejected a fully pinned workflow: %s" % "; ".join(accepted))
    workflows = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    if not workflows:
        failures.append("no workflows found under .github/workflows")
    for path in workflows:
        failures += workflow_violations(str(path.relative_to(ROOT)), path.read_text())
    if failures:
        for failure in failures:
            print("FAIL", failure)
        return 1
    print("PASS every CI action, image and Python package is pinned to an exact artifact (%d workflows), and the job token is read-only" % len(workflows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
