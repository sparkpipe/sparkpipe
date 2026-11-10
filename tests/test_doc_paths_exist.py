#!/usr/bin/env python3
"""Every repository path that a maintained document names in backticks exists.

A tool or module deleted from the tree leaves its documentation behind unless
something checks it. This gate reads the documents in DOCUMENTS, takes each
backticked token that starts with a tracked top-level directory, and fails
when that path is not in the tree.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCUMENTS = [
    "README.md",
    "docs/ROADMAP.md",
    "docs/TOOL_REFERENCE.md",
]

TOKEN = re.compile(r"`([^`\s]+)`")
HEADING = re.compile(r"^#{2,4} `?([\w.-]+\.(?:py|sh|c|cu|cuh|h|metal|json))`?\s*$", re.M)
PLACEHOLDER = re.compile(r"[<>*{}$~\[\]|]")


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split("\n")
    files = {line for line in out if line}
    dirs = set()
    for path in files:
        parts = path.split("/")
        for i in range(1, len(parts)):
            dirs.add("/".join(parts[:i]))
    return files, dirs


def candidates(text, roots):
    for token in TOKEN.findall(text):
        if PLACEHOLDER.search(token) or "/" not in token:
            continue
        path = re.sub(r":[0-9][0-9,-]*$", "", token).rstrip("/")
        if path.startswith("./"):
            path = path[2:]
        if path.split("/")[0] not in roots:
            continue
        if path.startswith("build/") or "..." in path:
            continue
        yield token, path


def missing_paths(documents):
    files, dirs = tracked()
    roots = {d for d in dirs if "/" not in d}
    basenames = {os.path.basename(path) for path in files}
    failures = []
    for document in documents:
        text = open(os.path.join(ROOT, document), encoding="utf-8").read()
        for token, path in candidates(text, roots):
            if path not in files and path not in dirs:
                failures.append(f"{document}: `{token}`")
        for name in HEADING.findall(text):
            if name not in basenames:
                failures.append(f"{document}: heading {name}")
    return failures


def check_detects_missing():
    files, dirs = tracked()
    roots = {d for d in dirs if "/" not in d}
    sample = "`tools/no_such_tool_for_this_gate.sh` and `README.md`"
    found = [path for _, path in candidates(sample, roots)]
    assert found == ["tools/no_such_tool_for_this_gate.sh"], found
    assert found[0] not in files
    assert HEADING.findall("### gone_tool.cu\n\n## Section\n") == ["gone_tool.cu"]


def main():
    check_detects_missing()
    failures = missing_paths(DOCUMENTS)
    if failures:
        print("documented paths missing from the tree:")
        for line in failures:
            print("  " + line)
        return 1
    print(f"doc paths: every backticked path in {len(DOCUMENTS)} documents exists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
