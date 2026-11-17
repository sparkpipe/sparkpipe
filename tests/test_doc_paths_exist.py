#!/usr/bin/env python3
"""Every repository path that a maintained document names in backticks exists.

A tool or module deleted from the tree leaves its documentation behind unless
something checks it. This gate reads the documents in DOCUMENTS, takes each
backticked token that starts with a top-level directory, and each section
titled after a tool file, and fails when that path or tool is not in the tree.
"""
import glob
import os
import re
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


def top_directories():
    return {name for name in os.listdir(ROOT)
            if os.path.isdir(os.path.join(ROOT, name)) and not name.startswith(".")}


def tool_exists(name):
    pattern = os.path.join(ROOT, "tools", "**", name)
    return bool(glob.glob(pattern, recursive=True))


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
    roots = top_directories()
    failures = []
    for document in documents:
        text = open(os.path.join(ROOT, document), encoding="utf-8").read()
        for token, path in candidates(text, roots):
            if not os.path.exists(os.path.join(ROOT, path)):
                failures.append(f"{document}: `{token}`")
        for name in HEADING.findall(text):
            if not tool_exists(name):
                failures.append(f"{document}: heading {name}")
    return failures


def check_detects_missing():
    sample = "`tools/no_such_tool_for_this_gate.sh` and `README.md`"
    found = [path for _, path in candidates(sample, top_directories())]
    assert found == ["tools/no_such_tool_for_this_gate.sh"], found
    assert not os.path.exists(os.path.join(ROOT, found[0]))
    assert HEADING.findall("### gone_tool.cu\n\n## Section\n") == ["gone_tool.cu"]
    assert not tool_exists("gone_tool.cu")
    assert tool_exists("hy4_fp8_rung.cu")


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
