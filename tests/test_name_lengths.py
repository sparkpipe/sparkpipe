#!/usr/bin/env python3
"""Every name in the repository is 64 characters or fewer: each component of every path that git lists, tracked or new and not
ignored, directories and files alike."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT = 64


def listed_files():
    run = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True)
    if run.returncode != 0:
        stderr = run.stderr.decode("utf-8", "replace").splitlines()
        return None, f"git ls-files exited {run.returncode}: {stderr[0] if stderr else ''}"
    entries = run.stdout.decode("utf-8", "surrogateescape").split("\0")
    return sorted({entry for entry in entries if entry}), None


def long_components(path):
    return [part for part in path.split("/") if len(part) > LIMIT]


def main():
    failures = 0
    planted = {
        "a/" + "d" * (LIMIT + 1) + "/file": ["d" * (LIMIT + 1)],
        "a/b/" + "f" * (LIMIT - 5) + ".jsonl": ["f" * (LIMIT - 5) + ".jsonl"],
        "d" * LIMIT + "/" + "f" * LIMIT: [],
    }
    for path, wanted in planted.items():
        if long_components(path) != wanted:
            print(f"  FAIL the check finds {long_components(path)} in a planted path, not {wanted}")
            failures += 1
    entries, error = listed_files()
    if error is not None:
        print(f"  FAIL file list: {error}")
        return 1
    reported = set()
    for path in entries:
        for part in long_components(path):
            prefix = path[:path.index(part) + len(part)]
            if prefix in reported:
                continue
            reported.add(prefix)
            print(f"  FAIL {prefix}: {len(part)} characters, over the {LIMIT}-character limit")
            failures += 1
    if failures:
        print(f"\nFAIL ({failures})")
        return 1
    print(f"PASS every path component of the {len(entries)} listed files is {LIMIT} characters or fewer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
