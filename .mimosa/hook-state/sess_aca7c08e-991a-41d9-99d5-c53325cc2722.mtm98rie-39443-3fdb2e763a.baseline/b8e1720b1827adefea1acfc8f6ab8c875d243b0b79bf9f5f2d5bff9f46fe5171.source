#!/usr/bin/env python3
"""Every path the build names must exist.

`make -n` proves the makefile parses. It does not prove the paths inside it are
real: a target nobody asked for can name a deleted file for months. Before this
gate the tree named sixteen files that were not there - twelve sources moved by
the reorganisation, four tools deleted outright - and eighteen other gates were
green the whole time.

Checked here:
  * every source in sources.mk exists
  * every .c/.cu/.cuh/.h path written literally in the Makefile exists
  * every directory the Makefile runs a sub-make into either exists or is
    guarded by a `test -d` in the same recipe
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []


def report(kind, path, detail=""):
    FAILURES.append(f"  {kind:22s} {path} {detail}".rstrip())


def check_sources_mk():
    text = open(os.path.join(ROOT, "sources.mk")).read()
    text = text.replace("\\\n", " ")
    count = 0
    for line in text.split("\n"):
        match = re.match(r"\s*([A-Z0-9_]+)\s*:?=\s*(.*)", line)
        if not match:
            continue
        for source in match.group(2).split():
            if not source.endswith(".c"):
                continue
            count += 1
            if not os.path.exists(os.path.join(ROOT, source)):
                report("sources.mk missing", source, f"({match.group(1)})")
    return count


def build_files():
    files = ["Makefile", "sources.mk", "tools/build.sh"]
    for pattern in ("modules/*/Makefile", "modules/*.mk", "modules/*/*.mk"):
        files.extend(sorted(glob.glob(os.path.join(ROOT, pattern))))
    return [f if os.path.isabs(f) else os.path.join(ROOT, f) for f in files]


def build_text(path):
    """Lines that are build inputs. A comment is documentation and an echo is
    output; neither is a path make will look for, and both legitimately name
    files that were deleted on purpose."""
    kept = []
    for line in open(path, errors="replace").read().split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("#") or "echo " in line:
            continue
        kept.append(line)
    return "\n".join(kept)


PATH_RE = re.compile(
    r"(?<![\w/.$)-])((?:\.\./)*(?:[\w.@-]+/)*[\w.@-]+\.(?:cuh|cu|c|h))(?![\w.])")


def check_makefile_paths():
    seen = set()
    for makefile in build_files():
        if not os.path.exists(makefile):
            continue
        base = os.path.dirname(makefile)
        rel = os.path.relpath(makefile, ROOT)
        for path in PATH_RE.findall(build_text(makefile)):
            key = (rel, path)
            if key in seen:
                continue
            seen.add(key)
            here = os.path.normpath(os.path.join(base, path))
            there = os.path.normpath(os.path.join(ROOT, path))
            if not os.path.exists(here) and not os.path.exists(there):
                report("build path missing", path, f"({rel})")
    return len(seen)


def check_sub_makes():
    count = 0
    for makefile in build_files():
        if not os.path.exists(makefile):
            continue
        count += _sub_makes_in(makefile)
    return count


def _sub_makes_in(makefile):
    lines = open(makefile, errors="replace").read().split("\n")
    rel = os.path.relpath(makefile, ROOT)
    base = os.path.dirname(makefile)
    count = 0
    for index, line in enumerate(lines):
        match = re.search(r"\$\(MAKE\)\s+-C\s+([\w/.-]+)", line)
        if not match:
            continue
        directory = match.group(1)
        if "$" in directory:
            continue
        count += 1
        if os.path.isdir(os.path.join(base, directory)) or os.path.isdir(os.path.join(ROOT, directory)):
            continue
        window = "\n".join(lines[max(0, index - 3):index + 1])
        if f"test -d {directory}" not in window:
            report("sub-make unguarded", directory, f"({rel}:{index + 1})")
    return count


def main():
    sources = check_sources_mk()
    paths = check_makefile_paths()
    submakes = check_sub_makes()
    print(f"sources.mk entries   {sources}")
    print(f"build paths          {paths}")
    print(f"sub-make directories {submakes}")
    if FAILURES:
        print(f"\n{len(FAILURES)} path(s) named by the build do not exist:")
        print("\n".join(FAILURES))
        return 1
    print("\nevery path the build names exists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
