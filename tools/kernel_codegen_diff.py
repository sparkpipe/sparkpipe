#!/usr/bin/env python3
import argparse
import hashlib
import re
import subprocess
import sys

CUOBJDUMP = "cuobjdump"
FUNCTION = re.compile(r"\s*Function : (\S+)")
RESOURCE_NAME = re.compile(r"\s*Function (\S+):")
RESOURCE = re.compile(r"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")
RELOCATIONS = ((re.compile(r"/\*[^*]*\*/"), ""), (re.compile(r"(MOV R\d+, )0x[0-9a-f]+"), r"\1IMM"), (re.compile(r"c\[0x4\]\[0x[0-9a-f]+\]"), "c[0x4][SYM]"))


def dump(cubin, flag):
    return subprocess.run([CUOBJDUMP, flag, cubin], check=True, capture_output=True, text=True).stdout


def resources(cubin):
    table, name = {}, None
    for line in dump(cubin, "-res-usage").splitlines():
        match = RESOURCE_NAME.match(line)
        if match:
            name = match.group(1)
            continue
        match = RESOURCE.match(line)
        if match and name:
            table[name] = "REG %s STACK %s SHARED %s LOCAL %s" % match.groups()
            name = None
    return table


def code(cubin):
    table, name, lines = {}, None, []
    for line in dump(cubin, "-sass").splitlines() + ["Function : end"]:
        match = FUNCTION.match(line)
        if match:
            if name:
                table[name] = hashlib.sha256("\n".join(lines).encode()).hexdigest()
            name, lines = match.group(1), []
            continue
        for pattern, replacement in RELOCATIONS:
            line = pattern.sub(replacement, line)
        lines.append(line.rstrip())
    return table


def demangle(name):
    try:
        result = subprocess.run(["c++filt", name], capture_output=True, text=True)
    except OSError:
        return name
    return result.stdout.strip() if result.returncode == 0 else name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("head")
    parser.add_argument("--allow", default="", help="regex of kernels allowed to change")
    args = parser.parse_args()
    base_code, head_code = code(args.base), code(args.head)
    empty = [path for path, table in ((args.base, base_code), (args.head, head_code)) if not table]
    if empty:
        print("no kernels parsed from %s: cuobjdump -sass printed no functions (is nvdisasm installed?)" % " and ".join(empty))
        return 1
    base_resources, head_resources = resources(args.base), resources(args.head)
    allow = re.compile(args.allow) if args.allow else None
    unexpected = 0
    for name in sorted(set(base_code) | set(head_code)):
        if base_code.get(name) == head_code.get(name):
            continue
        state = "added" if name not in base_code else "removed" if name not in head_code else "changed"
        permitted = allow is not None and allow.search(demangle(name)) is not None
        unexpected += 0 if permitted else 1
        print("%s %s %s -> %s %s" % ("allowed" if permitted else "UNEXPECTED", state, base_resources.get(name, "-"), head_resources.get(name, "-"), demangle(name)))
    print("kernels %d identical %d unexpected %d" % (len(set(base_code) | set(head_code)), sum(1 for name in base_code if base_code[name] == head_code.get(name)), unexpected))
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
