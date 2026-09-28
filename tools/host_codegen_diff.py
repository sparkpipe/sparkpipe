#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path

OBJDUMP = "objdump"
READELF = "readelf"
HEADER = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSTRUCTION = re.compile(r"^\s*([0-9a-f]+):\s*(.*)$")
RELOCATION = re.compile(r"^\s*[0-9a-f]+: (R_\S+)\s+(.*)$")
SECTION = re.compile(r"^\s*\[\s*(\d+)\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)")
BRANCH = re.compile(r"^(j\w*|call\w*|loop\w*|b\.?\w*)\s+([0-9a-f]+)$")
INLINE_RELOCATION = re.compile(r"\s*[0-9a-f]+: (R_[A-Z0-9_]+)\s+(\S+)")


def run(tool, *arguments):
    return subprocess.run([tool, *arguments], check=True, capture_output=True, text=True).stdout


def local_constants(path):
    sections, constants, data = {}, {}, Path(path).read_bytes()
    for line in run(READELF, "-SW", path).splitlines():
        match = SECTION.match(line)
        if match:
            sections[match.group(1)] = (match.group(2), int(match.group(3), 16), int(match.group(4), 16), int(match.group(5), 16))
    for line in run(READELF, "-sW", path).splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[7].startswith(".LC") or fields[6] not in sections:
            continue
        name, offset, size, entry = sections[fields[6]]
        start = offset + int(fields[1], 16)
        if ".str" in name:
            constants[fields[7]] = "STR(%s)" % data[start:data.index(b"\0", start)].hex()
        else:
            constants[fields[7]] = "CST(%s)" % data[start:start + (entry or size - int(fields[1], 16))].hex()
    return constants


def symbol(text, constants):
    return re.sub(r"\.LC\d+", lambda found: constants.get(found.group(0), found.group(0)), text)


def normalize(text, start, constants):
    relocations = ["reloc %s %s" % (kind, symbol(target, constants)) for kind, target in INLINE_RELOCATION.findall(text)]
    text = INLINE_RELOCATION.sub("", text)
    text = re.sub(r"\s*#.*$", "", text)
    text = re.sub(r"\s*<[^>]*>", "", text)
    text = re.sub(r"\s+", " ", symbol(text, constants)).strip()
    branch = BRANCH.match(text)
    if branch:
        text = "%s +%x" % (branch.group(1), int(branch.group(2), 16) - start)
    return [text] + relocations


def functions(path):
    constants, table, name, start, lines = local_constants(path), {}, None, 0, []
    for line in run(OBJDUMP, "-d", "-r", "--no-show-raw-insn", "-w", path).splitlines() + ["0 <end>:"]:
        header = HEADER.match(line)
        if header:
            if name is not None:
                table[name] = "\n".join(lines)
            name, start, lines = header.group(2), int(header.group(1), 16), []
            continue
        relocation = RELOCATION.match(line)
        instruction = INSTRUCTION.match(line)
        if name is None or (relocation is None and instruction is None):
            continue
        if relocation:
            lines.append("reloc %s %s" % (relocation.group(1), symbol(relocation.group(2), constants)))
        else:
            lines.extend(normalize(instruction.group(2), start, constants))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("head")
    parser.add_argument("--allow", default="", help="regex of functions allowed to change")
    args = parser.parse_args()
    base_code, head_code = functions(args.base), functions(args.head)
    empty = [path for path, table in ((args.base, base_code), (args.head, head_code)) if not table]
    if empty:
        print("no functions parsed from %s" % " and ".join(empty))
        return 1
    allow = re.compile(args.allow) if args.allow else None
    unexpected = 0
    for name in sorted(set(base_code) | set(head_code)):
        if base_code.get(name) == head_code.get(name):
            continue
        state = "added" if name not in base_code else "removed" if name not in head_code else "changed"
        permitted = allow is not None and allow.search(name) is not None
        unexpected += 0 if permitted else 1
        print("%s %s %s" % ("allowed" if permitted else "UNEXPECTED", state, name))
    print("functions %d identical %d unexpected %d" % (len(set(base_code) | set(head_code)), sum(1 for name in base_code if base_code[name] == head_code.get(name)), unexpected))
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
