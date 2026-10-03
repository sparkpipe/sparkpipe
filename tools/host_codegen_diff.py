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
STATIC_LOCAL = re.compile(r"(\.(?:bss|data|rodata|tbss|tdata)\.[A-Za-z_]\w*)\.(\d+)\b")
DATA_SECTION = re.compile(r"^\.(?:data|rodata)(?:\.|$)")
RELOCATION_HEADER = re.compile(r"^RELOCATION RECORDS FOR \[(.+)\]:$")
RELOCATION_ENTRY = re.compile(r"^([0-9a-f]+)\s+(R_\S+)\s+(\S+)$")
NUMBERED = re.compile(r"^(.+)\.(\d+)$")
PADDING = re.compile(r"^(?:nop\w*|int3|xchg %ax,%ax|data16 .*|cs nop\w* .*)$")


def run(tool, *arguments):
    return subprocess.run([tool, *arguments], check=True, capture_output=True, text=True).stdout


def sections(path):
    table = {}
    for line in run(READELF, "-SW", path).splitlines():
        match = SECTION.match(line)
        if match:
            table[match.group(1)] = (match.group(2), int(match.group(3), 16), int(match.group(4), 16), int(match.group(5), 16))
    return table


def local_constants(path):
    table, constants, data = sections(path), {}, Path(path).read_bytes()
    for line in run(READELF, "-sW", path).splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[7].startswith(".LC") or fields[6] not in table:
            continue
        name, offset, size, entry = table[fields[6]]
        start = offset + int(fields[1], 16)
        if ".str" in name:
            constants[fields[7]] = "STR(%s)" % data[start:data.index(b"\0", start)].hex()
        else:
            constants[fields[7]] = "CST(%s)" % data[start:start + (entry or size - int(fields[1], 16))].hex()
    for name, offset, size, entry in table.values():
        if ".str" in name:
            cursor = 0
            while cursor < size:
                end = data.index(b"\0", offset + cursor)
                constants[section_offset(name, cursor)] = "STR(%s)" % data[offset + cursor:end].hex()
                cursor = end + 1 - offset
        elif ".cst" in name and entry:
            for cursor in range(0, size, entry):
                constants[section_offset(name, cursor)] = "CST(%s)" % data[offset + cursor:offset + cursor + entry].hex()
    return constants


def section_offset(name, offset):
    return name if offset == 0 else "%s+0x%x" % (name, offset)


def symbol(text, constants, kind=""):
    if kind.startswith("R_AARCH64_") and text in constants:
        return constants[text]
    return re.sub(r"\.LC\d+", lambda found: constants.get(found.group(0), found.group(0)), text)


def normalize(text, start, constants):
    relocations = ["reloc %s %s" % (kind, symbol(target, constants, kind)) for kind, target in INLINE_RELOCATION.findall(text)]
    text = INLINE_RELOCATION.sub("", text)
    text = re.sub(r"\s*//.*$", "", text)
    text = re.sub(r"\s+#\s.*$", "", text)
    text = re.sub(r"\s*<[^>]*>", "", text)
    text = re.sub(r"\s+", " ", symbol(text, constants)).strip()
    branch = BRANCH.match(text)
    if branch:
        text = "%s +%x" % (branch.group(1), int(branch.group(2), 16) - start)
    return [text] + relocations


def renumber(lines):
    order = {}

    def local(found):
        if found.group(0) not in order:
            order[found.group(0)] = "%s.#%d" % (found.group(1), sum(1 for key in order if key.rsplit(".", 1)[0] == found.group(1)))
        return order[found.group(0)]

    return [STATIC_LOCAL.sub(local, line) for line in lines]


def section_relocations(path):
    table, section = {}, None
    for line in run(OBJDUMP, "-r", "-w", path).splitlines():
        header = RELOCATION_HEADER.match(line)
        entry = RELOCATION_ENTRY.match(line.strip())
        if header:
            section = header.group(1)
        elif section is not None and entry:
            table.setdefault(section, []).append((int(entry.group(1), 16), entry.group(2), entry.group(3)))
    return table


def ordinal_names(names):
    numbered = {}
    for name in names:
        match = NUMBERED.match(name)
        if match:
            numbered.setdefault(match.group(1), []).append((int(match.group(2)), name))
    renamed = {name: name for name in names}
    for base, entries in numbered.items():
        for rank, (_, name) in enumerate(sorted(entries)):
            renamed[name] = "%s.#%d" % (base, rank)
    return renamed


def data_objects(path, constants):
    table, objects, data, relocations = sections(path), {}, Path(path).read_bytes(), section_relocations(path)
    for line in run(READELF, "-sW", path).splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[3] != "OBJECT" or fields[6] not in table:
            continue
        name, offset = table[fields[6]][0], table[fields[6]][1]
        if not DATA_SECTION.match(name) or ".str" in name or ".cst" in name:
            continue
        value, size = int(fields[1], 16), int(fields[2], 0)
        entries = ["bytes " + data[offset + value:offset + value + size].hex()]
        entries += ["reloc +%x %s %s" % (at - value, kind, symbol(target, constants, kind)) for at, kind, target in relocations.get(name, []) if value <= at < value + size]
        objects[fields[7]] = "\n".join(renumber(entries))
    renamed = ordinal_names(objects)
    return {"data " + renamed[name]: text for name, text in objects.items()}


def functions(path):
    constants, table, name, start, lines = local_constants(path), {}, None, 0, []
    for line in run(OBJDUMP, "-d", "-r", "--no-show-raw-insn", "-w", path).splitlines() + ["0 <end>:"]:
        header = HEADER.match(line)
        if header:
            while lines and PADDING.match(lines[-1]):
                lines.pop()
            if name is not None:
                table[name] = "\n".join(renumber(lines))
            name, start, lines = header.group(2), int(header.group(1), 16), []
            continue
        relocation = RELOCATION.match(line)
        instruction = INSTRUCTION.match(line)
        if name is None or (relocation is None and instruction is None):
            continue
        if relocation:
            lines.append("reloc %s %s" % (relocation.group(1), symbol(relocation.group(2), constants, relocation.group(1))))
        else:
            lines.extend(normalize(instruction.group(2), start, constants))
    table.update(data_objects(path, constants))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("head")
    parser.add_argument("--allow", default="", help="regex of functions allowed to change")
    args = parser.parse_args()
    base_code, head_code = functions(args.base), functions(args.head)
    empty = [path for path, table in ((args.base, base_code), (args.head, head_code)) if not any(not name.startswith("data ") for name in table)]
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
    names = set(base_code) | set(head_code)
    print("functions %d data %d identical %d unexpected %d" % (sum(1 for name in names if not name.startswith("data ")), sum(1 for name in names if name.startswith("data ")), sum(1 for name in base_code if base_code[name] == head_code.get(name)), unexpected))
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
