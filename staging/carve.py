import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("file", type=Path)
parser.add_argument("names", nargs="+")
args = parser.parse_args()

fn_re = re.compile(r"^(?:static\s+)?[A-Za-z_][A-Za-z0-9_ \t\*]*?\b([A-Za-z_][A-Za-z0-9_]*)\(")
td_re = re.compile(r"^typedef struct")
td_name_re = re.compile(r"^\}\s*([A-Za-z_][A-Za-z0-9_]*)\s*;")

lines = args.file.read_text().splitlines(keepends=True)
out = []
i = 0
while i < len(lines):
    m = fn_re.match(lines[i])
    if m and m.group(1) in args.names:
        j = i
        while lines[j].rstrip("\n") != "}":
            j += 1
        if j + 1 < len(lines) and lines[j + 1].strip() == "":
            j += 1
        i = j + 1
        continue
    if td_re.match(lines[i]):
        j = i
        while j < len(lines):
            tm = td_name_re.match(lines[j])
            if tm and tm.group(1) in args.names:
                if j + 1 < len(lines) and lines[j + 1].strip() == "":
                    j += 1
                i = j + 1
                break
            j += 1
        else:
            out.append(lines[i])
            i += 1
            continue
        continue
    out.append(lines[i])
    i += 1

args.file.write_text("".join(out))
