import re
import sys
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", "node_modules", ".dsv4pro_binary_cache"}
C_EXTS = {".c", ".h"}
INTEREST = {".c", ".h", ".json", ".sh", ".mk", ".py"}

NUM_RE = re.compile(
    r"(?<![\w.])"
    r"(0[xX][0-9a-fA-F]+[uUlL]*|\d+\.?\d*[uUlLfF]*)"
    r"(?![\w.])"
)

DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+?)\s*$")
ASSERT_RE = re.compile(r"_Static_assert\s*\((.*?)\)\s*;", re.S)


def iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix in INTEREST:
            yield p


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def norm(tok):
    t = tok.rstrip("uUlLfF")
    try:
        if t.lower().startswith("0x"):
            return str(int(t, 16))
        if "." in t:
            return t
        return str(int(t))
    except ValueError:
        return t


def main():
    occurrences = defaultdict(list)
    defines = defaultdict(list)
    asserts = []
    for p in iter_files():
        rel = p.relative_to(ROOT)
        try:
            raw = p.read_text(errors="replace")
        except OSError:
            continue
        text = strip_comments(raw) if p.suffix in C_EXTS else raw
        if p.suffix in C_EXTS:
            for m in DEFINE_RE.finditer(text):
                defines[m.group(1)].append(str(rel))
            for m in ASSERT_RE.finditer(text):
                asserts.append((str(rel), " ".join(m.group(1).split())))
            for ln, line in enumerate(text.splitlines(), 1):
                if "#define" in line or "#include" in line or "#pragma" in line:
                    body = line.split("#define", 1)
                    scan_zone = body[1] if len(body) > 1 else ""
                else:
                    scan_zone = line
                for m in NUM_RE.finditer(scan_zone):
                    v = norm(m.group(1))
                    if v is not None:
                        occurrences[v].append((str(rel), ln))
        elif p.suffix == ".json":
            for ln, line in enumerate(text.splitlines(), 1):
                for m in NUM_RE.finditer(line):
                    v = norm(m.group(1))
                    if v is not None:
                        occurrences[v].append((str(rel), ln))

    rows = []
    for v, occ in occurrences.items():
        files = sorted({o[0] for o in occ})
        if v.isdigit() and int(v) < 2:
            continue
        rows.append({
            "value": v,
            "count": len(occ),
            "nfiles": len(files),
            "files": files,
            "sample": occ[:6],
        })
    rows.sort(key=lambda r: (-r["nfiles"], -r["count"]))

    out = {
        "root": str(ROOT),
        "top_by_filespread": rows[:150],
        "define_names": {k: v for k, v in sorted(defines.items(), key=lambda kv: -len(kv[1])) if len(v) > 1},
        "static_asserts": asserts,
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
