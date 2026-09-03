#!/usr/bin/env python3
"""Duplicate-code detector over the production trees (W4 redundancy lane).

Token-clone scan: every function in the scan roots is extracted, its body is
normalized to identifier-free token lines (numbers, strings and identifiers
collapse, so renamed clones still match), and every pair of functions sharing
a >= MIN_MATCHED_LINES aligned run is reported with exact locations. The
output is the auditor artifact required by docs/HOUSECLEANING_PLAN.md W4.1:
every duplication known; the triage (consolidate vs justified) lives in
docs/AGENT_LANE_BRIEFS/reports/redundancy-<date>.md.

Usage:
    python3 tools/dup_report.py                 # markdown report on stdout
    python3 tools/dup_report.py --json OUT      # also write machine-readable hits
    python3 tools/dup_report.py --min-lines 30  # threshold (default 30)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("modules", "runtime", "node", "cache", "ring")
EXTENSIONS = {".c", ".h", ".cu", ".cuh"}
WINDOW_LINES = 8          # seed window; a run must still exceed --min-lines
MAX_SEED_GAP = 6          # max line drift between consecutive seeds of a run
TRIVIAL_LINES = {"", "{", "}", "};", "else", "break;", "return;", "#..."}

C_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "return", "break", "continue", "goto", "typedef", "struct", "enum",
    "union", "static", "const", "extern", "inline", "sizeof", "volatile",
    "register", "auto", "signed", "unsigned", "void", "char", "short",
    "int", "long", "float", "double", "bool", "_Bool", "size_t", "ssize_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t",
    "int32_t", "int64_t", "uintptr_t", "ptrdiff_t",
}

NAME_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NON_FUNCTION_NAMES = {"if", "for", "while", "switch", "return", "sizeof", "typedef", "struct", "enum", "union", "using", "namespace", "operator", "defined"}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?[fFuUlL]*|\"[^\"]*\"|'[^']*'|.")


def strip_code(text: str) -> list[str]:
    """Return per-line code with comments blanked and literals neutralized."""
    out: list[str] = []
    in_block = False
    for raw in text.splitlines():
        line, i = [], 0
        while i < len(raw):
            if in_block:
                end = raw.find("*/", i)
                if end < 0:
                    i = len(raw)
                else:
                    in_block = False
                    i = end + 2
                continue
            two = raw[i:i + 2]
            if two == "/*":
                in_block = True
                i += 2
            elif two == "//":
                break
            elif raw[i] in "\"'":
                quote, j = raw[i], i + 1
                while j < len(raw) and raw[j] != quote:
                    j += 2 if raw[j] == "\\" else 1
                line.append(f"{quote}S{quote}")
                i = j + 1
            else:
                line.append(raw[i])
                i += 1
        out.append("".join(line))
    return out


def normalize_line(code_line: str) -> str:
    tokens = []
    for match in TOKEN_RE.finditer(code_line):
        token = match.group(0)
        if token[0] == '"' or token[0] == "'":
            tokens.append("S")
        elif token[0].isdigit():
            tokens.append("N")
        elif token in C_KEYWORDS:
            tokens.append(token)
        else:
            tokens.append("I")
    return " ".join(tokens)


class Function:
    __slots__ = ("path", "name", "start", "end", "norm")

    def __init__(self, path: Path, name: str, start: int, end: int, norm: list[str]):
        self.path, self.name, self.start, self.end, self.norm = path, name, start, end, norm

    def where(self) -> str:
        return f"{self.path}:{self.name} ({self.start + 1}-{self.end + 1})"


def _net(line: str, open_ch: str, close_ch: str) -> int:
    return line.count(open_ch) - line.count(close_ch)


def extract_functions(path: Path) -> list[Function]:
    lines = strip_code(path.read_text(encoding="utf-8", errors="surrogateescape"))
    functions = []
    depth = 0
    pending: list[str] = []
    paren = 0
    in_signature = False
    start_line = -1
    for index, line in enumerate(lines):
        stripped = line.strip()
        if depth == 0 and not in_signature:
            if stripped.startswith("#") or stripped.startswith("extern \"C\""):
                continue
            if "(" in stripped and not stripped.startswith(("}", "{", ")")):
                words = WORD_RE.findall(stripped.split("(")[0])
                name = words[-1] if words else None
                if not name or name in NON_FUNCTION_NAMES:
                    continue
                in_signature = True
                pending = [line]
                paren = _net(stripped, "(", ")")
                if paren <= 0:
                    if "{" in stripped:
                        start_line = index
                        depth = _net(stripped, "{", "}")
                    elif ";" in stripped:
                        in_signature = False
            continue
        if depth > 0:
            pending.append(line)
            depth += _net(line, "{", "}")
            if depth <= 0:
                words = WORD_RE.findall(pending[0].split("(")[0]) if pending else []
                name = words[-1] if words else None
                if name and name not in NON_FUNCTION_NAMES:
                    norm = [normalize_line(code) for code in pending[1:]]
                    functions.append(Function(path, name, start_line, index, norm))
                pending = []
                in_signature = False
                depth = 0
            continue
        # signature continuation: multi-line parameter lists
        pending.append(line)
        paren += _net(stripped, "(", ")")
        if paren > 0:
            continue
        if "{" in stripped:
            start_line = index
            depth = _net(stripped, "{", "}")
            if depth <= 0:
                # one-line body on the signature line
                words = WORD_RE.findall(pending[0].split("(")[0]) if pending else []
                name = words[-1] if words else None
                if name and name not in NON_FUNCTION_NAMES:
                    norm = [normalize_line(code) for code in pending[1:]]
                    functions.append(Function(path, name, start_line, index, norm))
                pending = []
                in_signature = False
        elif ";" in stripped or "}" in stripped:
            pending = []
            in_signature = False
    return functions


def informative(norm: str) -> bool:
    return norm not in TRIVIAL_LINES and len(norm) > 2


def window_hashes(norm: list[str]) -> dict[int, int]:
    hashes = {}
    for offset in range(len(norm) - WINDOW_LINES + 1):
        window = "\x1f".join(norm[offset:offset + WINDOW_LINES])
        hashes[hashlib.blake2b(window.encode(), digest_size=12).digest()] = offset
    return hashes


def find_runs(fn_a: Function, fn_b: Function, seeds: list[tuple[int, int]], min_lines: int) -> list[dict]:
    """Extend seed pairs into maximal diagonal runs; keep those >= min_lines."""
    seeds = sorted(seeds)
    runs = []
    cur: list[tuple[int, int]] = []
    matched = 0
    index = 0
    while index < len(seeds):
        a_off, b_off = seeds[index]
        if not cur:
            cur = [(a_off, b_off)]
            matched = sum(1 for k in range(WINDOW_LINES)
                          if informative(fn_a.norm[a_off + k]) and fn_a.norm[a_off + k] == fn_b.norm[b_off + k])
            index += 1
            continue
        prev_a, prev_b = cur[-1]
        delta_a, delta_b = a_off - prev_a, b_off - prev_b
        if delta_a == delta_b and 0 < delta_a <= MAX_SEED_GAP:
            step = sum(1 for k in range(WINDOW_LINES)
                       if informative(fn_a.norm[a_off + k]) and fn_a.norm[a_off + k] == fn_b.norm[b_off + k])
            overlap = max(0, WINDOW_LINES - delta_a)
            matched += step - overlap if step >= overlap else 0
            cur.append((a_off, b_off))
            index += 1
            continue
        if matched >= min_lines:
            runs.append({"a": (cur[0][0], cur[-1][0] + WINDOW_LINES),
                         "b": (cur[0][1], cur[-1][1] + WINDOW_LINES), "matched": matched})
        cur = []
    if cur and matched >= min_lines:
        runs.append({"a": (cur[0][0], cur[-1][0] + WINDOW_LINES),
                     "b": (cur[0][1], cur[-1][1] + WINDOW_LINES), "matched": matched})
    return runs


def scan(min_lines: int) -> tuple[list[dict], int]:
    functions: list[Function] = []
    for root in SCAN_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix in EXTENSIONS and path.is_file():
                functions.extend(extract_functions(path))

    by_hash: dict[bytes, list[tuple[int, int]]] = defaultdict(list)
    for fn_index, fn in enumerate(functions):
        for digest, offset in window_hashes(fn.norm).items():
            by_hash[digest].append((fn_index, offset))

    pair_seeds: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for occurrences in by_hash.values():
        if len(occurrences) < 2 or len(occurrences) > 40:
            continue  # ubiquitous windows are noise, not clones
        for i, (fn_a, off_a) in enumerate(occurrences):
            for fn_b, off_b in occurrences[i + 1:]:
                if fn_a == fn_b:
                    continue
                pair_seeds[(fn_a, fn_b) if fn_a < fn_b else (fn_b, fn_a)].append(
                    (off_a, off_b) if fn_a < fn_b else (off_b, off_a))

    hits = []
    for (fn_a_idx, fn_b_idx), seeds in pair_seeds.items():
        fn_a, fn_b = functions[fn_a_idx], functions[fn_b_idx]
        if len(seeds) * WINDOW_LINES < min_lines:
            continue
        for run in find_runs(fn_a, fn_b, seeds, min_lines):
            hits.append({
                "a": {"file": str(fn_a.path.relative_to(ROOT)), "function": fn_a.name,
                      "lines": [run["a"][0] + fn_a.start + 1, min(run["a"][1] + fn_a.start + 1, fn_a.end + 1)]},
                "b": {"file": str(fn_b.path.relative_to(ROOT)), "function": fn_b.name,
                      "lines": [run["b"][0] + fn_b.start + 1, min(run["b"][1] + fn_b.start + 1, fn_b.end + 1)]},
                "matched_lines": run["matched"],
            })
    hits.sort(key=lambda hit: -hit["matched_lines"])
    return hits, len(functions)


def render_markdown(hits: list[dict], scanned: int, functions: int, min_lines: int) -> str:
    lines = [
        "# Duplicate-code scan (tools/dup_report.py)",
        "",
        f"- scan roots: {', '.join(SCAN_ROOTS)} (`*.c *.h *.cu *.cuh`)",
        f"- functions parsed: {functions} across {scanned} files",
        f"- detection: identifier/number/string-normalized token lines; diagonal-run",
        f"  extension over {WINDOW_LINES}-line window seeds; report threshold >= {min_lines} matched lines",
        f"- hits: {len(hits)}",
        "",
        "| # | matched | A | B |",
        "|---|---------|---|---|",
    ]
    for number, hit in enumerate(hits, 1):
        a, b = hit["a"], hit["b"]
        lines.append(f"| {number} | {hit['matched_lines']} | `{a['file']}:{a['function']}:{a['lines'][0]}-{a['lines'][1]}` | `{b['file']}:{b['function']}:{b['lines'][0]}-{b['lines'][1]}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-lines", type=int, default=30)
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()

    files = 0
    for root in SCAN_ROOTS:
        base = ROOT / root
        if base.is_dir():
            files += sum(1 for path in base.rglob("*") if path.suffix in EXTENSIONS and path.is_file())
    hits, functions = scan(arguments.min_lines)
    print(render_markdown(hits, files, functions, arguments.min_lines))
    if arguments.json:
        arguments.json.write_text(json.dumps(hits, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
