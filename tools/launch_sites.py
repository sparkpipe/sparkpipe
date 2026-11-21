#!/usr/bin/env python3
"""Rewrite triple-chevron kernel launches as LM_LAUNCH calls.

Every kernel launch in production code goes through LM_LAUNCH from
runtime/launch.h, so a device backend can supply its own launcher behind the
one name. This tool finds each `kernel<<<grid, block[, shared[, stream]]>>>(...)`
in the production source roots and rewrites it as
`LM_LAUNCH((kernel), grid, block, shared, stream, ...)` with the omitted
shared size and stream written out as their defaults (0 and 0). The text of
the kernel, the configuration and the arguments is kept as written, including
every line break, so no line moves and the compiled device code is unchanged.

Usage:
    python3 tools/launch_sites.py --list
    python3 tools/launch_sites.py --rewrite [PATH ...]
    python3 tools/launch_sites.py --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_ROOTS = (
    "api",
    "cache",
    "common",
    "deployment/src",
    "include",
    "inference",
    "model-families",
    "modules",
    "node",
    "ring",
    "runtime",
    "scheduler",
    "serving",
    "src",
    "text",
)
LAUNCH_HEADER = "runtime/launch.h"
SUFFIXES = {".cu", ".cuh", ".h", ".hpp", ".cc", ".cpp"}
OPEN = {"(": ")", "[": "]", "{": "}"}
CLOSE = {")": "(", "]": "[", "}": "{"}
IDENTIFIER = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


class SiteError(Exception):
    pass


def code_mask(text):
    mask = bytearray(len(text))
    index = 0
    length = len(text)
    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = length if end < 0 else end
            for position in range(index, end):
                mask[position] = 1
            index = end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = length if end < 0 else end + 2
            for position in range(index, end):
                mask[position] = 1
            index = end
        elif text[index] in "\"'":
            quote = text[index]
            end = index + 1
            while end < length and text[end] != quote:
                end += 2 if text[end] == "\\" else 1
            end = min(end + 1, length)
            for position in range(index, end):
                mask[position] = 1
            index = end
        else:
            index += 1
    return mask


def skip_space_back(text, index):
    while index > 0 and text[index - 1].isspace():
        index -= 1
    return index


def match_back(text, index, opener, closer):
    depth = 0
    position = index - 1
    while position >= 0:
        character = text[position]
        if character == closer:
            depth += 1
        elif character == opener:
            depth -= 1
            if depth == 0:
                return position
        position -= 1
    raise SiteError("unbalanced %s%s before launch" % (opener, closer))


def match_angle_back(text, index):
    depth = 0
    position = index - 1
    while position >= 0:
        character = text[position]
        if character in CLOSE:
            position = match_back(text, position + 1, CLOSE[character], character) - 1
            continue
        if character == ">":
            depth += 1
        elif character == "<":
            depth -= 1
            if depth == 0:
                return position
        position -= 1
    raise SiteError("unbalanced template arguments before launch")


def kernel_start(text, end):
    position = skip_space_back(text, end)
    if position == 0:
        raise SiteError("launch has no kernel")
    if text[position - 1] == ")":
        return match_back(text, position, "(", ")")
    while True:
        if text[position - 1] == ">":
            position = skip_space_back(text, match_angle_back(text, position))
        start = position
        while start > 0 and text[start - 1] in IDENTIFIER:
            start -= 1
        if start == position:
            raise SiteError("launch kernel is not a name, a template or a parenthesized expression")
        qualifier = skip_space_back(text, start)
        if qualifier >= 2 and text[qualifier - 2:qualifier] == "::":
            position = skip_space_back(text, qualifier - 2)
            if position == 0 or text[position - 1] not in IDENTIFIER and text[position - 1] != ">":
                return qualifier - 2
            continue
        return start


def split_top(text, begin, end):
    items = []
    depth = 0
    start = begin
    position = begin
    while position < end:
        character = text[position]
        if character in OPEN:
            depth += 1
        elif character in CLOSE:
            depth -= 1
        elif character == "," and depth == 0:
            items.append((start, position))
            start = position + 1
        position += 1
    items.append((start, end))
    return items


def match_forward(text, index):
    depth = 0
    position = index
    while position < len(text):
        character = text[position]
        if character in OPEN:
            depth += 1
        elif character in CLOSE:
            depth -= 1
            if depth == 0:
                return position
        position += 1
    raise SiteError("unbalanced launch arguments")


def find_sites(text):
    mask = code_mask(text)
    sites = []
    position = 0
    while True:
        index = text.find("<<<", position)
        if index < 0:
            break
        position = index + 3
        if mask[index]:
            continue
        close = index + 3
        depth = 0
        while close < len(text):
            if text.startswith(">>>", close) and depth == 0 and not mask[close]:
                break
            character = text[close]
            if not mask[close]:
                if character in OPEN:
                    depth += 1
                elif character in CLOSE:
                    depth -= 1
            close += 1
        else:
            raise SiteError("launch configuration has no >>>")
        open_paren = close + 3
        while open_paren < len(text) and text[open_paren].isspace():
            open_paren += 1
        if open_paren >= len(text) or text[open_paren] != "(":
            raise SiteError("launch configuration is not followed by arguments")
        close_paren = match_forward(text, open_paren)
        start = kernel_start(text, index)
        sites.append((start, index, close, open_paren, close_paren))
        position = close_paren + 1
    return sites


def line_of(text, index):
    return text.count("\n", 0, index) + 1


def rewrite_site(text, site):
    start, chevron, close, open_paren, close_paren = site
    kernel_end = skip_space_back(text, chevron)
    kernel = text[start:kernel_end]
    before_config = text[kernel_end:chevron]
    items = split_top(text, chevron + 3, close)
    if len(items) < 2 or len(items) > 4:
        raise SiteError("launch configuration has %d parts" % len(items))
    configuration = [text[a:b] for a, b in items]
    for item in configuration:
        if not item.strip():
            raise SiteError("launch configuration has an empty part")
    compact = all(not item[:1].isspace() for item in configuration[1:])
    space = "" if compact else " "
    while len(configuration) < 4:
        configuration.append(space + "0")
    joined = ",".join(configuration)
    if not before_config and not joined[:1].isspace():
        joined = space + joined
    between = text[close + 3:open_paren]
    arguments = text[open_paren + 1:close_paren]
    if not arguments.strip():
        raise SiteError("launch without arguments")
    if not between and not arguments[:1].isspace():
        between = space
    replacement = "LM_LAUNCH((" + kernel + ")," + before_config + joined + "," + between + arguments + ")"
    if replacement.count("\n") != text[start:close_paren + 1].count("\n"):
        raise SiteError("rewrite would move lines")
    return replacement


def rewrite_text(text):
    sites = find_sites(text)
    pieces = []
    position = 0
    for site in sites:
        pieces.append(text[position:site[0]])
        pieces.append(rewrite_site(text, site))
        position = site[4] + 1
    pieces.append(text[position:])
    return "".join(pieces), len(sites)


def launch_files(root):
    for top in LAUNCH_ROOTS:
        base = root / top
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in SUFFIXES:
                relative = path.relative_to(root).as_posix()
                if relative != LAUNCH_HEADER:
                    yield path, relative


def scan(root):
    found = {}
    for path, relative in launch_files(root):
        text = path.read_text(errors="surrogateescape")
        if "<<<" not in text:
            continue
        try:
            sites = find_sites(text)
        except SiteError as error:
            found[relative] = error
            continue
        if sites:
            found[relative] = [(line_of(text, site[1]), text[site[0]:site[4] + 1]) for site in sites]
    return found


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--rewrite", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("paths", nargs="*")
    options = parser.parse_args(argv)
    if options.rewrite:
        targets = [ROOT / path for path in options.paths] if options.paths else [path for path, _ in launch_files(ROOT)]
        total = 0
        for path in targets:
            text = path.read_text(errors="surrogateescape")
            if "<<<" not in text:
                continue
            try:
                updated, count = rewrite_text(text)
            except SiteError as error:
                print("%s: %s" % (path.relative_to(ROOT), error), file=sys.stderr)
                return 1
            if count:
                path.write_text(updated, errors="surrogateescape")
                print("%s: %d launches" % (path.relative_to(ROOT), count))
                total += count
        print("rewrote %d launches" % total)
        return 0
    found = scan(ROOT)
    count = 0
    for relative, sites in sorted(found.items()):
        if isinstance(sites, SiteError):
            print("%s: %s" % (relative, sites))
            count += 1
            continue
        for line, source in sites:
            count += 1
            if options.list:
                print("%s:%d: %s" % (relative, line, " ".join(source.split())))
            else:
                print("%s:%d: launch outside LM_LAUNCH" % (relative, line))
    if options.check:
        print("%d launches outside LM_LAUNCH" % count)
        return 1 if count else 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
