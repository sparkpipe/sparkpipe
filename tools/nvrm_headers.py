#!/usr/bin/env python3
"""Generate the nvrm device backend's resource-manager, class and QMD headers.

The nvrm backend drives the GPU through the open kernel module's ioctl
interface. Its structures and constants come from two pinned upstream trees:
the open kernel module at one release tag (the ioctls, the resource-manager
SDK, the class numbers and the host and GPFIFO formats) and open-gpu-doc at
one commit (the compute and copy-engine methods and the QMD layout). This
tool copies the listed headers and every header they include, with their
comments removed, into runtime/device/nvrm/vendor/, rewrites their NOTICE
entries and writes the module version the backend accepts at open.

    python3 -I tools/nvrm_headers.py --module DIR --doc DIR [--check]

DIR is a checkout of the pinned commit; the tool refuses any other commit.
--check renders into memory and fails when the tree differs.
"""
import argparse
import re
import textwrap
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
VENDOR = PurePosixPath("runtime/device/nvrm/vendor")
VERSION_HEADER = PurePosixPath("runtime/device/nvrm/nvrm_version.h")
NOTICE = ROOT / "NOTICE"

MODULE = {
    "name": "open-gpu-kernel-modules",
    "upstream": "https://github.com/NVIDIA/open-gpu-kernel-modules",
    "tag": "580.178.04",
    "commit": "c8e699821c23e4335bf23330a54a23d15cfb95e9",
    "include": (
        "src/common/sdk/nvidia/inc",
        "src/nvidia/arch/nvalloc/unix/include",
        "kernel-open/common/inc",
        "kernel-open/nvidia-uvm",
    ),
    "headers": (
        "src/common/sdk/nvidia/inc/nvtypes.h",
        "src/common/sdk/nvidia/inc/nvstatus.h",
        "src/common/sdk/nvidia/inc/nvos.h",
        "src/common/sdk/nvidia/inc/nverror.h",
        "src/common/sdk/nvidia/inc/class/cl0000.h",
        "src/common/sdk/nvidia/inc/class/cl0005.h",
        "src/common/sdk/nvidia/inc/class/cl003e.h",
        "src/common/sdk/nvidia/inc/class/cl0040.h",
        "src/common/sdk/nvidia/inc/class/cl0070.h",
        "src/common/sdk/nvidia/inc/class/cl0071.h",
        "src/common/sdk/nvidia/inc/class/cl0080.h",
        "src/common/sdk/nvidia/inc/class/cl2080.h",
        "src/common/sdk/nvidia/inc/class/cl50a0.h",
        "src/common/sdk/nvidia/inc/class/cl9067.h",
        "src/common/sdk/nvidia/inc/class/cl90f1.h",
        "src/common/sdk/nvidia/inc/class/cla06c.h",
        "src/common/sdk/nvidia/inc/class/clc361.h",
        "src/common/sdk/nvidia/inc/class/clc56f.h",
        "src/common/sdk/nvidia/inc/class/clc661.h",
        "src/common/sdk/nvidia/inc/class/clc761.h",
        "src/common/sdk/nvidia/inc/class/clc96f.h",
        "src/common/sdk/nvidia/inc/class/clca6f.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000client.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000system.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000unix.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl003e.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl0041.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080fifo.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl0080/ctrl0080gpu.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080ce.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080event.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fb.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fifo.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gpu.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080gr.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080mc.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080rc.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080tmr.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrl90f1.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrla06c.h",
        "src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h",
        "src/nvidia/arch/nvalloc/unix/include/nv_escape.h",
        "src/nvidia/arch/nvalloc/unix/include/nv-unix-nvos-params-wrappers.h",
        "kernel-open/common/inc/nv-ioctl.h",
        "kernel-open/nvidia-uvm/uvm_ioctl.h",
        "kernel-open/nvidia-uvm/uvm_linux_ioctl.h",
    ),
}

DOC = {
    "name": "open-gpu-doc",
    "upstream": "https://github.com/NVIDIA/open-gpu-doc",
    "tag": None,
    "commit": "9fdf5c4062007929d9f4e6cbad9c9771fe61b880",
    "include": (".",),
    "external": ("nvtypes.h",),
    "headers": (
        "classes/compute/clcec0.h",
        "classes/compute/clcec0qmd.h",
        "classes/dma-copy/clcab5.h",
    ),
}

UPSTREAMS = (MODULE, DOC)
INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"])([^>"\n]+)[>"]', re.M)
DISABLED = re.compile(r"^[ \t]*#[ \t]*if[ \t]+0[ \t]*$")
CONDITIONAL_OPEN = re.compile(r"^[ \t]*#[ \t]*if(?:n?def)?\b")
CONDITIONAL_ELSE = re.compile(r"^[ \t]*#[ \t]*(?:else|elif)\b")
CONDITIONAL_CLOSE = re.compile(r"^[ \t]*#[ \t]*endif\b")
LEXEME = re.compile(r"//|/\*|\"|'")
GUARDED_DEFAULT = re.compile(r"^[ \t]*#[ \t]*(?:ifndef[ \t]+(\w+)|if[ \t]+![ \t]*defined[ \t]*\([ \t]*(\w+)[ \t]*\))[ \t]*\n"
                             r"([ \t]*#[ \t]*define[ \t]+(\w+)[ \t]+\S[^\n]*)\n[ \t]*#[ \t]*endif\b[^\n]*\n", re.M)
GUARD_LINE = re.compile(r"^[ \t]*#[ \t]*(?:ifndef[ \t]+(\w+)|if[ \t]+![ \t]*defined[ \t]*\([ \t]*(\w+)[ \t]*\))[ \t]*$")
DEFINE_LINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)[ \t]+\S")
STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')
CHARACTER = re.compile(r"'(?:[^'\\\n]|\\.)*'")
MIT_MARKERS = ("SPDX-License-Identifier: MIT", "Permission is hereby granted, free of charge")
COPYRIGHT = re.compile(r"(Copyright \(c\) [0-9][^\n]*?)[ \t*/]*$", re.M)


class GenerationError(Exception):
    pass


def strip_comments(text, label):
    pieces = []
    position = 0
    while True:
        match = LEXEME.search(text, position)
        if match is None:
            pieces.append(text[position:])
            break
        start = match.start()
        pieces.append(text[position:start])
        token = match.group(0)
        if token == "//":
            end = start
            while True:
                newline = text.find("\n", end)
                if newline < 0:
                    end = len(text)
                    break
                if text[newline - 1] == "\\":
                    end = newline + 1
                    continue
                end = newline
                break
            pieces.append(" ")
            position = end
        elif token == "/*":
            close = text.find("*/", start + 2)
            if close < 0:
                raise GenerationError(f"{label}:{text.count(chr(10), 0, start) + 1}: unterminated block comment")
            pieces.append(" ")
            position = close + 2
        else:
            literal = (STRING if token == '"' else CHARACTER).match(text, start)
            if literal is None:
                raise GenerationError(f"{label}:{text.count(chr(10), 0, start) + 1}: unterminated literal")
            pieces.append(literal.group(0))
            position = literal.end()
    return "".join(pieces)


def drop_disabled(text):
    kept = []
    depth = 0
    for line in text.split("\n"):
        if depth == 0:
            if DISABLED.match(line):
                depth = 1
                continue
            kept.append(line)
            continue
        if CONDITIONAL_OPEN.match(line):
            depth += 1
        elif CONDITIONAL_CLOSE.match(line):
            depth -= 1
        elif depth == 1 and CONDITIONAL_ELSE.match(line):
            raise GenerationError("an #if 0 group with an #else or #elif branch")
    if depth != 0:
        raise GenerationError("an #if 0 group without its #endif")
    return "\n".join(kept)


def tidy(text):
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    out = []
    for line in lines:
        if line == "" and (not out or out[-1] == ""):
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def pin_defaults(text, label):
    pinned = GUARDED_DEFAULT.sub(lambda match: match.group(3) + "\n" if match.group(4) == (match.group(1) or match.group(2)) else match.group(0), text)
    lines = pinned.split("\n")
    for line, following in zip(lines, lines[1:]):
        guard, define = GUARD_LINE.match(line), DEFINE_LINE.match(following)
        if guard and define and define.group(1) == (guard.group(1) or guard.group(2)):
            raise GenerationError(f"{label}: the build default {define.group(1)} has branches and cannot be fixed at its value")
    return pinned


def render(text, label):
    return tidy(pin_defaults(tidy(drop_disabled(strip_comments(text, label))), label))


def checkout_commit(directory):
    run = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
    if run.returncode != 0:
        raise GenerationError(f"{directory}: not a git checkout ({run.stderr.strip()})")
    return run.stdout.strip()


def check_checkout(upstream, directory):
    commit = checkout_commit(directory)
    if commit != upstream["commit"]:
        raise GenerationError(f"{upstream['name']} checkout {directory} is at {commit}, the pin is {upstream['commit']}")
    if upstream["tag"] is not None:
        version = re.search(r"^NVIDIA_VERSION = (\S+)$", (directory / "version.mk").read_text(), re.M)
        if version is None or version.group(1) != upstream["tag"]:
            raise GenerationError(f"{upstream['name']} checkout {directory} does not build version {upstream['tag']}")


def owning_root(upstream, relative):
    roots = sorted(upstream["include"], key=lambda root: len(PurePosixPath(root).parts), reverse=True)
    for root in roots:
        base = PurePosixPath(root)
        if base == PurePosixPath(".") or relative.parts[:len(base.parts)] == base.parts:
            return base
    raise GenerationError(f"{upstream['name']} {relative}: not under an include root")


def normalized(path):
    return PurePosixPath(*[part for part in path.parts if part != "."])


def canonical(upstream, directory, relative):
    output = relative.relative_to(owning_root(upstream, relative))
    for root in upstream["include"]:
        candidate = normalized(PurePosixPath(root) / output)
        if (directory / candidate).is_file():
            return candidate
    return relative


def resolve(upstream, directory, including, kind, name):
    candidates = []
    if kind == '"':
        candidates.append(including.parent / name)
    candidates += [PurePosixPath(root) / name for root in upstream["include"]]
    for candidate in candidates:
        path = normalized(candidate)
        if ".." in path.parts:
            raise GenerationError(f"{upstream['name']} {including}: include {name} leaves its root")
        if (directory / path).is_file():
            return canonical(upstream, directory, path)
    return None


def closure(upstream, directory):
    pending = [canonical(upstream, directory, PurePosixPath(header)) for header in upstream["headers"]]
    seen = {}
    while pending:
        relative = pending.pop(0)
        if relative in seen:
            continue
        path = directory / relative
        if not path.is_file():
            raise GenerationError(f"{upstream['name']} {relative}: missing from the checkout")
        text = path.read_text(encoding="utf-8", errors="strict")
        seen[relative] = text
        for kind, name in INCLUDE.findall(strip_comments(text, str(relative))):
            found = resolve(upstream, directory, relative, kind, name)
            if found is None and kind == '"' and name not in upstream.get("external", ()):
                raise GenerationError(f"{upstream['name']} {relative}: include \"{name}\" is not in the checkout")
            if found is not None and found not in seen:
                pending.append(found)
    return seen


def outputs_for(upstream, directory):
    files = {}
    copyrights = set()
    for relative, text in closure(upstream, directory).items():
        if not any(marker in text for marker in MIT_MARKERS):
            raise GenerationError(f"{upstream['name']} {relative}: carries no MIT notice")
        copyrights.update(" ".join(notice.split()) for notice in COPYRIGHT.findall(text))
        output = VENDOR / upstream["name"] / relative.relative_to(owning_root(upstream, relative))
        rendered = render(text, f"{upstream['name']}/{relative}")
        if output in files and files[output][1] != rendered:
            raise GenerationError(f"{upstream['name']} {relative} and {files[output][0]}: differ but both provide {output}")
        files.setdefault(output, (relative, rendered))
    return files, sorted(copyrights)


def license_text(directory, upstream):
    source = "COPYING" if upstream is MODULE else "LICENSE.md"
    text = (directory / source).read_text(encoding="utf-8")
    start = text.index("Permission is hereby granted")
    end = text.index("DEALINGS IN THE SOFTWARE.", start) + len("DEALINGS IN THE SOFTWARE.")
    return "\n".join(line.strip() for line in text[start:end].split("\n"))


def notice_entry(upstream, files, copyrights, directory):
    version = f"release {upstream['tag']}, commit {upstream['commit']}" if upstream["tag"] else f"commit {upstream['commit']}"
    roots = ", ".join(upstream["include"]) if upstream is MODULE else "the repository root"
    lines = [f"Files: {path}" for path in sorted(str(path) for path in files)]
    lines += [f"Upstream: {upstream['upstream']}", f"Commit: {upstream['commit']}", "License: MIT", ""]
    prose = (f"The files under {VENDOR}/{upstream['name']}/ are headers of {upstream['name']} "
             f"({upstream['upstream']}, {version}), generated by tools/nvrm_headers.py. "
             f"Each keeps its path relative to its include root ({roots}). "
             "Changes: the source comments are removed, and a default a build could override is fixed at its upstream value. "
             "They are distributed under the following license:")
    lines += textwrap.wrap(prose, width=78, break_on_hyphens=False) + [""]
    lines += copyrights
    lines += ["", license_text(directory, upstream), ""]
    return "\n".join(lines) + "\n"


def updated_notice(text, upstream, entry):
    prefix = f"Files: {VENDOR}/{upstream['name']}/"
    lines = text.split("\n")
    start = next((index for index, line in enumerate(lines) if line.startswith(prefix)), None)
    if start is None:
        return text.rstrip("\n") + "\n\n" + entry
    end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("Files: ") and not lines[index].startswith(prefix)), len(lines))
    before = "\n".join(lines[:start]).rstrip("\n")
    after = "\n".join(lines[end:])
    joined = before + "\n\n" + entry
    if after.strip():
        joined += "\n" + after
    return joined


def version_header():
    return "#pragma once\n\n#define SPARK_NVRM_MODULE_VERSION \"" + MODULE["tag"] + "\"\n"


def generate(directories):
    files = {}
    notice = NOTICE.read_text(encoding="utf-8")
    for upstream in UPSTREAMS:
        directory = directories[upstream["name"]]
        check_checkout(upstream, directory)
        rendered, copyrights = outputs_for(upstream, directory)
        for name in upstream.get("external", ()):
            if VENDOR / MODULE["name"] / name not in files:
                raise GenerationError(f"{upstream['name']} includes {name}, which {MODULE['name']} does not provide")
        files.update({path: text for path, (_, text) in rendered.items()})
        notice = updated_notice(notice, upstream, notice_entry(upstream, rendered, copyrights, directory))
    files[VERSION_HEADER] = version_header()
    return files, notice


def tracked_vendor_files():
    base = ROOT / VENDOR
    if not base.exists():
        return set()
    return {PurePosixPath(path.relative_to(ROOT).as_posix()) for path in base.rglob("*") if path.is_file()}


def differences(files, notice):
    found = []
    for path, text in sorted(files.items()):
        target = ROOT / path
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            found.append(f"{path}: differs from the generated header")
    for path in sorted(tracked_vendor_files() - set(files)):
        found.append(f"{path}: not generated from the pinned upstream")
    if NOTICE.read_text(encoding="utf-8") != notice:
        found.append("NOTICE: the nvrm entries differ from the generated entries")
    return found


def write(files, notice):
    for path in sorted(tracked_vendor_files() - set(files)):
        (ROOT / path).unlink()
    for path, text in files.items():
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    NOTICE.write_text(notice, encoding="utf-8")


def main(argv):
    parser = argparse.ArgumentParser(description="generate the nvrm backend headers from the pinned upstream trees")
    parser.add_argument("--module", required=True, type=Path, help=f"{MODULE['upstream']} at {MODULE['tag']}")
    parser.add_argument("--doc", required=True, type=Path, help=f"{DOC['upstream']} at {DOC['commit']}")
    parser.add_argument("--check", action="store_true", help="fail when the tree differs from the generated headers")
    arguments = parser.parse_args(argv)
    try:
        files, notice = generate({MODULE["name"]: arguments.module.resolve(), DOC["name"]: arguments.doc.resolve()})
    except (GenerationError, OSError, UnicodeDecodeError) as error:
        print(f"FAIL {error}")
        return 1
    if arguments.check:
        found = differences(files, notice)
        for line in found:
            print(f"FAIL {line}")
        if found:
            return 1
        print(f"PASS {len(files)} nvrm headers match {MODULE['tag']} and open-gpu-doc {DOC['commit'][:12]}")
        return 0
    write(files, notice)
    print(f"wrote {len(files)} nvrm headers from {MODULE['tag']} and open-gpu-doc {DOC['commit'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
