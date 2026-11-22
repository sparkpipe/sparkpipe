#!/usr/bin/env python3
"""Write the kernel manifest the nvrm backend loads beside a cubin.

One translation unit's nvcc outputs (its cubin, the PTX nvcc fed ptxas in
the same compile, and the compile's ptxas -v log) become <tu>.manifest:

    nvcc <version>
    flags <the compile's code-generation flags>
    source <path> <sha256>
    cubin <sha256> <bytes>
    kernel <name> short <short> registers R barriers B stack S spill_stores T spill_loads L smem M [cumulative C]
    param <name> <index> <ptx type> align A bytes N offset O
    bound <name> <directive> <values>

Kernels are the PTX's .entry functions; each needs ptxas's report for the
target in the log, which may also hold other translation units' reports.
stack is the kernel's own frame, from the "Function properties" block that
names the kernel; a callee's block is not the kernel's. cumulative is the
stack its whole call chain needs, recorded whenever ptxas reports it; the
runtime sizes local memory by it. A kernel reported twice for the target
with different numbers is refused.

short is the name the runtime stores, at most 64 characters: a name that
fits is its own short name; a longer one is its base identifier (the last
source name of an Itanium-mangled name, else the name itself), truncated,
then "_" and the first 12 hex digits of sha256 over the full name. Two
kernels of one unit with one short name are refused.
Parameters are laid out as the PTX ISA lays out .param space: in
declaration order, each at its alignment (.align N, else its type's size).

    python3 -I tools/nvrm_kernel_manifest.py --cubin TU.cubin --ptx TU.ptx \\
        --ptxas-log LOG --nvcc NVCC --flags "FLAGS" --source TU.cu \\
        --arch sm_121a --output TU.manifest

Nothing is written when anything is missing or inconsistent.
"""
import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ENTRY = re.compile(r"^ptxas info\s+: Compiling entry function '([^']+)' for '([^']+)'$")
PROPERTIES = re.compile(r"^ptxas info\s+: Function properties for (\S+)$")
FRAME = re.compile(r"^\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads$")
USED = re.compile(r"^ptxas info\s+: Used (\d+) registers, used (\d+) barriers((?:, [^,]+)*)$")
SMEM = re.compile(r", (\d+) bytes smem(?:,|$)")
CUMULATIVE = re.compile(r", (\d+) bytes cumulative stack size(?:,|$)")
VERSION = re.compile(r"release [0-9.]+, V([0-9.]+)")
PTX_ENTRY = re.compile(r"\.entry\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)([^{]*)\{")
PTX_PARAM = re.compile(r"\.param\s+(?:\.align\s+(\d+)\s+)?\.([a-z]\d+)(?:\s+\.ptr(?:\s+\.[a-z]+)*(?:\s+\.align\s+\d+)?)?\s+[\w$]+(?:\[(\d+)\])?")
PTX_DIRECTIVE = re.compile(r"\.(maxntid|reqntid|minnctapersm|maxnreg|reqnctapercluster|explicitcluster|maxclusterrank)\b([^.]*)")
REPORT_FIELDS = ("registers", "barriers", "stack", "spill_stores", "spill_loads", "smem")


class ManifestError(Exception):
    pass


SHORT_LIMIT = 64
SHORT_DIGEST_DIGITS = 12
SOURCE_NAME = re.compile(r"(\d+)")


def base_identifier(name):
    if not name.startswith("_Z"):
        return name
    rest = name[2:]
    if rest.startswith("N"):
        rest = rest[1:].lstrip("rVKRO")
    identifier = None
    while rest:
        if rest.startswith("L"):
            rest = rest[1:]
            continue
        digits = SOURCE_NAME.match(rest)
        if digits is None:
            break
        start = len(digits.group(1))
        length = int(digits.group(1))
        if length == 0 or len(rest) - start < length:
            break
        identifier = rest[start:start + length]
        rest = rest[start + length:]
    return identifier if identifier is not None else name


def short_name(name):
    if len(name) <= SHORT_LIMIT:
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:SHORT_DIGEST_DIGITS]
    return f"{base_identifier(name)[:SHORT_LIMIT - 1 - SHORT_DIGEST_DIGITS]}_{digest}"


def nvcc_version(nvcc):
    run = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    found = VERSION.search(run.stdout)
    if run.returncode != 0 or found is None:
        raise ManifestError(f"{nvcc} --version did not name a release: {run.stderr.strip()}")
    return found.group(1)


def ptxas_report(text, arch):
    kernels = {}
    current = None
    properties = None
    fields = {}
    for line in text.splitlines():
        entry = ENTRY.match(line)
        if entry:
            current = entry.group(1) if entry.group(2) == arch else None
            properties = None
            fields = {}
            continue
        named = PROPERTIES.match(line)
        if named:
            properties = named.group(1)
            continue
        frame = FRAME.match(line)
        if frame and current is not None and properties == current:
            fields.update(stack=int(frame.group(1)), spill_stores=int(frame.group(2)), spill_loads=int(frame.group(3)))
            continue
        used = USED.match(line)
        if used and current is not None:
            smem = SMEM.search(used.group(3))
            cumulative = CUMULATIVE.search(used.group(3))
            fields.update(registers=int(used.group(1)), barriers=int(used.group(2)), smem=int(smem.group(1)) if smem else 0)
            if cumulative is not None:
                fields.update(cumulative=int(cumulative.group(1)))
            if current in kernels and kernels[current] != fields:
                raise ManifestError(f"the ptxas log reports {current} for {arch} twice with different numbers: {kernels[current]} and {fields}")
            kernels[current] = fields
            current = None
            properties = None
            fields = {}
    return kernels


def ptx_parameters(text):
    kernels = {}
    for entry in PTX_ENTRY.finditer(text):
        name, parameters, directives = entry.group(1), entry.group(2), entry.group(3)
        layout = []
        offset = 0
        for parameter in PTX_PARAM.finditer(parameters):
            kind = parameter.group(2)
            element = int(kind[1:]) // 8
            count = int(parameter.group(3) or 1)
            align = int(parameter.group(1) or element)
            offset = (offset + align - 1) // align * align
            layout.append((kind, align, element * count, offset))
            offset += element * count
        bounds = []
        for directive in PTX_DIRECTIVE.finditer(directives):
            values = [value for value in re.split(r"[\s,]+", directive.group(2).strip()) if value]
            bounds.append((directive.group(1), values))
        kernels[name] = (layout, bounds)
    if not kernels:
        raise ManifestError("the PTX declares no entry function")
    return kernels


def manifest(version, flags, source_name, source_bytes, cubin_bytes, ptx_text, log_text, arch):
    parameters = ptx_parameters(ptx_text)
    reports = ptxas_report(log_text, arch)
    lines = [
        f"nvcc {version}",
        f"flags {flags}",
        f"source {source_name} {hashlib.sha256(source_bytes).hexdigest()}",
        f"cubin {hashlib.sha256(cubin_bytes).hexdigest()} {len(cubin_bytes)}",
    ]
    shorts = {}
    for name in parameters:
        short = short_name(name)
        if short in shorts:
            raise ManifestError(f"kernels {shorts[short]} and {name} share the short name {short}")
        shorts[short] = name
    for name in sorted(parameters):
        fields = reports.get(name)
        if fields is None:
            raise ManifestError(f"the ptxas log has no {arch} report for {name}")
        missing = [field for field in REPORT_FIELDS if field not in fields]
        if missing:
            raise ManifestError(f"ptxas reported no {', '.join(missing)} for {name}")
        if "cumulative" in fields and fields["cumulative"] < fields["stack"]:
            raise ManifestError(f"ptxas reported {name}'s cumulative stack {fields['cumulative']} below its own frame {fields['stack']}")
        cumulative = f" cumulative {fields['cumulative']}" if "cumulative" in fields else ""
        lines.append(f"kernel {name} short {short_name(name)} registers {fields['registers']} barriers {fields['barriers']} stack {fields['stack']} "
                     f"spill_stores {fields['spill_stores']} spill_loads {fields['spill_loads']} smem {fields['smem']}{cumulative}")
        layout, bounds = parameters[name]
        for index, (kind, align, size, offset) in enumerate(layout):
            lines.append(f"param {name} {index} {kind} align {align} bytes {size} offset {offset}")
        for directive, values in bounds:
            lines.append(f"bound {name} {directive}" + "".join(f" {value}" for value in values))
    return "\n".join(lines) + "\n"


def main(argv):
    parser = argparse.ArgumentParser(description="write the nvrm kernel manifest for one translation unit's cubin")
    parser.add_argument("--cubin", required=True, type=Path)
    parser.add_argument("--ptx", required=True, type=Path, help="the PTX nvcc fed ptxas in the same compile (-keep)")
    parser.add_argument("--ptxas-log", required=True, type=Path, help="the compile's ptxas -v output")
    parser.add_argument("--nvcc", required=True, help="the nvcc that compiled the cubin")
    parser.add_argument("--flags", required=True, help="the compile's code-generation flags")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        text = manifest(nvcc_version(arguments.nvcc), arguments.flags, str(arguments.source), arguments.source.read_bytes(),
                        arguments.cubin.read_bytes(), arguments.ptx.read_text(), arguments.ptxas_log.read_text(errors="replace"), arguments.arch)
    except (ManifestError, OSError) as error:
        print(f"FAIL {error}")
        return 1
    arguments.output.write_text(text)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
