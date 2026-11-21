#!/usr/bin/env python3
"""Check that every module in a staged root that embeds device code has its kernels beside it.

The nvrm backend loads a module's kernels from `<module path>.kernels/`, where the
path is the one the module is loaded by: its staged name, not its build name. Each
`.cubin` there needs its `.manifest`, and each manifest its cubin. This walks a
staged root, reads the section headers of every ELF file outside the kernels
directories, and treats a file with a `.nv_fatbin` or `__nv_relfatbin` section as
one that embeds device code. Every such file needs a `.kernels` directory beside it
under its own name, holding at least one cubin and only complete pairs. A kernels
directory with no module of its name beside it is reported too, since it is what a
module staged under another name leaves behind.

Usage:
    python3 -I tools/staged_kernels_audit.py ROOT [--quiet]

It exits 0 when every module with device code has its kernels, and 1 otherwise,
printing one line per finding.
"""
import argparse
import struct
import sys
from pathlib import Path

DEVICE_SECTIONS = (".nv_fatbin", "__nv_relfatbin")
KERNELS_SUFFIX = ".kernels"


class AuditError(Exception):
    pass


def section_names(path):
    with open(path, "rb") as stream:
        header = stream.read(64)
        if len(header) < 64 or header[:4] != b"\x7fELF":
            return None
        if header[4] != 2 or header[5] != 1:
            raise AuditError("%s is an ELF file but not a little-endian 64-bit one" % path)
        section_offset = struct.unpack_from("<Q", header, 0x28)[0]
        entry_size, entry_count, names_index = struct.unpack_from("<HHH", header, 0x3A)
        if section_offset == 0 or entry_count == 0:
            return []
        if entry_size < 64 or names_index >= entry_count:
            raise AuditError("%s has a malformed section header table" % path)
        stream.seek(section_offset)
        table = stream.read(entry_size * entry_count)
        if len(table) < entry_size * entry_count:
            raise AuditError("%s ends inside its section header table" % path)
        entries = [struct.unpack_from("<IIQQQQIIQQ", table, index * entry_size) for index in range(entry_count)]
        stream.seek(entries[names_index][4])
        names = stream.read(entries[names_index][5])
    result = []
    for entry in entries:
        start = entry[0]
        end = names.find(b"\0", start)
        result.append(names[start:end if end >= 0 else len(names)].decode("ascii", "replace"))
    return result


def inside_kernels(path, root):
    return any(part.endswith(KERNELS_SUFFIX) for part in path.relative_to(root).parts[:-1])


def kernels_findings(kernels, relative):
    if not kernels.is_dir():
        return ["%s embeds device code but has no %s%s/ beside it under the name it is loaded by" % (relative, relative.name, KERNELS_SUFFIX)]
    entries = sorted(child.name for child in kernels.iterdir())
    cubins = {name[:-len(".cubin")] for name in entries if name.endswith(".cubin")}
    manifests = {name[:-len(".manifest")] for name in entries if name.endswith(".manifest")}
    findings = []
    if not cubins:
        findings.append("%s%s/ beside %s holds no cubin" % (relative.name, KERNELS_SUFFIX, relative))
    for stem in sorted(cubins - manifests):
        findings.append("%s%s/%s.cubin has no manifest" % (relative, KERNELS_SUFFIX, stem))
    for stem in sorted(manifests - cubins):
        findings.append("%s%s/%s.manifest has no cubin" % (relative, KERNELS_SUFFIX, stem))
    for name in entries:
        if not name.endswith((".cubin", ".manifest")):
            findings.append("%s%s/%s is neither a cubin nor a manifest" % (relative, KERNELS_SUFFIX, name))
    return findings


def audit(root):
    root = Path(root)
    if not root.is_dir():
        raise AuditError("%s is not a directory" % root)
    findings = []
    modules = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or inside_kernels(path, root):
            continue
        names = section_names(path)
        if names is None or not any(name in DEVICE_SECTIONS for name in names):
            continue
        relative = path.relative_to(root)
        modules.append(relative)
        findings += kernels_findings(path.parent / (path.name + KERNELS_SUFFIX), relative)
    for kernels in sorted(root.rglob("*" + KERNELS_SUFFIX)):
        if kernels.is_dir() and not inside_kernels(kernels, root):
            module = kernels.parent / kernels.name[:-len(KERNELS_SUFFIX)]
            if not module.is_file():
                findings.append("%s/ has no module %s beside it; a module staged under another name loads none of it" % (kernels.relative_to(root), module.name))
    return modules, findings


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root")
    parser.add_argument("--quiet", action="store_true")
    options = parser.parse_args(argv)
    try:
        modules, findings = audit(options.root)
    except (AuditError, OSError) as error:
        print("FAIL %s" % error)
        return 1
    for finding in findings:
        print("FAIL %s" % finding)
    if findings:
        return 1
    if not options.quiet:
        print("PASS %s: %d modules embed device code, each with its kernels beside it under its staged name" % (options.root, len(modules)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
