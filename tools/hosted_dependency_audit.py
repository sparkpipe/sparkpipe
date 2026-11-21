#!/usr/bin/env python3

import argparse
import json
import os
import re
import struct
import sys


LIBRARY_STEMS = (
    "cuda[A-Za-z0-9_]*", "cudla[A-Za-z0-9_]*", "nvidia-[A-Za-z0-9_+-]+",
    "(?:GLX|EGL|GLESv1_CM|GLESv2|vdpau)_nvidia", "nvcuvid", "nvoptix",
    "nvrtc[A-Za-z0-9_-]*", "nvJitLink[A-Za-z0-9_]*", "nvjitlink[A-Za-z0-9_]*",
    "nvptxcompiler[A-Za-z0-9_]*", "nvvm", "nvfatbin[A-Za-z0-9_]*",
    "cublas[A-Za-z0-9_]*", "cufft[A-Za-z0-9_]*", "curand[A-Za-z0-9_]*",
    "cusparse[A-Za-z0-9_]*", "cusolver[A-Za-z0-9_]*", "cudss[A-Za-z0-9_]*",
    "cutensor[A-Za-z0-9_]*", "cufile[A-Za-z0-9_]*", "nvblas", "npp[a-z]*",
    "nvjpeg[A-Za-z0-9_]*", "cudnn[A-Za-z0-9_]*", "nccl[A-Za-z0-9_-]*",
    "nvshmem[A-Za-z0-9_]*", "cupti[A-Za-z0-9_]*", "nvperf_(?:host|target)",
    "pcsamplingutil", "nvToolsExt[A-Za-z0-9_]*", "nvtx[A-Za-z0-9_]*",
    "nvinfer[A-Za-z0-9_]*", "nvonnxparser[A-Za-z0-9_]*", "nvparsers",
    "nvcomp[A-Za-z0-9_]*", "dcgm[A-Za-z0-9_]*", "accinj(?:32|64)", "cuinj(?:32|64)",
    "torch_cuda[A-Za-z0-9_]*", "c10_cuda", "caffe2_nvrtc", "triton",
)
LIBRARY_ALTERNATION = "|".join(LIBRARY_STEMS)
LIBRARY_FILE = re.compile(rf"lib(?:{LIBRARY_ALTERNATION})\.(?:so(?:\.[0-9][0-9.]*)?|a)")
LIBRARY_STRING = re.compile(rf"lib(?:{LIBRARY_ALTERNATION})\.so(?:\.[0-9]+)*(?![A-Za-z0-9_])".encode())
TOOL_FILES = (
    "nvidia-smi", "nvidia-persistenced", "nvidia-debugdump", "nvidia-cuda-mps-control",
    "nvidia-cuda-mps-server", "nvidia-bug-report.sh", "nvidia-ctk", "nvidia-cdi-hook",
    "nvidia-container-cli", "nvidia-container-runtime", "nvidia-container-runtime-hook",
    "nvidia-container-toolkit", "nvidia-settings", "nvidia-xconfig", "nvidia-powerd",
    "nvidia-sleep.sh", "nv-fabricmanager", "nv-hostengine", "dcgmi", "nvidia-imex",
    "nvidia-imex-ctl",
)
TOOL_PREFIXES = sorted({name.split("-")[0] + "-" if "-" in name else name for name in TOOL_FILES})
TOOL_STRINGS = tuple(
    re.compile(re.escape(prefix.encode()) + rb"(?:" + b"|".join(
        re.escape(name[len(prefix):].encode()) for name in TOOL_FILES if name.startswith(prefix)) + rb")(?![A-Za-z0-9_-])")
    for prefix in TOOL_PREFIXES)
NAME_CHARACTERS = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.+-")
KERNEL_UAPI_HEADERS = ("include/linux/cuda.h",)
TOOLKIT_FILES = (
    "nvcc", "ptxas", "nvlink", "fatbinary", "cuobjdump", "nvdisasm", "cicc", "cudafe++",
    "nvprune", "cu++filt", "cuda-gdb", "cuda-gdbserver", "compute-sanitizer", "nsys", "ncu",
    "nvprof", "bin2c", "cuda.h", "cuda_runtime.h", "cuda_runtime_api.h", "nvml.h",
)
CUDA_DIRECTORY = re.compile(r"(?:^|/)cuda(?:-[0-9][0-9.]*)?(?:/|$)")
TORCH_CUDA_VERSION = re.compile(r"^cuda\s*(?::[^=]*)?=\s*(.+?)\s*$", re.M)
PACKAGE_STEMS = (
    "cuda", "cudart", "cublas", "cufft", "curand", "cusparse", "cusolver", "cudnn", "cutensor",
    "cufile", "cudss", "cupti", "cudla", "nccl", "npp", "nvjitlink", "nvjpeg", "nvptxcompiler",
    "nvvm", "nvfatbin", "nvrtc", "nvtoolsext", "nvinfer", "nvonnxparsers", "nvparsers",
    "nvblas", "nvshmem", "nvcomp", "nvidia-", "dcgm", "accinj", "cuinj",
)
CLOSED_KERNEL_MODULE = re.compile(r"linux-(?:modules|objects)-nvidia-[0-9]+(?:-server)?-(?!open-|server-)")
ALLOWED_PACKAGES = re.compile(
    r"linux-[a-z0-9.+-]+"
    r"|nvidia-firmware-[0-9]+(?:-server)?-[0-9.]+"
    r"|nvidia-kernel-source-[0-9]+(?:-server)?-open"
    r"|nvidia-dkms-[0-9]+(?:-server)?-open")
FORBIDDEN_PACKAGES = (
    ("driver user-space library", re.compile(r"lib(?:" + "|".join(map(re.escape, PACKAGE_STEMS)) + ")")),
    ("NVIDIA user-space package", re.compile(r"nvidia-|xserver-xorg-video-nvidia|nsight-|nvhpc|tensorrt|datacenter-gpu-manager")),
    ("CUDA toolkit package", re.compile(r"cuda(?:-|$)|python3-(?:torch-cuda|pycuda|cupy)")),
    ("DGX OS package", re.compile(r"dgx|nvsm")),
    ("DOCA host package", re.compile(r"doca|libdoca|mlnx|ofed|mft$|mft-|kernel-mft|xlio|libxlio|dpcp|libdpcp|hcoll|sharp$|clusterkit|ibutils2")),
)
VENDOR_VERSION = re.compile(r"nvidia|mlnx|doca", re.I)
APT_VENDOR_SOURCE = re.compile(
    r"(?:^|[/.@])(?:nvidia\.com|nvidia\.github\.io|mellanox\.com|nvcr\.io)(?:[/:\s]|$)|/canonical-nvidia/", re.I)
IMAGE_SKIPPED_DIRECTORIES = ("proc", "sys", "dev")
IMAGE_NAME_ONLY_PREFIXES = ("lib/firmware/", "usr/lib/firmware/", "lib/modules/", "usr/lib/modules/", "usr/src/")
DT_NEEDED, DT_STRTAB, DT_STRSZ, DT_SONAME, DT_RPATH, DT_RUNPATH = 1, 5, 10, 14, 15, 29
LINK_HOPS = 40
SCAN_CHUNK = 1 << 22
SCAN_OVERLAP = 256


class ElfError(Exception):
    pass


class Report:
    def __init__(self, mode, target):
        self.mode = mode
        self.target = target
        self.scanned = 0
        self.found = set()

    def add(self, kind, path, detail):
        self.found.add((path, kind, detail))

    def require_content(self, path):
        if self.scanned == 0:
            self.add("empty", path, "nothing to audit")

    def findings(self):
        return [{"kind": kind, "path": path, "detail": detail} for path, kind, detail in sorted(self.found)]

    def verdict(self):
        return "FAIL" if self.found else "PASS"

    def document(self):
        return {"mode": self.mode, "target": self.target, "scanned": self.scanned,
                "findings": self.findings(), "verdict": self.verdict()}


def usage_failure(message):
    print(f"hosted-dependency-audit: {message}", file=sys.stderr)
    raise SystemExit(2)


def forbidden_library(name):
    return LIBRARY_FILE.fullmatch(os.path.basename(name)) is not None


def read_at(source, offset, size):
    source.seek(offset)
    data = source.read(size)
    if len(data) != size:
        raise ElfError(f"truncated at offset {offset}")
    return data


def elf_dynamic(source):
    ident = read_at(source, 0, 16)
    if ident[:4] != b"\x7fELF" or ident[4] not in (1, 2) or ident[5] not in (1, 2):
        raise ElfError("bad ELF identification")
    wide = ident[4] == 2
    order = "<" if ident[5] == 1 else ">"
    if wide:
        header = struct.unpack(order + "HHIQQQIHHHHHH", read_at(source, 16, 48))
        program_format, dynamic_format = order + "IIQQQQQQ", order + "qQ"
    else:
        header = struct.unpack(order + "HHIIIIIHHHHHH", read_at(source, 16, 36))
        program_format, dynamic_format = order + "IIIIIIII", order + "iI"
    phoff, phentsize, phnum = header[4], header[8], header[9]
    if phnum == 0:
        return [], "", []
    if phnum == 0xffff or phentsize < struct.calcsize(program_format):
        raise ElfError("unsupported program header table")
    loads, dynamic = [], None
    for index in range(phnum):
        fields = struct.unpack(program_format, read_at(source, phoff + index * phentsize, struct.calcsize(program_format)))
        if wide:
            kind, offset, vaddr, filesz = fields[0], fields[2], fields[3], fields[5]
        else:
            kind, offset, vaddr, filesz = fields[0], fields[1], fields[2], fields[4]
        if kind == 1:
            loads.append((vaddr, offset, filesz))
        elif kind == 2:
            dynamic = (offset, filesz)
    if dynamic is None:
        return [], "", []
    entry_size = struct.calcsize(dynamic_format)
    entries = []
    for index in range(dynamic[1] // entry_size):
        tag, value = struct.unpack(dynamic_format, read_at(source, dynamic[0] + index * entry_size, entry_size))
        if tag == 0:
            break
        entries.append((tag, value))
    table = [value for tag, value in entries if tag == DT_STRTAB]
    size = [value for tag, value in entries if tag == DT_STRSZ]
    if len(table) != 1 or len(size) != 1:
        raise ElfError("dynamic section has no single string table")
    base = [offset + table[0] - vaddr for vaddr, offset, filesz in loads if vaddr <= table[0] < vaddr + filesz]
    if not base:
        raise ElfError("string table is outside every loaded segment")
    strings = read_at(source, base[0], size[0])

    def text(value):
        end = strings.find(b"\0", value)
        if value >= len(strings) or end < 0:
            raise ElfError(f"string offset {value} is outside the string table")
        return strings[value:end].decode("utf-8", "replace")

    needed = [text(value) for tag, value in entries if tag == DT_NEEDED]
    soname = "".join(text(value) for tag, value in entries if tag == DT_SONAME)
    paths = [item for tag, value in entries if tag in (DT_RPATH, DT_RUNPATH) for item in text(value).split(":")]
    return needed, soname, paths


def standalone(window, matches):
    return {match.group(0).decode() for match in matches
            if match.start() == 0 or window[match.start() - 1] not in NAME_CHARACTERS}


def scan_strings(source):
    libraries, tools = set(), set()
    source.seek(0)
    tail = b""
    while True:
        chunk = source.read(SCAN_CHUNK)
        if not chunk:
            break
        window = tail + chunk
        libraries |= standalone(window, LIBRARY_STRING.finditer(window))
        for pattern in TOOL_STRINGS:
            tools |= standalone(window, pattern.finditer(window))
        tail = window[-SCAN_OVERLAP:]
    return libraries, tools


def forbidden_name(path):
    name = os.path.basename(path.rstrip("/"))
    if LIBRARY_FILE.fullmatch(name):
        return "library-file"
    if name in TOOL_FILES:
        return "tool-file"
    if name in TOOLKIT_FILES and not path.endswith(KERNEL_UAPI_HEADERS):
        return "toolkit-file"
    return ""


def classify_name(report, relative, name):
    kind = forbidden_name(relative)
    if kind:
        report.add(kind, relative, name)
    return bool(kind)


def link_targets(top, path, rooted):
    targets = []
    current = path
    for _ in range(LINK_HOPS):
        try:
            target = os.readlink(current)
        except OSError:
            return targets
        targets.append(target)
        if os.path.isabs(target):
            if not rooted:
                return targets
            current = os.path.join(top, target.lstrip("/"))
        else:
            current = os.path.join(os.path.dirname(current), target)
        current = os.path.normpath(current)
        if current != top and not current.startswith(top + os.sep):
            return targets
    raise OSError(f"more than {LINK_HOPS} links from {path}")


def classify_link(report, top, relative, path, rooted):
    for target in link_targets(top, path, rooted):
        kind = forbidden_name(target)
        if kind:
            report.add(kind, relative, f"link to {target}")


def classify_python_package(report, relative, source):
    if relative == "torch/version.py" or relative.endswith("/torch/version.py"):
        match = TORCH_CUDA_VERSION.search(source.read().decode("utf-8", "replace"))
        if match and match.group(1) not in ("None", "'None'", '"None"'):
            report.add("pytorch-cuda", relative, f"cuda = {match.group(1)}")
    elif relative == "triton/__init__.py" or relative.endswith("/triton/__init__.py"):
        report.add("triton", relative, "Triton package")


def classify_elf(report, relative, source):
    needed, soname, paths = elf_dynamic(source)
    for name in needed:
        if forbidden_library(name):
            report.add("needed", relative, name)
    for path in paths:
        if CUDA_DIRECTORY.search(path):
            report.add("runpath", relative, path)
    if forbidden_library(soname):
        report.add("soname", relative, soname)
        return None
    return set(needed) | {soname}


def audit_file(report, relative, path, scan):
    with open(path, "rb") as source:
        classify_python_package(report, relative, source)
        source.seek(0)
        known = set()
        if source.read(4) == b"\x7fELF":
            known = classify_elf(report, relative, source)
            if known is None:
                return
        if not scan:
            return
        libraries, tools = scan_strings(source)
    for name in sorted(libraries - known):
        report.add("library-string", relative, name)
    for name in sorted(tools):
        report.add("tool-string", relative, name)


def entries(top, skipped):
    stack = [top]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as listing:
                items = sorted(listing, key=lambda item: item.name, reverse=True)
        except OSError as error:
            yield directory, None, error
            continue
        for item in items:
            relative = os.path.relpath(item.path, top).replace(os.sep, "/")
            if item.is_dir(follow_symlinks=False):
                if relative not in skipped:
                    stack.append(item.path)
            else:
                yield item.path, item, None


def walk(report, top, skipped, scan, name_only_prefixes, rooted):
    top = os.path.abspath(top)
    for path, item, error in entries(top, skipped):
        relative = os.path.relpath(path, top).replace(os.sep, "/")
        if error is not None:
            report.add("unreadable", relative, str(error))
            continue
        report.scanned += 1
        if classify_name(report, relative, item.name):
            continue
        if item.is_symlink():
            try:
                classify_link(report, top, relative, path, rooted)
            except OSError as error:
                report.add("unreadable", relative, str(error))
            continue
        if not item.is_file(follow_symlinks=False) or relative.startswith(name_only_prefixes):
            continue
        try:
            audit_file(report, relative, path, scan)
        except ElfError as error:
            report.add("unparsed", relative, str(error))
        except OSError as error:
            report.add("unreadable", relative, str(error))


def audit_root(root):
    report = Report("root", root)
    walk(report, root, (), True, (), False)
    report.require_content(".")
    return report


def dpkg_records(text):
    for paragraph in re.split(r"\n\s*\n", text):
        fields = dict(re.findall(r"^([A-Za-z0-9-]+):[ \t]*(.*)$", paragraph, re.M))
        if "Package" in fields:
            yield fields


def package_reason(name, version):
    if CLOSED_KERNEL_MODULE.match(name):
        return "closed kernel module package"
    if ALLOWED_PACKAGES.fullmatch(name) is None:
        for reason, pattern in FORBIDDEN_PACKAGES:
            if pattern.match(name):
                return reason
    if VENDOR_VERSION.search(version):
        return "vendor build"
    return ""


def audit_packages(report, path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as source:
            text = source.read()
    except OSError as error:
        report.add("unreadable", path, str(error))
        return
    for fields in dpkg_records(text):
        state = fields.get("Status", "").split()
        if state[-1:] == ["not-installed"]:
            continue
        report.scanned += 1
        reason = package_reason(fields["Package"], fields.get("Version", ""))
        if reason:
            report.add("package", fields["Package"], f"{reason}: {fields.get('Version', '')} {' '.join(state[-1:])}")


def audit_apt_sources(report, rootfs):
    paths = [os.path.join(rootfs, "etc/apt/sources.list")]
    directory = os.path.join(rootfs, "etc/apt/sources.list.d")
    if os.path.isdir(directory):
        paths += [os.path.join(directory, name) for name in sorted(os.listdir(directory))]
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as source:
            for line in source:
                if APT_VENDOR_SOURCE.search(line):
                    report.add("apt-source", os.path.relpath(path, rootfs), line.strip())


def audit_image(rootfs, dpkg_status, release_roots):
    if dpkg_status is not None:
        report = Report("dpkg-status", dpkg_status)
        audit_packages(report, dpkg_status)
        report.require_content(dpkg_status)
        return report
    if not os.path.isdir(rootfs):
        usage_failure(f"image rootfs {rootfs} is not a directory")
    report = Report("image", rootfs)
    status = os.path.join(rootfs, "var/lib/dpkg/status")
    if not os.path.isfile(status):
        report.add("package-database", "var/lib/dpkg/status", "missing")
    else:
        audit_packages(report, status)
    audit_apt_sources(report, rootfs)
    walk(report, rootfs, IMAGE_SKIPPED_DIRECTORIES, False, IMAGE_NAME_ONLY_PREFIXES, True)
    report.require_content(".")
    for relative in release_roots:
        if not os.path.isdir(os.path.join(rootfs, relative)):
            report.add("missing", relative, "release root is not a directory")
            continue
        root = audit_root(os.path.join(rootfs, relative))
        for path, kind, detail in root.found:
            report.add(kind, os.path.normpath(f"{relative}/{path}"), detail)
    return report


def process_ids(proc, selected):
    if selected:
        return sorted(selected)
    return sorted(int(name) for name in os.listdir(proc) if name.isdigit())


def audit_processes(proc, selected):
    report = Report("processes", proc)
    if not os.path.isdir(proc):
        usage_failure(f"process table {proc} is not a directory")
    for pid in process_ids(proc, selected):
        base = os.path.join(proc, str(pid))
        try:
            with open(os.path.join(base, "maps"), "r", encoding="utf-8", errors="replace") as source:
                mapped = {line.split(None, 5)[5].strip() for line in source if len(line.split(None, 5)) == 6}
        except (FileNotFoundError, ProcessLookupError):
            if selected:
                report.add("unreadable", str(pid), "process does not exist")
            continue
        except OSError as error:
            report.add("unreadable", str(pid), str(error))
            continue
        report.scanned += 1
        for path in sorted(mapped):
            name = os.path.basename(path.removesuffix(" (deleted)"))
            if forbidden_library(name) or name in TOOL_FILES:
                report.add("mapped", str(pid), f"{process_name(base)} maps {path}")
    report.require_content(proc)
    return report


def process_name(base):
    try:
        with open(os.path.join(base, "comm"), "r", encoding="utf-8", errors="replace") as source:
            return source.read().strip()
    except OSError:
        return "?"


def print_report(report, as_json):
    if as_json:
        print(json.dumps(report.document(), indent=2, sort_keys=True))
        return
    findings = report.findings()
    for finding in findings:
        print(f"FAIL {finding['kind']} {finding['path']}: {finding['detail']}")
    print(f"hosted-dependency-audit {report.mode} {report.target}: {report.verdict()} "
          f"({report.scanned} scanned, {len(findings)} findings)")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Dependency audit for hosted release roots, node images and running nodes.")
    parser.add_argument("--json", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    root = commands.add_parser("root")
    root.add_argument("path")
    processes = commands.add_parser("processes")
    processes.add_argument("--proc", default="/proc")
    processes.add_argument("--pid", type=int, action="append", default=[])
    image = commands.add_parser("image")
    source = image.add_mutually_exclusive_group(required=True)
    source.add_argument("--rootfs")
    source.add_argument("--dpkg-status")
    image.add_argument("--release-root", action="append", default=[])
    arguments = parser.parse_args(argv)
    if arguments.command == "image" and arguments.dpkg_status is not None and arguments.release_root:
        parser.error("--release-root needs --rootfs")
    return arguments


def main(argv):
    arguments = parse_args(argv)
    if arguments.command == "root":
        if not os.path.isdir(arguments.path):
            usage_failure(f"release root {arguments.path} is not a directory")
        report = audit_root(arguments.path)
    elif arguments.command == "processes":
        report = audit_processes(arguments.proc, arguments.pid)
    else:
        report = audit_image(arguments.rootfs, arguments.dpkg_status, arguments.release_root)
    print_report(report, arguments.json)
    return 0 if report.verdict() == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
