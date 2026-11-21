#!/usr/bin/env python3
"""Rename CUDA runtime calls, types and constants to the spark_gpu.h names.

include/sparkpipe/spark_gpu.h gives every GPU runtime function, type and
constant the code uses a name of our own (SparkGpuMalloc, SparkGpuStream,
SPARK_GPU_SUCCESS and so on), so a device backend can implement them. This
tool rewrites the CUDA runtime names in the given files or directories to
those names, outside comments and string literals, and adds
`#include "sparkpipe/spark_gpu.h"` to every file it changes. Names spark_gpu.h
does not mirror are left as they are and listed.

Usage:
    python3 tools/gpu_api_sites.py --rewrite PATH...
    python3 tools/gpu_api_sites.py --list PATH...
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_HEADER = "sparkpipe/spark_gpu.h"
SUFFIXES = {".c", ".h", ".cu", ".cuh", ".cc", ".cpp", ".hpp"}

FUNCTIONS = (
    "GetLastError", "PeekAtLastError", "GetErrorString", "GetDeviceCount", "GetDevice", "SetDevice",
    "GetDeviceProperties", "DeviceGetAttribute", "DeviceGetStreamPriorityRange", "DeviceSynchronize", "MemGetInfo",
    "Malloc", "MallocManaged", "MallocHost", "HostAlloc", "HostGetDevicePointer", "HostRegister", "HostUnregister",
    "Free", "FreeHost", "Memcpy", "MemcpyAsync", "Memcpy2DAsync", "Memset", "MemsetAsync", "Memset2DAsync",
    "StreamCreateWithFlags", "StreamCreateWithPriority", "StreamDestroy", "StreamSynchronize", "StreamQuery",
    "StreamWaitEvent", "StreamIsCapturing", "StreamBeginCapture", "StreamEndCapture", "LaunchHostFunc",
    "EventCreate", "EventCreateWithFlags", "EventDestroy", "EventRecord", "EventQuery", "EventSynchronize",
    "EventElapsedTime", "GraphInstantiate", "GraphUpload", "GraphLaunch", "GraphGetNodes", "GraphExecDestroy",
    "GraphDestroy", "FuncSetAttribute", "FuncGetAttributes", "OccupancyMaxActiveBlocksPerMultiprocessor",
)

TYPES = {
    "cudaError_t": "SparkGpuError",
    "cudaStream_t": "SparkGpuStream",
    "cudaEvent_t": "SparkGpuEvent",
    "cudaGraph_t": "SparkGpuGraph",
    "cudaGraphExec_t": "SparkGpuGraphExec",
    "cudaGraphNode_t": "SparkGpuGraphNode",
    "cudaHostFn_t": "SparkGpuHostFunction",
}

TAGGED_TYPES = {
    "cudaMemcpyKind": "SparkGpuCopyKind",
    "cudaStreamCaptureStatus": "SparkGpuCaptureStatus",
    "cudaStreamCaptureMode": "SparkGpuCaptureMode",
    "cudaDeviceAttr": "SparkGpuDeviceAttribute",
    "cudaFuncAttribute": "SparkGpuFunctionAttribute",
    "cudaDeviceProp": "SparkGpuDeviceProperties",
    "cudaFuncAttributes": "SparkGpuFunctionAttributes",
}

CONSTANTS = {
    "cudaSuccess": "SPARK_GPU_SUCCESS",
    "cudaErrorInvalidValue": "SPARK_GPU_ERROR_INVALID_VALUE",
    "cudaErrorMemoryAllocation": "SPARK_GPU_ERROR_MEMORY_ALLOCATION",
    "cudaErrorInvalidDevice": "SPARK_GPU_ERROR_INVALID_DEVICE",
    "cudaErrorNotReady": "SPARK_GPU_ERROR_NOT_READY",
    "cudaErrorNotSupported": "SPARK_GPU_ERROR_NOT_SUPPORTED",
    "cudaErrorUnknown": "SPARK_GPU_ERROR_UNKNOWN",
    "cudaMemcpyHostToDevice": "SPARK_GPU_COPY_HOST_TO_DEVICE",
    "cudaMemcpyDeviceToHost": "SPARK_GPU_COPY_DEVICE_TO_HOST",
    "cudaMemcpyDeviceToDevice": "SPARK_GPU_COPY_DEVICE_TO_DEVICE",
    "cudaMemcpyDefault": "SPARK_GPU_COPY_DEFAULT",
    "cudaStreamNonBlocking": "SPARK_GPU_STREAM_NON_BLOCKING",
    "cudaEventDisableTiming": "SPARK_GPU_EVENT_DISABLE_TIMING",
    "cudaHostAllocDefault": "SPARK_GPU_HOST_ALLOC_DEFAULT",
    "cudaHostAllocPortable": "SPARK_GPU_HOST_ALLOC_PORTABLE",
    "cudaHostAllocMapped": "SPARK_GPU_HOST_ALLOC_MAPPED",
    "cudaHostRegisterDefault": "SPARK_GPU_HOST_REGISTER_DEFAULT",
    "cudaStreamCaptureStatusNone": "SPARK_GPU_CAPTURE_STATUS_NONE",
    "cudaStreamCaptureModeThreadLocal": "SPARK_GPU_CAPTURE_MODE_THREAD_LOCAL",
    "cudaDevAttrMultiProcessorCount": "SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT",
    "cudaDevAttrPageableMemoryAccess": "SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS",
    "cudaDevAttrPageableMemoryAccessUsesHostPageTables": "SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES",
    "cudaDevAttrCanUseHostPointerForRegisteredMem": "SPARK_GPU_ATTRIBUTE_HOST_POINTER_FOR_REGISTERED_MEMORY",
    "cudaFuncAttributeMaxDynamicSharedMemorySize": "SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES",
}

NAMES = dict(("cuda" + name, "SparkGpu" + name) for name in FUNCTIONS)
NAMES.update(TYPES)
NAMES.update(CONSTANTS)
NAMES.update(TAGGED_TYPES)
IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])cuda[A-Z][A-Za-z0-9_]*(?![A-Za-z0-9_])")
TAGGED = re.compile(r"\b(enum|struct)(\s+)(cuda[A-Z][A-Za-z0-9_]*)(?![A-Za-z0-9_])")
INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*["<]([^">]+)[">]', re.M)


def code_spans(text):
    spans = []
    index = 0
    start = 0
    length = len(text)
    while index < length:
        if text.startswith("//", index):
            spans.append((start, index))
            end = text.find("\n", index)
            index = start = length if end < 0 else end
        elif text.startswith("/*", index):
            spans.append((start, index))
            end = text.find("*/", index + 2)
            index = start = length if end < 0 else end + 2
        elif text[index] == '"' or (text[index] == "'" and not (index > 0 and text[index - 1].isalnum())):
            spans.append((start, index))
            quote = text[index]
            end = index + 1
            while end < length and text[end] != quote and text[end] != "\n":
                end += 2 if text[end] == "\\" else 1
            index = start = min(end + 1, length)
        elif text[index] == "#" and INCLUDE.match(text, text.rfind("\n", 0, index) + 1):
            spans.append((start, index))
            end = text.find("\n", index)
            index = start = length if end < 0 else end
        else:
            index += 1
    spans.append((start, length))
    return [(a, b) for a, b in spans if b > a]


def rename_code(code):
    code = TAGGED.sub(lambda match: TAGGED_TYPES[match.group(3)] if match.group(3) in TAGGED_TYPES else match.group(0), code)
    return IDENTIFIER.sub(lambda match: NAMES.get(match.group(0), match.group(0)), code)


def rewrite_text(text):
    pieces = []
    position = 0
    for start, end in code_spans(text):
        pieces.append(text[position:start])
        pieces.append(rename_code(text[start:end]))
        position = end
    pieces.append(text[position:])
    return "".join(pieces)


def remaining(text):
    names = {}
    for start, end in code_spans(text):
        for match in IDENTIFIER.finditer(text, start, end):
            names[match.group(0)] = names.get(match.group(0), 0) + 1
    return names


def guard_closes_at_end(lines):
    depth = 0
    closed = None
    for index, line in enumerate(lines):
        directive = re.match(r"\s*#[ \t]*(\w+)", line)
        if not directive:
            continue
        if directive.group(1) in ("if", "ifdef", "ifndef"):
            depth += 1
        elif directive.group(1) == "endif":
            depth -= 1
            if depth == 0 and closed is None:
                closed = index
    return closed is not None and all(not line.strip() for line in lines[closed + 1:])


def add_include(text):
    if re.search(r'^[ \t]*#[ \t]*include[ \t]*"' + re.escape(GPU_HEADER) + '"', text, re.M):
        return text
    lines = text.split("\n")
    depth = 0
    top = 0
    last = None
    anchor = None
    guard = re.match(r"\s*#[ \t]*ifndef[ \t]+(\w+)[ \t]*\n\s*#[ \t]*define[ \t]+(\w+)", text)
    if guard and guard.group(1) == guard.group(2) and guard_closes_at_end(lines):
        top = 1
    for index, line in enumerate(lines):
        stripped = line.strip()
        directive = re.match(r"#[ \t]*(\w+)", stripped)
        if directive:
            word = directive.group(1)
            if word in ("if", "ifdef", "ifndef"):
                depth += 1
            elif word == "endif":
                depth -= 1
            elif word == "include" and depth == top:
                last = index
            elif (word == "pragma" and "once" in stripped or word == "define" and depth == top == 1) and anchor is None:
                anchor = index
            continue
        if not stripped or re.match(r"(?:struct|class)\s+\w+\s*;$", stripped):
            continue
        break
    if last is None:
        if anchor is None:
            raise SystemExit("no include block or #pragma once to extend")
        last = anchor
    lines.insert(last + 1, '#include "%s"' % GPU_HEADER)
    return "\n".join(lines)


def files(paths):
    for raw in paths:
        path = (ROOT / raw).resolve()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix in SUFFIXES:
                    yield child
        elif path.is_file():
            yield path
        else:
            raise SystemExit("%s: no such file or directory" % raw)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rewrite", action="store_true")
    mode.add_argument("--list", action="store_true")
    parser.add_argument("paths", nargs="+")
    options = parser.parse_args(argv)
    total = 0
    refused = 0
    for path in files(options.paths):
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(errors="surrogateescape")
        if options.rewrite:
            updated = rewrite_text(text)
            if updated != text:
                try:
                    updated = add_include(updated)
                except SystemExit as error:
                    print("%s: %s; left unchanged" % (relative, error), file=sys.stderr)
                    refused += 1
                    continue
                path.write_text(updated, errors="surrogateescape")
                changed = sum(remaining(text).values()) - sum(remaining(updated).values())
                total += changed
                print("%s: %d names" % (relative, changed))
            text = updated
        left = remaining(text)
        if left:
            print("%s: not mirrored: %s" % (relative, ", ".join("%s x%d" % item for item in sorted(left.items()))))
    if options.rewrite:
        print("renamed %d names" % total)
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
