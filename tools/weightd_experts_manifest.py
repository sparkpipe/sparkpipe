#!/usr/bin/env python3
"""Write the weightd .experts manifest for a stagepack to STDOUT — the
placement-time sidecar that makes ATTACH_LAZY work. Redirect stdout to
<p>.experts beside the pack at the placement step.

weightd is model-generic and cannot parse family pack directories, so the
placement (which owns the format) emits one 40-byte record per
(layer, expert_key): {u32 layer, u32 expert_key, u64 offset, u64 bytes,
u8[16] ck128}. expert_key = expert * parts + part_index keeps the daemon's
(layer, key) lookup unique with zero daemon changes. Blocks are contiguous
per-expert slices of the pack image (expert-major stacking survives TP row
sharding for W1/W3 and W2's expert-major-row layout), and Ensure commits
bytes at canonical arena offsets, so consumer pointer arithmetic is
unchanged.

The ck128 digests are the install/acceptance record; runtime Ensure only
compares them when SPARK_WEIGHTD_VERIFY=1. Hashing runs through the repo's
own src/spark_ck128.c compiled at tool time (exact + fast).

usage: weightd_experts_manifest.py --pack P --experts 256 --kinds 19,20,21 \
           > P.experts
"""
import argparse
import ctypes
import os
import struct
import subprocess
import sys
import tempfile

PACK_MAGIC = 0x34565344
HEADER = struct.Struct("<16I2Q")
ENTRY = struct.Struct("<6I2Q")
MANIFEST_MAGIC = 0x58504557
MANIFEST_VERSION = 1
RECORD = struct.Struct("<IIQQ16s")
COUNT_MAX = 65535
BLOCK_BYTES_MAX = 64 * 1024 * 1024


class SparkCk128Context(ctypes.Structure):
    _fields_ = [("h1", ctypes.c_uint64), ("h2", ctypes.c_uint64),
                ("total_bytes", ctypes.c_uint64),
                ("tail_bytes", ctypes.c_uint32),
                ("tail", ctypes.c_uint8 * 16)]


def load_ck128(repo_root):
    source = os.path.join(repo_root, "src", "spark_ck128.c")
    build_dir = tempfile.mkdtemp(prefix="ck128_")
    library = os.path.join(build_dir, "libspark_ck128.so")
    subprocess.run(["cc", "-O2", "-fPIC", "-shared", "-I", repo_root,
        "-I", os.path.join(repo_root, "include"), source, "-o", library],
        check=True)
    lib = ctypes.CDLL(library)
    lib.SparkCk128Initialize.argtypes = [ctypes.POINTER(SparkCk128Context)]
    lib.SparkCk128Update.argtypes = [ctypes.POINTER(SparkCk128Context),
        ctypes.c_void_p, ctypes.c_size_t]
    lib.SparkCk128Finalize.argtypes = [ctypes.POINTER(SparkCk128Context),
        ctypes.c_char_p]
    return lib


def ck128(lib, data):
    context = SparkCk128Context()
    lib.SparkCk128Initialize(ctypes.byref(context))
    lib.SparkCk128Update(ctypes.byref(context), data, len(data))
    digest = ctypes.create_string_buffer(16)
    lib.SparkCk128Finalize(ctypes.byref(context), digest)
    return digest.raw


def payload_bytes(weight_format, rows, columns):
    elements = rows * columns
    if weight_format == 3:
        return elements // 2
    if weight_format in (1, 2):
        return elements * 4
    if weight_format == 4:
        return elements
    return elements * 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--kinds", required=True,
        help="comma-separated expert tensor kinds, e.g. 19,20,21")
    args = parser.parse_args()
    kinds = [int(k) for k in args.kinds.split(",") if k != ""]
    assert kinds, "no kinds parsed"

    pack_path = os.path.realpath(args.pack)
    assert os.path.isfile(pack_path), pack_path

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ck = load_ck128(repo_root)

    file_bytes = os.path.getsize(pack_path)
    records = []
    with open(pack_path, "rb") as pack:
        header = pack.read(HEADER.size)
        assert len(header) == HEADER.size, "short pack header"
        fields = HEADER.unpack(header)
        assert fields[0] == PACK_MAGIC, "not a stagepack"
        entry_count = fields[8]
        directory_offset = struct.unpack_from("<Q", header, 64)[0]
        assert directory_offset + ENTRY.size * entry_count <= file_bytes, \
            "directory past EOF"
        pack.seek(directory_offset)
        directory = pack.read(ENTRY.size * entry_count)

        planned = []
        for index in range(entry_count):
            kind, layer, weight_format, rows, columns, _reserved, poff, soff = \
                ENTRY.unpack_from(directory, index * ENTRY.size)
            if kind not in kinds:
                continue
            entry_bytes = payload_bytes(weight_format, rows, columns)
            if entry_bytes < args.experts:
                continue
            block_bytes = entry_bytes // args.experts
            assert 0 < block_bytes <= BLOCK_BYTES_MAX, \
                f"block {block_bytes} outside (0, 64MiB] kind={kind} layer={layer}"
            assert poff + entry_bytes <= file_bytes, "payload past EOF"
            planned.append((layer, kind, poff, block_bytes))

        total = len(planned)
        assert 0 < total <= COUNT_MAX, \
            f"record count {total} outside (0, {COUNT_MAX}]"
        for layer, kind, poff, block_bytes in planned:
            for expert in range(args.experts):
                pack.seek(poff + expert * block_bytes)
                block = pack.read(block_bytes)
                assert len(block) == block_bytes, "short payload read"
                key = expert * len(kinds) + kinds.index(kind)
                records.append(RECORD.pack(layer, key, poff + expert *
                    block_bytes, block_bytes, ck128(ck, block)))

    sys.stdout.buffer.write(struct.pack("<III", MANIFEST_MAGIC,
        MANIFEST_VERSION, len(records)))
    for record in records:
        sys.stdout.buffer.write(record)
    sys.stdout.buffer.flush()
    print(f"experts manifest: {len(records)} records "
        f"({len(planned)} tensors x {args.experts} experts) "
        f"for {pack_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
