#!/usr/bin/env python3
"""Exercise the shared v2 parser with real GLM cardinality and corrupt files."""
import ctypes as C
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


class Range(C.Structure):
    _fields_ = [("offset", C.c_uint64), ("bytes", C.c_uint64),
                ("digest", C.c_uint8 * 16), ("layer", C.c_uint32),
                ("expert", C.c_uint32), ("kind", C.c_uint32)]


class Group(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in
                ("layer", "expert", "first_range", "range_count")]


class Span(C.Structure):
    _fields_ = [("offset", C.c_uint64), ("bytes", C.c_uint64),
                ("compact_offset", C.c_uint64)]


class Manifest(C.Structure):
    _fields_ = [("ranges", C.POINTER(Range)), ("groups", C.POINTER(Group)),
                ("range_count", C.c_uint32), ("group_count", C.c_uint32),
                ("spine", C.POINTER(Span)), ("spine_bytes", C.c_uint64),
                ("spine_count", C.c_uint32), ("spine_allocation_bytes", C.c_uint64)]


def record(layer, expert, kind, offset, size=32):
    return struct.pack("<4I2Q16s", layer, expert, kind, 0, offset, size,
                       bytes([kind % 256]) * 16)


def payload(records, version=2, reserved=0):
    return struct.pack("<4I", 0x58504557, version, len(records), reserved) + b"".join(records)


def main():
    with tempfile.TemporaryDirectory(prefix="weightd-manifest-") as directory:
        root = Path(directory)
        library = root / "manifest.so"
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-shared", "-fPIC", "-I", str(ROOT / "include"),
                        str(ROOT / "runtime/spark_weightd_manifest.c"),
                        "-o", str(library)], check=True)
        lib = C.CDLL(str(library))
        lib.SparkWeightdManifestLoad.argtypes = [C.c_char_p, C.c_uint64, C.POINTER(Manifest)]
        lib.SparkWeightdManifestLoad.restype = C.c_int
        lib.SparkWeightdManifestDestroy.argtypes = [C.POINTER(Manifest)]
        lib.SparkWeightdManifestFind.argtypes = [C.POINTER(Manifest), C.c_uint32, C.c_uint32]
        lib.SparkWeightdManifestFind.restype = C.POINTER(Group)
        lib.SparkWeightdManifestSpineSlice.argtypes = [C.POINTER(Manifest), C.c_uint64, C.c_uint64, C.POINTER(C.c_uint64)]
        lib.SparkWeightdManifestSpineSlice.restype = C.c_int
        path = root / "pack.experts"
        records = [record(layer, expert, kind, ((layer - 3) * 288 * 4 + expert * 4 + kind) * 64)
                   for layer in range(3, 45) for expert in range(288) for kind in range(4)]
        arena_bytes = len(records) * 64
        # Two weight and two scale ranges: cardinality beyond the old 4096 cap.
        path.write_bytes(payload(list(reversed(records))))
        result = Manifest()
        assert lib.SparkWeightdManifestLoad(bytes(path), arena_bytes, C.byref(result)) == 0
        assert result.range_count == 48384 and result.group_count == 12096
        assert result.spine_count == len(records)
        assert result.spine_bytes == arena_bytes // 2
        for index in range(result.spine_count):
            span = result.spine[index]
            assert (span.offset, span.bytes) == (index * 64 + 32, 32)
        for layer in range(3, 45):
            for expert in range(288):
                group = lib.SparkWeightdManifestFind(C.byref(result), layer, expert).contents
                assert group.range_count == 4
                for kind in range(4):
                    item = result.ranges[group.first_range + kind]
                    assert (item.layer, item.expert, item.kind) == (layer, expert, kind)
                    assert bytes(item.digest) == bytes([kind]) * 16
        assert not lib.SparkWeightdManifestFind(C.byref(result), 45, 0)
        lib.SparkWeightdManifestDestroy(C.byref(result))
        # Exact coverage, adjacent experts, leading/trailing bytes and wide offsets.
        for offsets, pack_bytes, expected in [
                ([0, 32], 64, []),
                ([32, 64], 128, [(0, 32), (96, 32)]),
                ([0, 64], 128, [(32, 32), (96, 32)]),
                ([(1 << 40)], (1 << 40) + 64, [(0, 1 << 40), ((1 << 40) + 32, 32)])]:
            path.write_bytes(payload([record(0, i, 0, off) for i, off in enumerate(offsets)]))
            assert lib.SparkWeightdManifestLoad(bytes(path), pack_bytes, C.byref(result)) == 0
            assert [(result.spine[i].offset, result.spine[i].bytes)
                    for i in range(result.spine_count)] == expected
            assert result.spine_bytes + len(offsets) * 32 == pack_bytes
            lib.SparkWeightdManifestDestroy(C.byref(result))
            assert not result.spine and result.spine_count == 0
        path.write_bytes(payload([record(0, 0, 0, 7, 17), record(0, 1, 0, 31, 9)]))
        assert lib.SparkWeightdManifestLoad(bytes(path), 80, C.byref(result)) == 0
        compact = C.c_uint64()
        previous_end = 0
        for i in range(result.spine_count):
            span = result.spine[i]
            assert span.compact_offset >= previous_end
            assert span.compact_offset % 256 == span.offset % 256
            previous_end = span.compact_offset + span.bytes
        assert previous_end == result.spine_allocation_bytes
        for offset in range(81):
            for size in range(1, 82 - offset):
                expected = offset + size <= 80 and not any(
                    offset < end and start < offset + size for start, end in [(7, 24), (31, 40)])
                status = lib.SparkWeightdManifestSpineSlice(C.byref(result), offset, size, C.byref(compact))
                assert (status == 0) == expected
                if expected:
                    assert compact.value % 256 == offset % 256
                    assert compact.value + size <= result.spine_allocation_bytes
                else:
                    assert compact.value == 0
        assert lib.SparkWeightdManifestSpineSlice(C.byref(result), 0, 0, C.byref(compact)) != 0
        assert lib.SparkWeightdManifestSpineSlice(C.byref(result), (1 << 64) - 1, 2, C.byref(compact)) != 0
        lib.SparkWeightdManifestDestroy(C.byref(result))
        good = [record(3, 0, 0, 0), record(3, 0, 1, 64)]
        malformed = [payload(good)[:-1], payload(good) + b"x",
                     payload(good, version=1), payload(good, reserved=1),
                     payload([]), payload([good[0], record(3, 1, 0, 16)]),
                     payload([good[0], record(3, 0, 0, 64)]),
                     payload([record(3, 0, 0, 128)]),
                     payload([record(3, 0, 0, 0, 0)]),
                     payload([record(3, 0, 0, (1 << 64) - 1)]),
                     struct.pack("<4I", 0x58504557, 2, 131073, 0)]
        for data in malformed:
            path.write_bytes(data)
            assert lib.SparkWeightdManifestLoad(bytes(path), 128, C.byref(result)) != 0
            assert not result.ranges and not result.groups and result.range_count == 0
        path.write_bytes(payload([record(3, 0, k, k * 64) for k in range(17)]))
        assert lib.SparkWeightdManifestLoad(bytes(path), 4096, C.byref(result)) != 0
        assert not result.ranges and not result.groups
        path.unlink()
        assert lib.SparkWeightdManifestLoad(bytes(path), 128, C.byref(result)) != 0
        import os
        os.mkfifo(path)
        assert lib.SparkWeightdManifestLoad(bytes(path), 128, C.byref(result)) != 0
        path.unlink()
        print("PASS v2 expert manifest: 12096 groups, 48384 ranges, lookup and corruption gates")


if __name__ == "__main__":
    main()
