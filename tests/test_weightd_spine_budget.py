#!/usr/bin/env python3
"""The launcher spine budget equals the runtime's compact spine allocation plus alignment slack."""
import ctypes as C
import importlib.util
import random
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Manifest(C.Structure):
    _fields_ = [("ranges", C.c_void_p), ("groups", C.c_void_p),
                ("range_count", C.c_uint32), ("group_count", C.c_uint32),
                ("spine", C.c_void_p), ("spine_bytes", C.c_uint64),
                ("spine_count", C.c_uint32), ("spine_allocation_bytes", C.c_uint64)]


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "weightd_spine_budget", ROOT / "tools/weightd_spine_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_bytes(ranges):
    records = [struct.pack("<4I2Q16s", layer, expert, kind, 0, offset, size, bytes(16))
               for layer, expert, kind, offset, size in ranges]
    return struct.pack("<4I", 0x58504557, 2, len(records), 0) + b"".join(records)


def random_layout(rng):
    cursor = rng.randrange(0, 4096)
    ranges = []
    for layer in range(rng.randrange(1, 6)):
        cursor += rng.randrange(0, 3000)
        for expert in range(rng.randrange(1, 9)):
            for kind in range(rng.randrange(1, 3)):
                size = rng.randrange(1, 5000)
                ranges.append((layer, expert, kind, cursor, size))
                cursor += size + rng.choice((0, 0, rng.randrange(1, 700)))
    pack_bytes = cursor + rng.choice((0, rng.randrange(1, 900)))
    rng.shuffle(ranges)
    return ranges, pack_bytes


def main():
    tool = load_tool()
    rng = random.Random(20260928)
    with tempfile.TemporaryDirectory(prefix="weightd-spine-budget-") as directory:
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
        pack = root / "rank.pack"
        for case in range(200):
            ranges, pack_bytes = random_layout(rng)
            with open(pack, "wb") as handle:
                handle.truncate(pack_bytes)
            (root / "rank.pack.experts").write_bytes(manifest_bytes(ranges))
            manifest = Manifest()
            status = lib.SparkWeightdManifestLoad(str(root / "rank.pack.experts").encode(),
                                                  pack_bytes, C.byref(manifest))
            assert status == 0, (case, status)
            record = tool.budget_for(str(pack))
            assert record["spine_bytes"] == manifest.spine_bytes, (case, record)
            assert record["spine_allocation_bytes"] == manifest.spine_allocation_bytes, (case, record)
            assert record["spine_budget_bytes"] == manifest.spine_allocation_bytes + 255
            lib.SparkWeightdManifestDestroy(C.byref(manifest))
        overlapping = manifest_bytes([(0, 0, 0, 100, 50), (0, 1, 0, 120, 10)])
        try:
            tool.spine_allocation(overlapping, 1000)
        except ValueError:
            pass
        else:
            raise AssertionError("overlapping ranges must be rejected")
        try:
            tool.spine_allocation(manifest_bytes([(0, 0, 0, 900, 200)]), 1000)
        except ValueError:
            pass
        else:
            raise AssertionError("a range past the pack end must be rejected")
        output = subprocess.run(["python3", str(ROOT / "tools/weightd_spine_budget.py"), str(pack)],
                                check=True, capture_output=True, text=True).stdout.strip()
        assert output == str(tool.budget_for(str(pack))["spine_budget_bytes"])
    print("weightd spine budget: 200 layouts match the runtime allocation")


if __name__ == "__main__":
    main()
