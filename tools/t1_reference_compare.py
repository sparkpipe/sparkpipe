import argparse
import hashlib
import json
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import T1R_MAGIC, read_fixture, sha256_file

DEFAULT_REL = 0.02
DEFAULT_ABS = 1e-3


def compare_array(name, want, got, rel, absolute, meta_dtype=None):
    if want.dtype.kind in ("i", "u") and meta_dtype not in ("BF16", "F16"):
        if want.shape != got.shape or not np.array_equal(want, got):
            return {"name": name, "kind": "exact",
                    "detail": f"shape {want.shape} vs {got.shape}"}
        return None
    if want.shape != got.shape:
        return {"name": name, "kind": "shape",
                "detail": f"{want.shape} vs {got.shape}"}
    a = want.astype(np.float32)
    b = got.astype(np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"name": name, "kind": "nonfinite", "detail": "nonfinite values"}
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    rel_err = np.where(denom > 0, diff / denom, diff)
    bad = (rel_err > rel) & (diff > absolute)
    if bad.any():
        index = int(np.argmax(bad))
        return {"name": name, "kind": "tolerance", "index": int(index),
                "detail": f"want {a.flat[index]!r} got {b.flat[index]!r} "
                          f"rel {float(rel_err.flat[index]):.6g}"}
    return None


def compare_fixtures(reference_path, candidate_path, rel, absolute):
    ref_meta, ref = read_fixture(reference_path)
    cand_meta, cand = read_fixture(candidate_path)
    kinds = {e["name"]: e["dtype"] for e in ref_meta["arrays"]}
    failures = []
    missing = sorted(set(ref) - set(cand))
    extra = sorted(set(cand) - set(ref))
    for name in missing:
        failures.append({"name": name, "kind": "missing",
                         "detail": "absent from candidate"})
    for name in sorted(set(ref) & set(cand)):
        failure = compare_array(name, ref[name], cand[name], rel, absolute,
                                kinds.get(name))
        if failure is not None:
            failures.append(failure)
    return failures, extra


def corrupt_fixture(source_path, target_path, array_name, offset=0):
    with open(source_path, "rb") as fh:
        data = bytearray(fh.read())
    if data[:4] != T1R_MAGIC:
        raise ValueError(f"{source_path}: not a T1R1 fixture")
    off = 4
    n = struct.unpack("<Q", data[off:off + 8])[0]
    meta_off = off + 8
    meta = json.loads(bytes(data[meta_off:meta_off + n]))
    payload_off = meta_off + n
    cursor = payload_off
    target = next(e for e in meta["arrays"] if e["name"] == array_name)
    for entry in meta["arrays"]:
        size = struct.unpack("<Q", data[cursor:cursor + 8])[0]
        if entry["name"] == array_name:
            raw = bytearray(zlib.decompress(bytes(data[cursor + 8:
                                                    cursor + 8 + size])))
            index = offset % len(raw)
            raw[index] ^= 0xFF
            comp = zlib.compress(bytes(raw), 6)
            entry["compressed"] = len(comp)
            entry["sha256"] = hashlib.sha256(bytes(raw)).hexdigest()
            data[cursor:cursor + 8 + size] = struct.pack("<Q", len(comp)) + comp
            break
        cursor += 8 + size
    blob = json.dumps(meta, sort_keys=True).encode("utf-8")
    with open(target_path, "wb") as fh:
        fh.write(T1R_MAGIC)
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(bytes(data[payload_off:]))
    return {"array": array_name, "byte_index_flipped": index,
            "target": target_path}


def verify_manifest(fixture_dir):
    manifest = json.load(open(os.path.join(fixture_dir, "MANIFEST.json")))
    failures = []
    for name, record in sorted(manifest["fixtures"].items()):
        path = os.path.join(fixture_dir, name)
        if not os.path.exists(path):
            failures.append(f"{name}: fixture absent")
            continue
        digest = sha256_file(path)
        if digest != record["sha256"]:
            failures.append(f"{name}: sha256 {digest} != manifest {record['sha256']}")
        if os.path.getsize(path) != record["bytes"]:
            failures.append(f"{name}: byte size mismatch")
    return failures


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--rel", type=float, default=DEFAULT_REL)
    compare.add_argument("--abs", type=float, default=DEFAULT_ABS)
    corrupt = sub.add_parser("corrupt-fixture")
    corrupt.add_argument("--source", required=True)
    corrupt.add_argument("--target", required=True)
    corrupt.add_argument("--array", required=True)
    corrupt.add_argument("--offset", type=int, default=0)
    verify = sub.add_parser("verify-manifest")
    verify.add_argument("--fixture-dir", required=True)
    arguments = parser.parse_args()
    if arguments.command == "compare":
        failures, extra = compare_fixtures(arguments.reference,
                                           arguments.candidate,
                                           arguments.rel, arguments.abs)
        for failure in failures:
            print(json.dumps(failure, sort_keys=True))
        for name in extra:
            print(json.dumps({"name": name, "kind": "extra",
                              "detail": "absent from reference"}))
        if failures:
            print(f"FIRST DIVERGENCE: {failures[0]['name']} ({failures[0]['kind']})")
            print(f"RESULT: FAIL ({len(failures)} arrays, {len(extra)} extra)")
            return 1
        print(f"RESULT: PASS ({len(extra)} extra arrays)")
        return 0
    if arguments.command == "corrupt-fixture":
        print(json.dumps(corrupt_fixture(arguments.source, arguments.target,
                                         arguments.array, arguments.offset),
                         sort_keys=True))
        return 0
    failures = verify_manifest(arguments.fixture_dir)
    for failure in failures:
        print(failure)
    if failures:
        print(f"RESULT: FAIL ({len(failures)})")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
