#!/usr/bin/env python3
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import spark_pack_common  # noqa: E402


def write_shard(path, tensors, payload_bytes, declared_header=None):
    header = json.dumps(tensors).encode()
    with path.open("wb") as file:
        file.write(struct.pack("<Q", declared_header if declared_header is not None else len(header)))
        file.write(header)
        file.write(b"\0" * payload_bytes)


def source(root, tensors, payload_bytes, declared_header=None):
    write_shard(root / "model.safetensors", tensors, payload_bytes, declared_header)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}}))
    (root / "config.json").write_text("{}")
    return spark_pack_common.SafetensorsSource(root)


def refuses(call):
    try:
        call()
    except spark_pack_common.PackFailure:
        return True
    return False


def main():
    failures = []
    tensors = {"inside": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]},
               "past_end": {"dtype": "BF16", "shape": [8], "data_offsets": [8, 24]},
               "reversed": {"dtype": "BF16", "shape": [4], "data_offsets": [8, 0]}}
    with tempfile.TemporaryDirectory() as scratch:
        reader = source(pathlib.Path(scratch), tensors, 16)
        shard, meta, offset = reader.resolve("inside")
        if offset != reader.data_start[shard]:
            failures.append("a tensor inside the shard resolves to its payload start")
        if not refuses(lambda: reader.resolve("past_end")):
            failures.append("a tensor whose data_offsets end past the shard is refused")
        if not refuses(lambda: reader.resolve("reversed")):
            failures.append("a tensor whose data_offsets are reversed is refused")
    with tempfile.TemporaryDirectory() as scratch:
        reader = source(pathlib.Path(scratch), {"inside": tensors["inside"]}, 16, declared_header=1 << 40)
        if not refuses(lambda: reader.resolve("inside")):
            failures.append("a shard declaring a header longer than the file is refused")
    if not refuses(lambda: spark_pack_common.tp_window(10, 4, 0)):
        failures.append("a tensor-parallel window that does not divide is refused")
    optimized = subprocess.run([sys.executable, "-O", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import spark_pack_common as c\n"
                                "try:\n    c.tp_window(10, 4, 0)\nexcept c.PackFailure:\n    sys.exit(0)\nsys.exit(1)", str(ROOT / "tools")])
    if optimized.returncode != 0:
        failures.append("the window check holds under python -O, where asserts are stripped")
    for failure in failures:
        print("FAIL", failure)
    if failures:
        raise SystemExit(1)
    print("PASS spark pack common extents: tensor data_offsets and shard headers must lie inside the file; window checks survive python -O")


if __name__ == "__main__":
    main()
