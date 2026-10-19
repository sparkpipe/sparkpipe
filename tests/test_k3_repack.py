"""Convert a small format 2 cell-split K3 rank pack to format 3 and check the contract: this rank's router and MLA down rows, paired 4-bit scale cells that decode to the source experts bit for bit, copied tensors, a .sha256 of the bytes on disk, a weightd .experts sidecar, and loud refusals."""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "k3_repack.py"

if not hasattr(os, "preadv") or not hasattr(os, "posix_fadvise") or shutil.which("cc") is None:
    print("SKIP test_k3_repack (needs Linux preadv/posix_fadvise and cc)")
    raise SystemExit(0)

import numpy as np  # noqa: E402

HIDDEN, EXPERTS, DEGREE, RANK = 64, 8, 4, 2
Q_LORA, KV_A = 32, 16
W1 = (EXPERTS, 32, 128)
W2 = (EXPERTS, 64, 64)


def build_library(directory):
    library = Path(directory) / "libk3manifest.so"
    subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC", f"-I{ROOT}", f"-I{ROOT}/include",
                    str(ROOT / "src/spark_ck128.c"), str(ROOT / "src/spark_status.c"),
                    str(ROOT / "runtime/spark_weightd_manifest.c"), "-o", str(library)], check=True)
    return library


def interleave(shape, low, high, rng):
    experts, out_dim, k_dim = shape
    tiles, cells = k_dim // 32, out_dim // 16
    payload = rng.integers(0, 256, size=(experts, tiles, cells, 16, 16), dtype=np.uint8)
    scales = rng.integers(low, high + 1, size=(experts, tiles, cells, 16), dtype=np.uint8)
    grid = np.empty((experts, tiles, cells, 17, 16), dtype=np.uint8)
    grid[:, :, :, :16, :] = payload
    grid[:, :, :, 16, :] = scales
    geom = {"kind": "mxfp4_ws_interleaved_v1", "out_dim": out_dim, "k_dim": k_dim, "experts": experts, "tile_k": 32, "group": 32,
            "stored_bits": 4, "row_bytes": 16, "scale_bytes_per_neuron_tile": 1, "cell_payload_rows": 16, "cell_rows": 17,
            "cells": cells, "k_tiles": tiles, "rows_per_expert": tiles * cells * 17, "expert_bytes": tiles * cells * 17 * 16,
            "tensor_bytes": experts * tiles * cells * 17 * 16}
    return grid.tobytes(), geom


def write_pack(path, w1_scales=(110, 123), expert_split="cells"):
    rng = np.random.default_rng(7)
    tensors, blobs = {}, []
    offset = 0

    def add(name, data, kind, shape, extra=None):
        nonlocal offset
        offset += (-offset) % 128
        entry = {"offset": offset, "bytes": len(data), "align": 128, "kind": kind, "shape": list(shape)}
        entry.update(extra or {})
        tensors[name] = entry
        blobs.append((offset, data))
        offset += len(data)

    add("model.embed_tokens.weight", rng.integers(0, 256, size=40 * HIDDEN * 2, dtype=np.uint8).tobytes(), "bf16", (40, HIDDEN))
    add("model.layers.0.mla_q_down_weight", rng.integers(0, 256, size=Q_LORA * HIDDEN * 2, dtype=np.uint8).tobytes(), "bf16", (Q_LORA, HIDDEN))
    add("model.layers.0.mla_kv_a_weight", rng.integers(0, 256, size=KV_A * HIDDEN * 2, dtype=np.uint8).tobytes(), "bf16", (KV_A, HIDDEN))
    add("model.layers.1.router_weight", rng.integers(0, 256, size=EXPERTS * HIDDEN * 2, dtype=np.uint8).tobytes(), "bf16", (EXPERTS, HIDDEN))
    add("model.layers.1.router_bias", rng.integers(0, 256, size=EXPERTS * 4, dtype=np.uint8).tobytes(), "f32", (EXPERTS,))
    w1, g1 = interleave(W1, w1_scales[0], w1_scales[1], rng)
    add("model.layers.1.expert_w1_weight", w1, "mxfp4_ws_interleaved_v1", W1, {"interleave": g1, "shard_class": "expert_cells"})
    w2, g2 = interleave(W2, 119, 124, rng)
    add("model.layers.1.expert_w2_weight", w2, "mxfp4_ws_interleaved_v1", W2, {"interleave": g2, "shard_class": "expert_k"})
    add("model.norm.weight", rng.integers(0, 256, size=HIDDEN * 2, dtype=np.uint8).tobytes(), "bf16", (HIDDEN,))
    manifest = {"format": {"version": 2, "alignment": 128, "mxfp4_interleave": {"tile_k": 128}},
                "config": {"hidden": HIDDEN, "experts": EXPERTS, "tp_degree": DEGREE, "tp_rank": RANK, "expert_split": expert_split},
                "tensors": tensors}
    reserve = 4096 - 16
    text = json.dumps(manifest).encode()
    payload_base = 4096
    with open(path, "wb") as out:
        out.write(struct.pack("<IIQ", 0x4B33504B, 2, reserve))
        out.write(text + b" " * (reserve - len(text)))
        for start, data in blobs:
            out.seek(payload_base + start)
            out.write(data)
        out.truncate(payload_base + offset)
    return manifest, payload_base


def read_pack(path):
    raw = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<IIQ", raw, 0)
    manifest = json.loads(raw[16:16 + length])
    base = 16 + length
    base += (-base) % 128
    return version, manifest, base, raw


class Repack(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="k3-repack-")
        self.library = build_library(self.directory)
        self.source = Path(self.directory) / "rank.pack"

    def tearDown(self):
        shutil.rmtree(self.directory)

    def run_tool(self, *args):
        return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True)

    def test_convert_keeps_every_value(self):
        manifest, base = write_pack(self.source)
        target = Path(self.directory) / "rank.v3.pack"
        result = self.run_tool("convert", self.source, target, self.library)
        self.assertEqual(result.returncode, 0, result.stderr)
        version, out, out_base, raw = read_pack(target)
        _, _, _, source_raw = read_pack(self.source)
        self.assertEqual(version, 3)
        self.assertEqual(out_base, base)
        self.assertEqual(out["format"]["expert_codec"], "mxfp4_ws_paired_v1")
        self.assertNotIn("mxfp4_interleave", out["format"])
        self.assertEqual(out["config"]["weights_identity"], hashlib.sha256(source_raw).hexdigest())
        for name, total in (("model.layers.1.router_weight", EXPERTS), ("model.layers.0.mla_q_down_weight", Q_LORA), ("model.layers.0.mla_kv_a_weight", KV_A)):
            entry, source_entry = out["tensors"][name], manifest["tensors"][name]
            count = total // DEGREE
            self.assertEqual(entry["rows"], {"first": RANK * count, "count": count, "total": total})
            self.assertEqual(entry["shape"], [count, HIDDEN])
            row = HIDDEN * 2
            got = raw[out_base + entry["offset"]:out_base + entry["offset"] + entry["bytes"]]
            want = source_raw[base + source_entry["offset"] + RANK * count * row:base + source_entry["offset"] + (RANK + 1) * count * row]
            self.assertEqual(got, want, name)
        for name in ("model.embed_tokens.weight", "model.layers.1.router_bias", "model.norm.weight"):
            entry, source_entry = out["tensors"][name], manifest["tensors"][name]
            self.assertEqual(raw[out_base + entry["offset"]:out_base + entry["offset"] + entry["bytes"]],
                             source_raw[base + source_entry["offset"]:base + source_entry["offset"] + source_entry["bytes"]], name)
        for name, low in (("model.layers.1.expert_w1_weight", 110), ("model.layers.1.expert_w2_weight", 119)):
            entry, source_entry = out["tensors"][name], manifest["tensors"][name]
            geom, source_geom = entry["interleave"], source_entry["interleave"]
            self.assertEqual(geom["scale_base"], low)
            self.assertEqual(geom["cell_rows"], 33)
            self.assertEqual(entry["bytes"], geom["k_pairs"] * geom["cells"] * 528 * geom["experts"])
            pairs = np.frombuffer(raw, dtype=np.uint8, count=entry["bytes"], offset=out_base + entry["offset"]).reshape(
                geom["experts"], geom["k_pairs"], geom["cells"], 33, 16)
            tiles = np.frombuffer(source_raw, dtype=np.uint8, count=source_entry["bytes"], offset=base + source_entry["offset"]).reshape(
                source_geom["experts"], source_geom["k_tiles"], source_geom["cells"], 17, 16)
            for t in range(source_geom["k_tiles"]):
                p, position = divmod(t, 2)
                self.assertTrue(np.array_equal(pairs[:, p, :, position * 16:(position + 1) * 16, :], tiles[:, t, :, :16, :]), (name, t))
                codes = (pairs[:, p, :, 32, :] >> (4 * position)) & 15
                self.assertTrue(np.array_equal(codes.astype(np.int32) + low, tiles[:, t, :, 16, :].astype(np.int32)), (name, t))
        self.assertEqual((Path(str(target) + ".sha256")).read_text(), f"{hashlib.sha256(raw).hexdigest()}  {target.name}\n")
        sidecar = Path(str(target) + ".experts").read_bytes()
        magic, version, records, _ = struct.unpack_from("<IIII", sidecar, 0)
        self.assertEqual((magic, version, records), (0x58504557, 2, 2 * EXPERTS))
        self.assertFalse(any(name.endswith(".partial") for name in os.listdir(self.directory)))
        again = self.run_tool("verify", self.source, target, self.library)
        self.assertEqual(again.returncode, 0, again.stderr)

    def test_scale_span_over_sixteen_is_refused(self):
        write_pack(self.source, w1_scales=(100, 123))
        target = Path(self.directory) / "rank.v3.pack"
        result = self.run_tool("convert", self.source, target, self.library)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("do not fit one 16-code window", result.stderr)
        self.assertFalse(target.exists())
        self.assertFalse(any(name.startswith("rank.v3.pack") for name in os.listdir(self.directory)))

    def test_non_cell_pack_is_refused(self):
        write_pack(self.source, expert_split="k")
        result = self.run_tool("convert", self.source, Path(self.directory) / "out.pack", self.library)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not cells", result.stderr)

    def test_existing_output_is_refused(self):
        write_pack(self.source)
        target = Path(self.directory) / "rank.v3.pack"
        target.write_bytes(b"keep")
        result = self.run_tool("convert", self.source, target, self.library)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(target.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
