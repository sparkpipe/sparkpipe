#!/usr/bin/env python3
"""Host checks for the qwen38_27b expert-manifest producer and the
smoke-experts generator: synthetic packs, real binary, fail-closed paths.

The producer compiles from the tree (cc, no GPU) against the module's own
stagepack wire structs, so a format drift breaks the build, not the fleet.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

spec = importlib.util.spec_from_file_location(
    "qwen38_27b_smoke_experts", ROOT / "tools" / "qwen38_27b_smoke_experts.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)

MAGIC = 0x50533651
GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE
KIND_EMBEDDING, KIND_FFN_GATE, KIND_FFN_UP, KIND_FFN_DOWN = 0, 5, 6, 7
WEIGHT_BF16, WEIGHT_NVFP4 = 0, 8
MANIFEST_MAGIC = 0x58504557


def synthetic_pack(path: Path, layers: int, tp_degree: int = 4,
                   weight_format: int = WEIGHT_NVFP4, with_scale: bool = True,
                   skip_up_layer: int | None = None, mtp_ffn: bool = True) -> bytes:
    """Write a minimal qwen38_27b stagepack: header(120) + directory + padded
    payloads. Per-layer FFN gate/up/down (payload + optional scale plane),
    one global embedding tensor, optional MTP-layer FFN."""
    entries = []
    payloads = bytearray()

    def add_entry(kind, layer, fmt, payload_bytes, scale_bytes):
        offset = 120
        # directory sits between header and payloads; its size is only known
        # after collecting entries, so record offsets against a fixed prefix
        # and patch once the directory size is final.
        entries.append([kind, layer, fmt, payload_bytes, scale_bytes,
                        offset, payload_bytes])
        payloads.extend(bytes([(layer + kind) % 251] * payload_bytes))
        if scale_bytes:
            payloads.extend(bytes([9] * scale_bytes))

    payload = 8192 if weight_format == WEIGHT_NVFP4 else 16384
    scale = 512 if (with_scale and weight_format == WEIGHT_NVFP4) else 0
    add_entry(KIND_EMBEDDING, GLOBAL_LAYER, WEIGHT_BF16, 4096, 0)
    for layer in range(layers):
        add_entry(KIND_FFN_GATE, layer, weight_format, payload, scale)
        if layer != skip_up_layer:
            add_entry(KIND_FFN_UP, layer, weight_format, payload, scale)
        add_entry(KIND_FFN_DOWN, layer, weight_format, payload, scale)
    if mtp_ffn:
        add_entry(KIND_FFN_GATE, MTP_LAYER, WEIGHT_BF16, 4096, 0)
        add_entry(KIND_FFN_UP, MTP_LAYER, WEIGHT_BF16, 4096, 0)
        add_entry(KIND_FFN_DOWN, MTP_LAYER, WEIGHT_BF16, 4096, 0)

    directory_bytes = 56 * len(entries)
    directory = bytearray()
    for kind, layer, fmt, payload_bytes, scale_bytes, _fixed, _ in []:
        pass
    for index, (kind, layer, fmt, payload_bytes, scale_bytes, _f, _p) in enumerate(entries):
        # walk payloads again to compute true offsets (payloads were appended
        # in the same order as entries)
        pass
    # compute true payload offsets by replaying sizes in entry order
    offset = 120 + directory_bytes
    directory = bytearray()
    for kind, layer, fmt, payload_bytes, scale_bytes, _f, _p in entries:
        scale_offset = offset + payload_bytes if scale_bytes else 0
        directory += struct.pack("<6I", kind, layer, fmt, 4096, 512, 32)
        directory += struct.pack("<4Q", offset, payload_bytes, scale_offset, scale_bytes)
        offset += payload_bytes + scale_bytes
    file_bytes = offset
    header = struct.pack("<26I", MAGIC, 3, 120, 56, len(entries), 5120, layers, 0,
                         64, 4, 3, 16, 48, 128, 128, 4, 24, 4, 256, 64, 17408,
                         248320, 32, 1, tp_degree, 0)
    header += struct.pack("<2Q", 120, file_bytes)
    assert len(header) == 120
    blob = header + bytes(directory) + bytes(payloads)
    path.write_bytes(blob)
    return blob


def compile_producer(build: Path) -> Path:
    binary = build / "qwen38_27b_experts_manifest_test"
    subprocess.run([
        "cc", "-std=c11", "-O1",
        f"-I{ROOT / 'include'}",
        f"-I{ROOT / 'model-families' / 'qwen38_27b' / 'include'}",
        f"-I{ROOT / 'modules' / 'qwen38_27b_resident_decode_stage' / 'include'}",
        str(ROOT / "tools" / "qwen38_27b_experts_manifest.c"),
        str(ROOT / "runtime" / "spark_weightd_manifest.c"),
        str(ROOT / "src" / "spark_ck128.c"),
        "-o", str(binary),
    ], check=True, timeout=120)
    return binary


def parse_manifest(path: Path) -> dict:
    raw = path.read_bytes()
    magic, version, count, zero = struct.unpack_from("<4I", raw, 0)
    records = []
    for index in range(count):
        base = 16 + index * 48
        layer, expert, kind, reserved = struct.unpack_from("<4I", raw, base)
        offset, nbytes = struct.unpack_from("<2Q", raw, base + 16)
        records.append({"layer": layer, "expert": expert, "kind": kind,
                        "reserved": reserved, "offset": offset, "bytes": nbytes})
    return {"magic": magic, "version": version, "count": count, "zero": zero,
            "records": records, "trailing": len(raw) - (16 + count * 48)}


class ProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="q38-manifest-")
        cls.build = Path(cls.tmp.name)
        cls.binary = compile_producer(cls.build)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_producer(self, pack: Path):
        return subprocess.run([str(self.binary), str(pack)],
                              capture_output=True, text=True, timeout=60)

    def test_nvfp4_pack_six_planes_per_layer(self):
        pack = self.build / "nvfp4.q38sp"
        synthetic_pack(pack, layers=3, weight_format=WEIGHT_NVFP4, with_scale=True)
        result = self.run_producer(pack)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = parse_manifest(Path(str(pack) + ".experts"))
        self.assertEqual(manifest["magic"], MANIFEST_MAGIC)
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["count"], 18)
        self.assertEqual(manifest["trailing"], 0)
        kinds = [record["kind"] for record in manifest["records"]]
        self.assertEqual(sorted(kinds), sorted([10, 11, 12, 13, 14, 15] * 3))
        for record in manifest["records"]:
            self.assertEqual(record["expert"], 0)
            self.assertLess(record["layer"], 3)
            self.assertGreater(record["bytes"], 0)
            self.assertEqual(record["reserved"], 0)

    def test_bf16_pack_three_planes_per_layer(self):
        pack = self.build / "bf16.q38sp"
        synthetic_pack(pack, layers=2, weight_format=WEIGHT_BF16, with_scale=False)
        result = self.run_producer(pack)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = parse_manifest(Path(str(pack) + ".experts"))
        self.assertEqual(manifest["count"], 6)
        kinds = sorted(record["kind"] for record in manifest["records"])
        self.assertEqual(kinds, [10, 10, 12, 12, 14, 14])

    def test_mtp_ffn_stays_in_spine(self):
        pack = self.build / "mtp.q38sp"
        synthetic_pack(pack, layers=2, mtp_ffn=True)
        result = self.run_producer(pack)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = parse_manifest(Path(str(pack) + ".experts"))
        self.assertTrue(all(record["layer"] < 2 for record in manifest["records"]))
        self.assertEqual(manifest["count"], 12)

    def test_incomplete_layer_fails_closed(self):
        pack = self.build / "incomplete.q38sp"
        synthetic_pack(pack, layers=2, skip_up_layer=1)
        result = self.run_producer(pack)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(Path(str(pack) + ".experts").exists())

    def test_stale_output_preserved(self):
        pack = self.build / "stale.q38sp"
        synthetic_pack(pack, layers=1)
        self.assertEqual(self.run_producer(pack).returncode, 0)
        stale = Path(str(pack) + ".experts")
        stale.write_bytes(stale.read_bytes())
        before = stale.read_bytes()
        result = self.run_producer(pack)
        self.assertEqual(result.returncode, 1)
        self.assertIn("stale", result.stderr)
        self.assertEqual(stale.read_bytes(), before)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="q38-smoke-")
        self.build = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def generate(self, pack: Path, *extra):
        output = self.build / "smoke_experts.json"
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "qwen38_27b_smoke_experts.py"),
             "--pack", str(pack), "--output", str(output), *extra],
            capture_output=True, text=True, timeout=60)
        return result, output

    def test_manifest_fields_and_scaling(self):
        pack = self.build / "rank.q38sp"
        synthetic_pack(pack, layers=4, tp_degree=4, weight_format=WEIGHT_NVFP4)
        result, output = self.generate(pack, "--allow-prefix", "--kv-floor-tokens", "512")
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(output.read_text())
        self.assertEqual(manifest["family"], "qwen38_27b")
        self.assertEqual(manifest["topology"], "TP4")
        self.assertEqual(manifest["nodes"], 4)
        self.assertEqual(manifest["expert_shard"], "tp")
        self.assertEqual(len(manifest["experts"]), 4)
        self.assertEqual([expert["layer"] for expert in manifest["experts"]], [0, 1, 2, 3])
        self.assertTrue(all(expert["codec"] == "nvfp4a16" for expert in manifest["experts"]))
        per_rank = sum(expert["bytes"] for expert in manifest["experts"]) // 4
        self.assertEqual(per_rank * 4, manifest["provenance"]["expert_bytes_full_model"])
        # spine = pack bytes minus per-rank experts, scaled back to full model
        self.assertEqual(
            manifest["spine_bytes"],
            (pack.stat().st_size - per_rank) * 4)
        # 16 attn layers x 512 tokens x 1 kv head x 256 x 2(K+V) x 2B
        # + 48 GDN layers x 16 seq x ((4x128 + 12x128) x 2B)
        expected_kv = 16 * 512 * 1 * 256 * 2 * 2 + 48 * 16 * ((4 * 128 + 12 * 128) * 2)
        self.assertEqual(manifest["kv_floor_bytes"], expected_kv)
        self.assertEqual(manifest["workspace_bytes"], 1024 * 1024 * 1024)

    def test_size_mismatch_requires_prefix_flag(self):
        pack = self.build / "rank.q38sp"
        blob = synthetic_pack(pack, layers=2)
        truncated = self.build / "prefix.q38sp"
        truncated.write_bytes(blob[:len(blob) - 1])
        result, _ = self.generate(truncated)
        self.assertEqual(result.returncode, 1)
        self.assertIn("allow-prefix", result.stderr)
        result, _ = self.generate(truncated, "--allow-prefix")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixed_formats_refused(self):
        pack = self.build / "mixed.q38sp"
        synthetic_pack(pack, layers=2, weight_format=WEIGHT_NVFP4)
        blob = bytearray(pack.read_bytes())
        # flip the FFN_UP weight format of layer 1 (second UP entry, index 4:
        # embedding + g0 + u0 + d0 + g1 -> entry 4 is layer-1 gate)
        base = 120 + 4 * 56 + 8
        struct.pack_into("<I", blob, base, WEIGHT_BF16)
        pack.write_bytes(bytes(blob))
        result, _ = self.generate(pack)
        self.assertEqual(result.returncode, 1)
        self.assertIn("mixed FFN", result.stderr)


if __name__ == "__main__":
    unittest.main()
