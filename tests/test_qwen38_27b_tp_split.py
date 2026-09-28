#!/usr/bin/env python3
"""Host checks for the qwen38_27b TP rank-pack splitter and TP4 deployment writer.

A synthetic TP1 pack with a replicated norm, a row-split e8m0 linear, a
fused GDN QKV and a column-split e8m0 linear is split to TP4: every rank keeps
the header geometry with its tp_degree/tp_rank, the shards concatenate back to
the source rows and columns (scale planes included), --verify accepts the
result and rejects a corrupted shard. The deployment writer emits one fanout
node per host and per-rank adapter configs with the adapter's member set.
"""
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPOSITORY_ROOT / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = load("qwen38_27b_tp_split_pack")
packer = split.load_packer()

HIDDEN = packer.HIDDEN
ENTRIES = [
    (packer.KIND_ATTENTION_NORM, 3, split.FMT_BF16, 1, HIDDEN),
    (packer.KIND_ATTN_KEY, 3, split.FMT_E8M0B128, packer.ATTN_KV_DIM, HIDDEN),
    (packer.KIND_GDN_QKV, 0, split.FMT_E8M0B128, 2 * packer.GDN_QK_DIM + packer.GDN_VALUE_DIM, HIDDEN),
    (packer.KIND_GDN_OUTPUT, 0, split.FMT_E8M0B128, HIDDEN, packer.GDN_VALUE_DIM),
]


def write_source(path, rng):
    header = [split.MAGIC, split.FORMAT_VERSION, split.HEADER_BYTES, split.ENTRY_BYTES, len(ENTRIES)] + [0] * 19 + [1, 0]
    arrays = []
    cursor = split.HEADER_BYTES + len(ENTRIES) * split.ENTRY_BYTES
    directory = []
    blobs = []
    for kind, layer, fmt, rows, cols in ENTRIES:
        cursor += (-cursor) % split.ALIGNMENT
        payload = rng.integers(0, 256, size=(rows, cols * split.ELEMENT_BYTES[fmt]), dtype=np.uint8)
        scale = rng.integers(100, 140, size=(rows, cols // 128), dtype=np.uint8) if fmt == split.FMT_E8M0B128 else None
        pbytes = payload.size
        sbytes = scale.size if scale is not None else 0
        directory.append((kind, layer, fmt, rows, cols, 128 if scale is not None else 0,
                          cursor, pbytes, cursor + pbytes if sbytes else 0, sbytes))
        blobs.append((cursor, payload.tobytes() + (scale.tobytes() if scale is not None else b"")))
        arrays.append((payload, scale))
        cursor += pbytes + sbytes
    with open(path, "wb") as f:
        f.write(struct.pack("<26I2Q", *header, split.HEADER_BYTES, cursor))
        for entry in directory:
            f.write(struct.pack("<6I4Q", *entry))
        for offset, blob in blobs:
            f.seek(offset)
            f.write(blob)
    return arrays


class TpSplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.source = self.dir / "tp1.qwen36sp"
        self.arrays = write_source(self.source, np.random.default_rng(7))
        self.outputs = [self.dir / "out" / f"t.rank{r}.qwen36sp" for r in range(4)]
        (self.dir / "out").mkdir()
        for rank in range(4):
            split.write_rank(self.source, self.outputs[rank], 4, rank, packer)

    def rank_entry(self, rank, index):
        _, fields, entries = split.read_pack(self.outputs[rank])
        data = np.memmap(self.outputs[rank], dtype=np.uint8, mode="r")
        e = entries[index]
        width = split.ELEMENT_BYTES[e["fmt"]]
        payload = np.array(data[e["poff"]:e["poff"] + e["pbytes"]]).reshape(e["rows"], e["cols"] * width)
        scale = np.array(data[e["soff"]:e["soff"] + e["sbytes"]]).reshape(e["rows"], e["cols"] // 128) if e["sbytes"] else None
        return fields, e, payload, scale

    def test_headers_carry_rank_and_file_size(self):
        for rank in range(4):
            fields, _, _, _ = self.rank_entry(rank, 0)
            self.assertEqual((fields[24], fields[25]), (4, rank))
            self.assertEqual(fields[27], self.outputs[rank].stat().st_size)

    def test_replicated_norm_is_whole(self):
        for rank in range(4):
            _, e, payload, scale = self.rank_entry(rank, 0)
            self.assertEqual((e["rows"], e["cols"]), (1, HIDDEN))
            self.assertTrue((payload == self.arrays[0][0]).all())
            self.assertIsNone(scale)

    def test_row_split_concatenates(self):
        payloads, scales = zip(*[self.rank_entry(r, 1)[2:] for r in range(4)])
        self.assertTrue((np.concatenate(payloads) == self.arrays[1][0]).all())
        self.assertTrue((np.concatenate(scales) == self.arrays[1][1]).all())

    def test_fused_qkv_takes_q_k_v_windows(self):
        qk, v = packer.GDN_QK_DIM, packer.GDN_VALUE_DIM
        payload, scale = self.arrays[2]
        for rank in range(4):
            _, e, got, got_scale = self.rank_entry(rank, 2)
            rows = np.r_[rank * qk // 4:(rank + 1) * qk // 4,
                         qk + rank * qk // 4:qk + (rank + 1) * qk // 4,
                         2 * qk + rank * v // 4:2 * qk + (rank + 1) * v // 4]
            self.assertEqual(e["rows"], (2 * qk + v) // 4)
            self.assertTrue((got == payload[rows]).all())
            self.assertTrue((got_scale == scale[rows]).all())

    def test_column_split_slices_scale_groups(self):
        payloads, scales = zip(*[self.rank_entry(r, 3)[2:] for r in range(4)])
        self.assertTrue((np.concatenate(payloads, axis=1) == self.arrays[3][0]).all())
        self.assertTrue((np.concatenate(scales, axis=1) == self.arrays[3][1]).all())
        self.assertEqual(self.rank_entry(0, 3)[1]["cols"], packer.GDN_VALUE_DIM // 4)

    def test_verify_accepts_and_rejects_corruption(self):
        result = split.verify(self.source, self.outputs, 4, packer)
        self.assertEqual((result["sharded"], result["replicated"]), (3, 1))
        _, e, _, _ = self.rank_entry(2, 3)
        with open(self.outputs[2], "r+b") as f:
            f.seek(e["poff"] + 5)
            byte = f.read(1)
            f.seek(e["poff"] + 5)
            f.write(bytes([byte[0] ^ 0xFF]))
        with self.assertRaises(SystemExit):
            split.verify(self.source, self.outputs, 4, packer)

    def test_refuses_a_tp_source(self):
        with self.assertRaises(SystemExit):
            split.write_rank(self.outputs[0], self.dir / "again.qwen36sp", 4, 0, packer)


class Tp4DeploymentTest(unittest.TestCase):
    def test_fanout_nodes_and_rank_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = Path(tmp) / "tokenizer.json"
            tok.write_text("{}")
            out = Path(tmp) / "cfg"
            subprocess.run([sys.executable, str(TOOLS / "qwen38_27b_tp4_deployment.py"),
                            "--hosts", "a,b,c,d", "--runtime-root", "/home/{host}/root", "--pack-name", "p",
                            "--weightd-socket", "/tmp/w.sock", "--control-port-base", "23048",
                            "--collective-port-base", "53048", "--transport-port-base", "64048",
                            "--attempt", "0a0b0c0d0e0f1011", "--sequences", "8", "--max-positions", "8192",
                            "--kv-logical-pages", "256", "--kv-physical-pages", "64", "--draft-count", "8",
                            "--eos-token-ids", "248046,248044", "--tokenizer", str(tok),
                            "--tokenizer-vocabulary-size", "248077", "--output", str(out)], check=True,
                           capture_output=True)
            deployment = json.loads((out / "model_resident.dflash2.json").read_text())
            nodes = deployment["nodes"]
            self.assertEqual([n["rank_index"] for n in nodes], [0, 1, 2, 3])
            self.assertEqual([n["stage_index"] for n in nodes], [0, 1, 2, 3])
            self.assertEqual([n["runtime_root"] for n in nodes], [f"/home/{h}/root" for h in "abcd"])
            self.assertEqual([n["control_endpoint"]["port"] for n in nodes], [23048, 23049, 23050, 23051])
            self.assertTrue(all(n["adapter_configuration_path"] == "config/stage.dflash2.json" for n in nodes))
            self.assertEqual(deployment["eos_token_ids"], [248046, 248044])
            for rank in range(4):
                stage = json.loads((out / f"stage.rank{rank}.json").read_text())
                draft = json.loads((out / f"stage.dflash2.rank{rank}.json").read_text())
                self.assertEqual((stage["tp_degree"], stage["tp_rank"]), (4, rank))
                self.assertEqual(stage["stage_pack_path"], f"packs/p.rank{rank}.qwen36sp")
                self.assertNotIn("speculative_draft_count", stage)
                self.assertEqual(draft["speculative_draft_count"], 8)
                self.assertEqual(stage["tp_collective"]["listen_port"], 53048 + rank)
            self.assertIn("tokenizer", json.loads((out / "api.model_resident.json").read_text()))


if __name__ == "__main__":
    unittest.main()
