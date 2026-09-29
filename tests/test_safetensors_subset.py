import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import safetensors_subset as subset


def write_shard(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        raw = array.tobytes()
        header[name] = {"dtype": {np.dtype(np.uint16): "BF16", np.dtype(np.uint8): "F8_E4M3",
                                  np.dtype(np.float32): "F32"}[array.dtype],
                        "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    header["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


class SafetensorsSubsetTest(unittest.TestCase):
    def test_subset_copies_selected_tensors_byte_for_byte(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as root:
            source, out = os.path.join(root, "ck"), os.path.join(root, "out")
            os.makedirs(source)
            first = {"layers.45.a.weight": rng.integers(0, 255, (4, 6), dtype=np.uint8),
                     "layers.44.a.weight": rng.integers(0, 60000, (3,), dtype=np.uint16)}
            second = {"layers.45.b.weight_scale_inv": rng.standard_normal((2, 2)).astype(np.float32),
                      "lm_head.weight": rng.integers(0, 60000, (5, 2), dtype=np.uint16)}
            write_shard(os.path.join(source, "s1.safetensors"), first)
            write_shard(os.path.join(source, "s2.safetensors"), second)
            weight_map = {n: "s1.safetensors" for n in first}
            weight_map.update({n: "s2.safetensors" for n in second})
            json.dump({"weight_map": weight_map}, open(os.path.join(source, "model.safetensors.index.json"), "w"))
            json.dump({"hidden_size": 4}, open(os.path.join(source, "config.json"), "w"))
            self.assertEqual(subset.main(["--checkpoint", source, "--out", out,
                                          "--include", r"layers\.45\..*", "--include", "lm_head.weight"]), 0)
            with open(os.path.join(out, "model.safetensors"), "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(size))
                base = 8 + size
                for name, array in {**first, **second}.items():
                    if name.startswith("layers.44"):
                        self.assertNotIn(name, header)
                        continue
                    start, end = header[name]["data_offsets"]
                    fh.seek(base + start)
                    self.assertEqual(fh.read(end - start), array.tobytes())
                    self.assertEqual(header[name]["shape"], list(array.shape))
            manifest = json.load(open(os.path.join(out, "SUBSET.json")))
            self.assertEqual(manifest["tensor_count"], 3)
            self.assertTrue(os.path.exists(os.path.join(out, "config.json")))

    def test_pattern_matching_nothing_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            write_shard(os.path.join(root, "model.safetensors"), {"x": np.zeros(2, dtype=np.float32)})
            with self.assertRaises(SystemExit):
                subset.main(["--checkpoint", root, "--out", os.path.join(root, "o"), "--include", "missing"])


if __name__ == "__main__":
    unittest.main()
