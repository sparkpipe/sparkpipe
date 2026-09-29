#!/usr/bin/env python3
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import pack_spine_sha

MAGIC = pack_spine_sha.LAYOUTS["glm5_next"]["magic"]


def write_pack(path: Path, entries: list, tp_rank: int = 0, magic: int = MAGIC) -> None:
    blobs = bytearray()
    directory = []
    offset = 264
    for kind, layer, codec, scale_encoding, shape, payload, scale in entries:
        payload_offset = offset + len(blobs)
        blobs += payload
        scale_offset = offset + len(blobs) if scale else 0
        blobs += scale
        directory.append(struct.pack("<8I4Q", kind, layer, 4 if codec != 1 else 1, codec, scale_encoding, *shape,
                                     payload_offset, len(payload), scale_offset, len(scale)))
    directory_offset = offset + len(blobs)
    file_bytes = directory_offset + 64 * len(directory)
    words = [magic, 1, 264, 64, 1, 0, len(entries), 1, 0, 0, 46, 46, 4096, 154880, 288, 1, 5, 1, 16, tp_rank]
    header = struct.pack("<20I", *words) + struct.pack("<QQ", directory_offset, file_bytes)
    header += b"\0" * (264 - len(header))
    path.write_bytes(header + bytes(blobs) + b"".join(directory))


def spine():
    return [
        (0, 0xFFFFFFFF, 1, 0, (1, 4, 8), bytes(range(64)), b""),
        (6, 3, 5, 1, (1, 8, 8), bytes(range(64, 128)), b"\x01\x02\x03\x04"),
        (20, 3, 1, 0, (1, 2, 8), bytes(range(128, 160)), b""),
    ]


def experts(fill: int):
    return [(22, 3, 5, 1, (4, 8, 8), bytes([fill]) * 256, bytes([fill + 1]) * 16),
            (23, 3, 5, 1, (4, 8, 8), bytes([fill + 2]) * 256, bytes([fill + 3]) * 16)]


class SpineDigestTest(unittest.TestCase):
    def digest(self, directory, name, entries, **kwargs):
        path = Path(directory) / name
        write_pack(path, entries, **kwargs)
        return pack_spine_sha.spine_digest(path, "glm5_next")

    def test_experts_and_layout_do_not_move_the_spine_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.digest(directory, "a.sp", spine() + experts(1))
            other_experts = self.digest(directory, "b.sp", experts(9) + list(reversed(spine())))
            nvfp4_like = self.digest(directory, "c.sp", spine() + [(22, 3, 6, 3, (4, 8, 4), b"\x07" * 128, b"\x08" * 40)])
            self.assertEqual(base["spine_digest"], other_experts["spine_digest"])
            self.assertEqual(base["spine_digest"], nvfp4_like["spine_digest"])
            self.assertEqual((base["spine_entries"], base["expert_entries"]), (3, 2))

    def test_any_spine_change_moves_the_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self.digest(directory, "a.sp", spine() + experts(1))["spine_digest"]
            changes = []
            byte = spine()
            byte[1] = byte[1][:5] + (bytes([0]) + byte[1][5][1:],) + byte[1][6:]
            changes.append(byte)
            scale = spine()
            scale[1] = scale[1][:6] + (b"\x01\x02\x03\x05",)
            changes.append(scale)
            codec = spine()
            codec[2] = (20, 3, 5, 0) + codec[2][4:]
            changes.append(codec)
            layer = spine()
            layer[2] = (20, 4) + layer[2][2:]
            changes.append(layer)
            for index, entries in enumerate(changes):
                self.assertNotEqual(self.digest(directory, f"c{index}.sp", entries + experts(1))["spine_digest"], base, index)

    def test_refusals_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.sp"
            write_pack(path, spine(), magic=0x12345678)
            with self.assertRaisesRegex(pack_spine_sha.PackError, "magic"):
                pack_spine_sha.spine_digest(path, "glm5_next")
            good = Path(directory) / "good.sp"
            write_pack(good, spine() + experts(1), tp_rank=7)
            data = good.read_bytes()
            (Path(directory) / "short.sp").write_bytes(data[:-10])
            with self.assertRaisesRegex(pack_spine_sha.PackError, "file_bytes"):
                pack_spine_sha.spine_digest(Path(directory) / "short.sp", "glm5_next")
            result = subprocess.run([sys.executable, str(ROOT / "tools" / "pack_spine_sha.py"), "--layout", "glm5_next", str(good)],
                                    capture_output=True, text=True, check=True)
            self.assertIn("spine=3 experts=2 tp_rank=7", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=1)
