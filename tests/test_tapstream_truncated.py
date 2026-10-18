import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "spec_offline"))

import tapstream  # noqa: E402

HIDDEN = 4
ENTRIES = [(1, 3), (3, 0)]
FINGERPRINT = 0x1234
ROW_BYTES = len(ENTRIES) * HIDDEN * 2


def hello_bytes():
    entries = b"".join(struct.pack("<HH", form, layer) for form, layer in ENTRIES).ljust(512, b"\0")
    return tapstream.HELLO.pack(tapstream.MAGIC, 1, 1, tapstream.HELLO.size, b"test", b"\0" * 32, HIDDEN, 8, 1, len(ENTRIES), 0, 16, FINGERPRINT, entries)


def frame_bytes(serial, first, rows):
    payload = rows * (4 + ROW_BYTES)
    header = tapstream.FRAME.pack(tapstream.MAGIC, 1, 2, tapstream.FRAME.size, FINGERPRINT, 1, 100, 1, 7, first, serial, rows, 1, 0, len(ENTRIES), payload)
    tokens = struct.pack(f"<{rows}I", *range(first, first + rows))
    data = bytes((serial * 31 + index) % 251 for index in range(rows * ROW_BYTES))
    return header + tokens + data


class TruncatedTapFileTest(unittest.TestCase):
    def write(self, tail):
        handle = tempfile.NamedTemporaryFile(suffix=".spt2", delete=False)
        handle.write(hello_bytes() + frame_bytes(0, 0, 3) + frame_bytes(1, 3, 2) + tail)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def check_two_frames(self, path):
        seen = [frame for _, frame in tapstream.frames(path)]
        self.assertEqual([frame["serial"] for frame in seen], [0, 1])
        size = os.path.getsize(path)
        for frame in seen:
            self.assertLessEqual(frame["offset"] + frame["rows"] * frame["row_bytes"], size)
        positions = sorted(frame["first_position"] + row for frame in seen for row in range(frame["rows"]))
        self.assertEqual(positions, [0, 1, 2, 3, 4])

    def test_complete_file_reads_every_frame(self):
        self.check_two_frames(self.write(b""))

    def test_cut_inside_a_frame_header_ends_the_stream(self):
        self.check_two_frames(self.write(frame_bytes(2, 5, 4)[:40]))

    def test_cut_inside_the_tokens_ends_the_stream(self):
        self.check_two_frames(self.write(frame_bytes(2, 5, 4)[:tapstream.FRAME.size + 6]))

    def test_cut_inside_the_rows_ends_the_stream(self):
        self.check_two_frames(self.write(frame_bytes(2, 5, 4)[:-3]))

    def test_corruption_before_the_end_still_fails(self):
        broken = bytearray(frame_bytes(2, 5, 4))
        broken[0] ^= 0xFF
        with self.assertRaises(ValueError):
            list(tapstream.frames(self.write(bytes(broken) + frame_bytes(3, 9, 1))))


if __name__ == "__main__":
    unittest.main()
