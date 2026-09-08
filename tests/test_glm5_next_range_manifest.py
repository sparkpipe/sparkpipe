#!/usr/bin/env python3
"""Production C generator: complete planes, bounded framing and atomic publish."""
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def fixture(codec=5):
    data = bytearray(1280)
    header = [0x33584C47, 1, 264, 64, 1, 0, 2, 1, 0, 3, 1, 45,
              4096, 154880, 2, 1, codec, 0, 0, 0]
    struct.pack_into("<20I2Q", data, 0, *header, 512, len(data))
    for index, kind in enumerate((22, 23)):
        offset = 768 + index * 256
        struct.pack_into("<8I4Q", data, 512 + index * 64,
                         kind, 3, 4, codec, 1, 2, 1, 32, offset, 64, offset + 64, 16)
        data[offset:offset + 80] = bytes(range(index * 80, index * 80 + 80))
    return data


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binary = root / "generator"
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-I", str(ROOT / "include"), "-I", str(ROOT / "model-families/glm5_next/include"),
                        str(ROOT / "tools/glm5_next_experts_manifest.c"),
                        str(ROOT / "runtime/spark_weightd_manifest.c"), str(ROOT / "src/spark_ck128.c"),
                        "-o", str(binary)], check=True)
        path = root / "pack.sp"
        output = root / "pack.sp.experts"
        path.write_bytes(fixture())
        subprocess.run([str(binary), str(path)], check=True)
        original = output.read_bytes()
        assert struct.unpack_from("<4I", original) == (0x58504557, 2, 8, 0)
        records = [struct.unpack_from("<4I2Q16s", original, 16 + i * 48) for i in range(8)]
        for expert in range(2):
            ranges = {r[2]: (r[4], r[5]) for r in records if r[1] == expert}
            assert ranges == {44: (768 + expert * 32, 32), 45: (832 + expert * 8, 8),
                              46: (1024 + expert * 32, 32), 47: (1088 + expert * 8, 8)}
        assert subprocess.run([str(binary), str(path)], capture_output=True).returncode != 0
        assert output.read_bytes() == original
        output.unlink()
        malformed = [fixture()[:-1], fixture(codec=6)]
        missing_down = fixture()
        struct.pack_into("<I", missing_down, 576, 24)
        malformed.append(missing_down)
        missing_layer = fixture()
        struct.pack_into("<I", missing_layer, 40, 2)
        malformed.append(missing_layer)
        wrong_scale_encoding = fixture()
        struct.pack_into("<I", wrong_scale_encoding, 512 + 16, 4)
        malformed.append(wrong_scale_encoding)
        malformed.append(fixture(codec=1))  # BF16 must not carry FP8 scales.
        for data in malformed:
            path.write_bytes(data)
            assert subprocess.run([str(binary), str(path)], capture_output=True).returncode != 0
            assert not output.exists() and not list(root.glob("*.partial.*"))
        print("PASS GLM v2 manifest: both weights/scales, malformed rejection, existing-output preservation")


if __name__ == "__main__":
    main()
