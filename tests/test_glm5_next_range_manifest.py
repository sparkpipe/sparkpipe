#!/usr/bin/env python3
"""Production C generator: codec planes, bounded framing and atomic publish."""
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


GLOBAL_KIND_BASE = 0x10000


def fixture(codec=5, encoding=1, scale_bytes=16):
    data = bytearray(1280)
    header = [0x33584C47, 1, 264, 64, 1, 0, 2, 1, 0, 3, 1, 45,
              4096, 154880, 2, 1, codec, 0, 0, 0]
    struct.pack_into("<20I2Q", data, 0, *header, 512, len(data))
    for index, kind in enumerate((22, 23)):
        offset = 768 + index * 256
        scale_offset = offset + 64 if scale_bytes else 0
        struct.pack_into("<8I4Q", data, 512 + index * 64,
                         kind, 3, 4, codec, encoding, 2, 1, 32, offset, 64, scale_offset, scale_bytes)
        data[offset:offset + 64 + scale_bytes] = bytes((index * 97 + i) % 256 for i in range(64 + scale_bytes))
    return data


def records_of(output):
    words = struct.unpack_from("<4I", output)
    assert words[:2] == (0x58504557, 2) and words[3] == 0, words
    assert len(output) == 16 + 48 * words[2], (len(output), words)
    return [struct.unpack_from("<4I2Q16s", output, 16 + i * 48) for i in range(words[2])]


def planes_per_expert(binary, root, data):
    path = root / "codec.sp"
    output = root / "codec.sp.experts"
    path.write_bytes(data)
    subprocess.run([str(binary), str(path)], check=True, capture_output=True)
    records = records_of(output.read_bytes())
    checked = subprocess.run([str(binary), "--check", str(path)], capture_output=True, text=True)
    assert checked.returncode == 0, checked
    output.unlink()
    result = {}
    for layer, expert, kind, zero, offset, size, _ in records:
        assert layer == 3 and zero == 0
        assert (expert, kind) not in result, (expert, kind)
        result[(expert, kind)] = (offset, size)
    return result


def check_codec_planes(binary, root):
    bf16 = planes_per_expert(binary, root, fixture(codec=1, encoding=0, scale_bytes=0))
    assert bf16 == {(e, 44 + 2 * t): (768 + t * 256 + e * 32, 32) for e in range(2) for t in range(2)}, bf16
    nvfp4 = planes_per_expert(binary, root, fixture(codec=6, encoding=4, scale_bytes=24))
    expected = {}
    for t in range(2):
        payload = 768 + t * 256
        for e in range(2):
            expected[(e, 44 + 2 * t)] = (payload + e * 32, 32)
            expected[(e, 45 + 2 * t)] = (payload + 64 + 8 + e * 8, 8)
            expected[(e, GLOBAL_KIND_BASE + 22 + t)] = (payload + 64 + e * 4, 4)
    assert nvfp4 == expected, nvfp4
    for t in range(2):
        payload = 768 + t * 256
        covered = sorted(v for (e, k), v in nvfp4.items() if k in (45 + 2 * t, GLOBAL_KIND_BASE + 22 + t))
        assert sum(size for _, size in covered) == 24
        cursor = payload + 64
        for offset, size in covered:
            assert offset == cursor, (covered, cursor)
            cursor += size
    mxfp4 = planes_per_expert(binary, root, fixture(codec=7, encoding=3, scale_bytes=16))
    assert set(k for _, k in mxfp4) == {44, 45, 46, 47} and len(mxfp4) == 8
    int8 = planes_per_expert(binary, root, fixture(codec=4, encoding=1, scale_bytes=16))
    assert set(k for _, k in int8) == {44, 45, 46, 47} and len(int8) == 8


def rejected(binary, path, error):
    result = subprocess.run([str(binary), str(path)], capture_output=True)
    assert result.returncode == 1, (error, result.returncode, result.stderr)
    assert result.stderr.startswith(b"expert manifest failed: error=%d " % error), (error, result.stderr)


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
        checked = subprocess.run([str(binary), "--check", str(path)], capture_output=True, text=True)
        assert checked.returncode == 0 and checked.stdout.startswith("checked "), checked
        tampered = bytearray(original)
        struct.pack_into("<I", tampered, 16 + 1 * 48 + 8, 999)
        output.write_bytes(tampered)
        checked = subprocess.run([str(binary), "--check", str(path)], capture_output=True, text=True)
        assert checked.returncode == 1 and "error=-28" in checked.stderr, checked
        output.write_bytes(original)
        rejected(binary, path, -22)
        assert output.read_bytes() == original
        output.unlink()
        check_codec_planes(binary, root)
        malformed = [(fixture()[:-1], -13), (fixture(codec=6), -25), (fixture(codec=9), -4),
                     (fixture(codec=6, encoding=4, scale_bytes=8), -6),
                     (fixture(codec=6, encoding=4, scale_bytes=25), -6),
                     (fixture(codec=5, encoding=1, scale_bytes=0), -6)]
        missing_down = fixture()
        struct.pack_into("<I", missing_down, 576, 24)
        malformed.append((missing_down, -10))
        missing_layer = fixture()
        struct.pack_into("<I", missing_layer, 40, 2)
        malformed.append((missing_layer, -10))
        wrong_scale_encoding = fixture()
        struct.pack_into("<I", wrong_scale_encoding, 512 + 16, 4)
        malformed.append((wrong_scale_encoding, -25))
        malformed.append((fixture(codec=1), -24))
        for data, error in malformed:
            path.write_bytes(data)
            rejected(binary, path, error)
            assert not output.exists() and not list(root.glob("*.partial.*"))
        print("PASS GLM v2 manifest: fp8/bf16/nvfp4/mxfp4/int8 planes exactly once per expert, malformed rejection, existing-output preservation")


if __name__ == "__main__":
    main()
