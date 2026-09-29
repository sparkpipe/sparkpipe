import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAYERS, FIRST_ROUTED, EXPERTS = 78, 3, 2
HEADER_BYTES, ENTRY_BYTES, DIRECTORY = 264, 64, 512
GLOBAL_KIND_BASE = 0x10000


def fixture(codec, encoding, scale_per_expert, global_bytes=0):
    routed = LAYERS - FIRST_ROUTED
    payload_per_expert = 32
    region = EXPERTS * (payload_per_expert + scale_per_expert) + global_bytes
    region = (region + 255) // 256 * 256
    data_start = DIRECTORY + routed * 2 * ENTRY_BYTES
    data_start = (data_start + 255) // 256 * 256
    size = data_start + routed * 2 * region
    data = bytearray(size)
    header = [0x32534C47, 3, HEADER_BYTES, ENTRY_BYTES, 1, 0, routed * 2, 1, 0, 0, LAYERS, LAYERS,
              6144, 154880, EXPERTS, 1, codec, 1, 0, 0]
    struct.pack_into("<20I2Q", data, 0, *header, DIRECTORY, size)
    cursor = data_start
    for index in range(routed * 2):
        layer = FIRST_ROUTED + index // 2
        kind = 22 + index % 2
        scale_bytes = EXPERTS * scale_per_expert + global_bytes
        scale_offset = cursor + EXPERTS * payload_per_expert if scale_bytes else 0
        struct.pack_into("<8I4Q", data, DIRECTORY + index * ENTRY_BYTES, kind, layer, 4, codec, encoding, EXPERTS, 1, 32,
                         cursor, EXPERTS * payload_per_expert, scale_offset, scale_bytes)
        for i in range(EXPERTS * payload_per_expert + scale_bytes):
            data[cursor + i] = (index * 31 + i) % 251
        cursor += region
    return data


def generate(binary, root, data):
    path = root / "pack.sp"
    output = root / "pack.sp.experts"
    if output.exists():
        output.unlink()
    path.write_bytes(data)
    result = subprocess.run([str(binary), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    raw = output.read_bytes()
    count = struct.unpack_from("<4I", raw)[2]
    return [struct.unpack_from("<4I2Q", raw, 16 + i * 48) for i in range(count)]


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        binary = root / "generator"
        subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "include"),
                        "-I", str(ROOT / "model-families/glm52/include"), str(ROOT / "tools/glm52_experts_manifest.c"),
                        str(ROOT / "runtime/spark_weightd_manifest.c"), str(ROOT / "src/spark_ck128.c"), "-o", str(binary)], check=True)
        routed = LAYERS - FIRST_ROUTED
        fp8 = generate(binary, root, fixture(5, 1, 8))
        assert len(fp8) == routed * 2 * EXPERTS * 2 and {r[2] for r in fp8} == {44, 45, 46, 47}
        bf16 = generate(binary, root, fixture(1, 0, 0))
        assert len(bf16) == routed * 2 * EXPERTS and {r[2] for r in bf16} == {44, 46}
        nvfp4 = generate(binary, root, fixture(6, 4, 4, EXPERTS * 4))
        assert len(nvfp4) == routed * 2 * EXPERTS * 3
        assert {r[2] for r in nvfp4} == {44, 45, 46, 47, GLOBAL_KIND_BASE + 22, GLOBAL_KIND_BASE + 23}
        layer3_up = {(r[1], r[2]): (r[4], r[5]) for r in nvfp4 if r[0] == 3 and r[2] in (45, GLOBAL_KIND_BASE + 22)}
        payload = {(r[1]): r[4] for r in nvfp4 if r[0] == 3 and r[2] == 44}
        scale = payload[0] + EXPERTS * 32
        assert layer3_up[(0, GLOBAL_KIND_BASE + 22)] == (scale, 4) and layer3_up[(1, GLOBAL_KIND_BASE + 22)] == (scale + 4, 4)
        assert layer3_up[(0, 45)] == (scale + 8, 4) and layer3_up[(1, 45)] == (scale + 12, 4)
        (root / "pack.sp.experts").unlink()
        (root / "pack.sp").write_bytes(fixture(6, 1, 4, EXPERTS * 4))
        result = subprocess.run([str(binary), str(root / "pack.sp")], capture_output=True, text=True)
        assert result.returncode == 1 and "error=-25" in result.stderr, result.stderr
    print("PASS glm52 v2 manifest: fp8/bf16/nvfp4 planes exactly once per expert, nvfp4 global and block addressing, encoding mismatch refused")


if __name__ == "__main__":
    main()
