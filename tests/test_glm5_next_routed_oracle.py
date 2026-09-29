import importlib.util
import struct
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oracle", ROOT / "tools/glm5_next_routed_oracle.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def synthetic_pack(path, codec, encoding, payload, scale, cols=32):
    rows, groups = 2, 2
    header = [0x33584C47, 1, 264, 64, 1, 0, 2, 1, 0, 3, 1, 45, 4096, 154880, groups, 1, codec, 1, 0, 0]
    data = bytearray(8192)
    struct.pack_into("<20I2Q", data, 0, *header, 512, len(data))
    offset = 1024
    for index, kind in enumerate((oracle.EXPERT_UP_GATE, oracle.EXPERT_DOWN)):
        scale_offset = offset + len(payload) if scale else 0
        struct.pack_into("<8I4Q", data, 512 + index * 64, kind, 3, 4, codec, encoding, groups, rows, cols,
                         offset, len(payload), scale_offset, len(scale))
        data[offset:offset + len(payload) + len(scale)] = payload + scale
        offset += 2048
    Path(path).write_bytes(data)
    return oracle.Pack(path)


def main():
    assert oracle.E4M3[0x38] == 1.0 and oracle.E4M3[0x7E] == 448.0 and oracle.E4M3[0x01] == 2.0 ** -9
    assert np.isnan(oracle.E4M3[0x7F]) and oracle.E4M3[0xB8] == -1.0
    assert list(oracle.E2M1[:8]) == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] and oracle.E2M1[0xA] == -1.0
    assert oracle.bf16_round(np.array([1.0 + 2.0 ** -8]))[0] == 1.0
    assert oracle.bf16_round(np.array([1.0 + 3 * 2.0 ** -8]))[0] == 1.0 + 2.0 ** -6
    with tempfile.TemporaryDirectory() as directory:
        codes = bytes([0x21] * 32 + [0x1A] * 32)
        blocks = bytes([0x38, 0x40] * 2 + [0x30] * 4)
        globals_ = struct.pack("<2f", 0.5, 4.0)
        pack = synthetic_pack(Path(directory) / "nvfp4.sp", 6, 4, codes, globals_ + blocks)
        w0, codec = pack.weight(3, oracle.EXPERT_UP_GATE, 0)
        w1, _ = pack.weight(3, oracle.EXPERT_UP_GATE, 1)
        assert codec == 6 and w0.shape == (2, 32)
        assert w0[0, 0] == 0.5 * 1.0 * 0.5 and w0[0, 1] == 1.0 * 1.0 * 0.5 and w0[0, 16] == 0.5 * 2.0 * 0.5
        assert w1[0, 0] == -1.0 * 0.5 * 4.0 and w1[0, 1] == 0.5 * 0.5 * 4.0
        fp8 = synthetic_pack(Path(directory) / "fp8.sp", 5, 1, bytes([0x38] * 512), struct.pack("<4f", 1.0, 1.0, 0.25, 3.0), cols=128)
        w, _ = fp8.weight(3, oracle.EXPERT_DOWN, 1)
        assert w[0, 0] == 0.25 and w[1, 5] == 3.0
    print("PASS routed oracle: e4m3/e2m1 tables, bf16 RNE, nvfp4 global + block addressing, fp8 block scales")


if __name__ == "__main__":
    main()
