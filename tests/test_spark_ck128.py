#!/usr/bin/env python3
"""tools/spark_ck128.py computes ck128 through src/spark_ck128.c with the digests the pack tools' pure
Python ck128 produced before it (recorded here at every tail length class), for bytes, bytearray and memoryview
input and streamed in pieces, and its library builds from the Makefile target."""
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

EXPECTED = {
    0: "00000000000000000000000000000000",
    1: "897859f6655555855a890e51483ab5e6",
    15: "e92549fd98152347e97dc688ee6d84cd",
    16: "303f9091b524494445e82f76566490ab",
    17: "0ec2e79f0ff4765c24a8da9e6b025fc1",
    33: "1246bafa1b28417d0ba3d6a77380ac55",
    65536: "3ed04282b92d148a9461e4755f915815",
}


def build_library() -> None:
    extension = "dylib" if sys.platform == "darwin" else "so"
    subprocess.run(["make", "-s", f"build/libsparkpipe_ck128.{extension}"], cwd=ROOT, check=True)


def cases():
    generator = random.Random(1108)
    return {0: b"", 1: b"a", 15: bytes(range(15)), 16: bytes(range(16)), 17: bytes(range(17)), 33: bytes(range(33)),
            65536: bytes(generator.getrandbits(8) for _ in range(1 << 16))}


def main() -> int:
    build_library()
    import spark_ck128
    failures = []
    for length, data in cases().items():
        for label, value in (("bytes", data), ("bytearray", bytearray(data)), ("memoryview", memoryview(data))):
            got = spark_ck128.ck128(value).hex()
            if got != EXPECTED[length]:
                failures.append(f"{label} of {length} bytes: {got} != {EXPECTED[length]}")
        state = spark_ck128.Ck128()
        for begin in range(0, len(data), 7):
            state.update(data[begin:begin + 7])
        if state.digest().hex() != EXPECTED[length]:
            failures.append(f"{length} bytes streamed in 7-byte pieces: {state.digest().hex()}")
    for failure in failures:
        print("FAIL spark_ck128:", failure)
    if failures:
        return 1
    print("PASS spark_ck128: the C ck128 gives the recorded digests for bytes, bytearray, memoryview and streamed input at every tail length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
