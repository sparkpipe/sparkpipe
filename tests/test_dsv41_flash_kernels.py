#!/usr/bin/env python3
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from t1_reference_common import bf16_to_f32, f32_to_bf16_u16  # noqa: E402
from t1_reference_dsv41 import _fp4_qdq_e4m3_16, engram_gate  # noqa: E402

BINARY = ROOT / "build/test_dsv41_flash_kernels"
HIDDEN = 5120
HC = 4
HEAD_DIM = 512
EPSILON = 1e-20


def bf16(values):
    return f32_to_bf16_u16(values.astype(np.float32))


def run(*argv):
    result = subprocess.run([str(BINARY), *[str(item) for item in argv]], capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)


def ordered(words):
    magnitude = (words & 0x7FFF).astype(np.int64)
    return np.where(words & 0x8000, -magnitude, magnitude)


def ulp_distance(a, b):
    return np.abs(ordered(a) - ordered(b))


def check_kv_fp4(workspace, rng):
    rows = 7
    values = rng.standard_normal((rows, HEAD_DIM)).astype(np.float32) * np.float32(3.0)
    values[1] *= np.float32(1e-4)
    values[2] = 0.0
    values[3, :16] = np.float32(1e3)
    words = bf16(values)
    source, output = workspace / "kv_in.bin", workspace / "kv_out.bin"
    words.tofile(source)
    run("kv-fp4", source, output, rows, HEAD_DIM)
    got = np.fromfile(output, dtype=np.uint16).reshape(rows, HEAD_DIM)
    expected = bf16(_fp4_qdq_e4m3_16(bf16_to_f32(words)))
    mismatches = int((got != expected).sum())
    if mismatches:
        index = np.argwhere(got != expected)[0]
        raise AssertionError(f"kv fp4 e4m3/16 qdq: {mismatches} of {got.size} values differ, first at {tuple(index)}: "
                             f"cuda {bf16_to_f32(got[tuple(index)])} reference {bf16_to_f32(expected[tuple(index)])}")
    return got.size


def check_engram(workspace, rng):
    rows = 3
    streams = bf16(rng.standard_normal((rows, HC, HIDDEN)) * 4.0)
    kv = bf16(rng.standard_normal((rows, (HC + 1) * HIDDEN)) * 0.5)
    q_weight = bf16(1.0 + rng.standard_normal((HC, HIDDEN)) * 0.1)
    k_weight = bf16(1.0 + rng.standard_normal((HC, HIDDEN)) * 0.1)
    kv[1, :HC * HIDDEN] = bf16(-bf16_to_f32(streams[1]).reshape(-1))
    paths = {name: workspace / f"engram_{name}.bin" for name in ("streams", "kv", "q", "k", "out")}
    streams.tofile(paths["streams"])
    kv.tofile(paths["kv"])
    q_weight.tofile(paths["q"])
    k_weight.tofile(paths["k"])
    run("engram", paths["streams"], paths["kv"], paths["q"], paths["k"], paths["out"], rows, HC, HIDDEN, repr(EPSILON))
    got = np.fromfile(paths["out"], dtype=np.uint16).reshape(rows, HC, HIDDEN)
    weight = bf16_to_f32(q_weight) * bf16_to_f32(k_weight)
    worst = 0
    exact = 0
    for row in range(rows):
        key = bf16_to_f32(kv[row, :HC * HIDDEN]).reshape(HC, HIDDEN)
        value = bf16_to_f32(kv[row, HC * HIDDEN:])
        expected = bf16(engram_gate(bf16_to_f32(streams[row]), key, value, weight, EPSILON))
        distance = ulp_distance(got[row], expected)
        worst = max(worst, int(distance.max()))
        exact += int((distance == 0).sum())
    if worst > 1:
        raise AssertionError(f"engram gate differs from the reference by {worst} bf16 ulp")
    if exact < got.size * 0.99:
        raise AssertionError(f"engram gate: only {exact} of {got.size} values bit-equal to the reference")
    return got.size, exact


def main():
    built = subprocess.run(["make", "-s", str(BINARY.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    if built.returncode != 0:
        print(built.stdout + built.stderr)
        return 1
    if not BINARY.is_file():
        print("SKIP test_dsv41_flash_kernels (nvcc unavailable on this host)")
        return 0
    if shutil.which("nvidia-smi") is None:
        print("SKIP test_dsv41_flash_kernels (no GPU on this host)")
        return 0
    rng = np.random.default_rng(20260928)
    workspace = Path(tempfile.mkdtemp(prefix="dsv41-kernels-"))
    try:
        fp4_values = check_kv_fp4(workspace, rng)
        engram_values, engram_exact = check_engram(workspace, rng)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    print(f"PASS dsv41 kernels: kv fp4 e4m3/16 qdq bit-equal to the reference on {fp4_values} values; "
          f"engram gate {engram_exact}/{engram_values} bit-equal, rest within 1 bf16 ulp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
