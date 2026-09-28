#!/usr/bin/env python3
import os
import shutil
import subprocess
import types
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from t1_reference_common import Safetensors, bf16_to_f32, f32_to_bf16_u16  # noqa: E402
from t1_reference_dsv41 import (Dsv41FlashEngine, _E4M3_F32, _fp4_qdq_e4m3_16,  # noqa: E402
                                _fp8_qdq_rows, engram_gate)

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


def check_candidates(workspace, rng):
    block_size, topk_blocks, stride = 8, 2048, 20000
    widths = np.array([20000, 16383, 5, 19997, 20000], dtype=np.uint32)
    scores = rng.standard_normal((widths.size, stride)).astype(np.float32)
    scores[1, 100:] = np.float32(0.5)
    scores[3, :8000] = -np.inf
    scores[4] = np.round(scores[4] * 4.0) / np.float32(4.0)
    for row, width in enumerate(widths):
        scores[row, width:] = np.float32(7.0)
    source, output = workspace / "candidates_in.bin", workspace / "candidates_out.bin"
    with open(source, "wb") as handle:
        handle.write(widths.tobytes())
        handle.write(scores.tobytes())
    run("candidates", source, output, widths.size, stride, block_size, topk_blocks)
    got = np.fromfile(output, dtype=np.float32).reshape(widths.size, stride)
    engine = types.SimpleNamespace(candidate_block=block_size, candidate_blocks=topk_blocks)
    for row, width in enumerate(widths):
        mask = Dsv41FlashEngine._select_candidate_blocks(engine, scores[row, :width].copy(), int(width))
        expected = np.where(mask, scores[row, :width], -np.inf).astype(np.float32)
        if not np.array_equal(got[row, :width], expected):
            differ = int((got[row, :width] != expected).sum())
            raise AssertionError(f"candidate blocks row {row} (width {width}): {differ} positions differ from the reference")
        if not np.array_equal(got[row, width:], scores[row, width:]):
            raise AssertionError(f"candidate blocks row {row}: positions beyond the width were written")
    return int(widths.sum())


def fp8_linear_case(workspace, name, weight, scale, activations):
    rows, input_dimension = activations.shape
    output_dimension = weight.shape[0]
    paths = {key: workspace / f"{name}_{key}.bin" for key in ("w", "s", "x", "y")}
    np.ascontiguousarray(weight, dtype=np.uint8).tofile(paths["w"])
    np.ascontiguousarray(scale, dtype=np.uint8).tofile(paths["s"])
    words = bf16(activations)
    words.tofile(paths["x"])
    run("fp8-linear", paths["w"], paths["s"], paths["x"], paths["y"], rows, input_dimension, output_dimension)
    got = np.fromfile(paths["y"], dtype=np.uint16).reshape(rows, output_dimension)
    block = np.exp2(scale.astype(np.float32) - 127.0)
    dequantized = _E4M3_F32[weight] * np.repeat(np.repeat(block, 32, axis=0), 32, axis=1)
    expected = bf16(_fp8_qdq_rows(bf16_to_f32(words)) @ dequantized.T)
    distance = ulp_distance(got, expected)
    worst = int(distance.max())
    exact = int((distance == 0).sum())
    if worst > 1 or exact < got.size * 0.98:
        raise AssertionError(f"fp8 block linear {name}: worst {worst} bf16 ulp, {exact}/{got.size} bit-equal")
    return got.size, exact


def check_fp8_linear(workspace, rng):
    total = exact = 0
    for name, (n, k) in {"random_1280x5120": (1280, 5120), "random_4096x1280": (4096, 1280)}.items():
        codes = rng.integers(0, 0x7E, size=(n, k), dtype=np.uint8) | (rng.integers(0, 2, size=(n, k), dtype=np.uint8) << 7)
        scale = rng.integers(118, 128, size=(n // 32, k // 32), dtype=np.uint8)
        size, equal = fp8_linear_case(workspace, name, codes, scale, rng.standard_normal((2, k)).astype(np.float32))
        total += size
        exact += equal
    checkpoint = os.environ.get("DSV41_CHECKPOINT")
    real = []
    if checkpoint:
        tensors = Safetensors(checkpoint)
        for tensor in ("layers.0.attn.wq_a", "layers.0.attn.wq_b", "layers.2.ffn.shared_experts.w2"):
            weight = tensors.pread(tensor + ".weight")
            scale = tensors.pread(tensor + ".scale")
            activations = rng.standard_normal((1, weight.shape[1])).astype(np.float32)
            size, equal = fp8_linear_case(workspace, tensor.replace(".", "_"), weight, scale, activations)
            total += size
            exact += equal
            real.append(tensor)
    return total, exact, real


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
        candidate_values = check_candidates(workspace, rng)
        linear_values, linear_exact, real = check_fp8_linear(workspace, rng)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    print(f"PASS dsv41 kernels: kv fp4 e4m3/16 qdq bit-equal to the reference on {fp4_values} values; "
          f"engram gate {engram_exact}/{engram_values} bit-equal, rest within 1 bf16 ulp; "
          f"candidate-block mask equal on {candidate_values} positions (top 2048 of 8-blocks, ties, -inf, pinned block); "
          f"fp8 32x32-block linear {linear_exact}/{linear_values} bit-equal, rest within 1 bf16 ulp"
          + (f" (real weights: {', '.join(real)})" if real else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
